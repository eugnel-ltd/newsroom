"""Keychain broker for private-beta credentials. Never returns secret bytes to logs."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Callable, Final, Protocol

from newsroom.control_plane.child_environment import unprivileged_child_environment
from newsroom.graphiti_adapter.evaluation_packet import OPENROUTER_BASE_URL
from newsroom.graphiti_adapter.types import BROKER_SETUP_FAILURE_REASON_CODES

if TYPE_CHECKING:
    from newsroom.projection.neo4j.models import Neo4jProjectorConfig

OPENROUTER_ACCOUNT = "OPENROUTER_API"
OPENROUTER_SERVICE = "newsroom.shadow.v1"
NEO4J_ACCOUNT = "newsroom"
NEO4J_SERVICE = "NEO4J_COMMUNITY_LOCAL"
NEO4J_BOLT_HOST = "127.0.0.1"
NEO4J_BOLT_PORT = 7687
NEO4J_PROJECTOR_ACCOUNT = "newsroom_projector"
NEO4J_PROJECTOR_SERVICE = "NEWSROOM_NEO4J_PROJECTOR_LOCAL"
NEO4J_PROJECTOR_USERNAME = "newsroom_projector"
NEO4J_DATABASE = "neo4j"
_MIN_SECRET_CHARS = 20
OPENROUTER_KEYCHAIN_SKIP: Final[str] = "OPENROUTER_API Keychain class not on this host"
NEO4J_KEYCHAIN_SKIP: Final[str] = (
    "NEO4J_COMMUNITY_LOCAL Keychain class not on this host"
)
NEO4J_PROJECTOR_KEYCHAIN_SKIP: Final[str] = (
    "NEWSROOM_NEO4J_PROJECTOR_LOCAL Keychain class not on this host"
)


class BrokerError(RuntimeError):
    """Credential injection failed closed."""

    def __init__(self, message: str, *, reason_code: str | None = None) -> None:
        super().__init__(message)
        if reason_code is not None and reason_code not in BROKER_SETUP_FAILURE_REASON_CODES:
            raise ValueError("broker setup failure reason is not allow-listed")
        self.reason_code = reason_code


class _Neo4jRecord(Protocol):
    def __getitem__(self, key: str) -> object: ...


class _Neo4jResult(Protocol):
    def single(self) -> _Neo4jRecord | None: ...


class _Neo4jSession(Protocol):
    def __enter__(self) -> _Neo4jSession: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None: ...

    def run(self, query: str) -> _Neo4jResult: ...


class Neo4jProbeDriver(Protocol):
    def session(self) -> _Neo4jSession: ...

    def close(self) -> None: ...


Neo4jDriverFactory = Callable[..., Neo4jProbeDriver]


def keychain_present(*, account: str, service: str) -> bool:
    if shutil.which("security") is None:
        return False
    result = subprocess.run(
        ["security", "find-generic-password", "-a", account, "-s", service],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        env=unprivileged_child_environment(),
    )
    return result.returncode == 0


def _keychain_failure_code(account: str, service: str, failure: str) -> str | None:
    credential = {
        (OPENROUTER_ACCOUNT, OPENROUTER_SERVICE): "OPENROUTER",
        (NEO4J_ACCOUNT, NEO4J_SERVICE): "NEO4J_COMMUNITY",
        (NEO4J_PROJECTOR_ACCOUNT, NEO4J_PROJECTOR_SERVICE): "NEO4J_PROJECTOR",
    }.get((account, service))
    return None if credential is None else f"BROKER_{credential}_{failure}"


def _keychain_password(*, account: str, service: str) -> str:
    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-a",
                account,
                "-s",
                service,
                "-w",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env=unprivileged_child_environment(),
        )
    except subprocess.TimeoutExpired:
        raise BrokerError(f"Keychain class {account} lookup timed out", reason_code=_keychain_failure_code(account, service, "LOOKUP_TIMEOUT")) from None
    except OSError:
        raise BrokerError(f"Keychain class {account} lookup failed", reason_code=_keychain_failure_code(account, service, "LOOKUP_FAILED")) from None
    except UnicodeError:
        raise BrokerError(f"Keychain class {account} lookup failed", reason_code=_keychain_failure_code(account, service, "LOOKUP_FAILED")) from None
    if result.returncode != 0:
        raise BrokerError(f"Keychain class {account} lookup unavailable", reason_code=_keychain_failure_code(account, service, "LOOKUP_UNAVAILABLE"))
    secret = result.stdout.strip()
    if len(secret) < _MIN_SECRET_CHARS:
        raise BrokerError(f"Keychain class {account} is empty", reason_code=_keychain_failure_code(account, service, "EMPTY_OR_TOO_SHORT"))
    return secret


def _secret_shape(secret: str) -> str:
    if len(secret) < _MIN_SECRET_CHARS:
        return "too_short"
    if secret.startswith("sk-"):
        return "ok"
    return "bad_prefix"


def openrouter_api_key() -> str:
    return _keychain_password(account=OPENROUTER_ACCOUNT, service=OPENROUTER_SERVICE)


def neo4j_community_password() -> str:
    return _keychain_password(account=NEO4J_ACCOUNT, service=NEO4J_SERVICE)


def neo4j_projector_password() -> str:
    return _keychain_password(
        account=NEO4J_PROJECTOR_ACCOUNT,
        service=NEO4J_PROJECTOR_SERVICE,
    )


def openrouter_keychain_ready() -> bool:
    return keychain_present(account=OPENROUTER_ACCOUNT, service=OPENROUTER_SERVICE)


def neo4j_keychain_ready() -> bool:
    return keychain_present(account=NEO4J_ACCOUNT, service=NEO4J_SERVICE)


def neo4j_projector_keychain_ready() -> bool:
    return keychain_present(
        account=NEO4J_PROJECTOR_ACCOUNT,
        service=NEO4J_PROJECTOR_SERVICE,
    )


def neo4j_projector_config() -> Neo4jProjectorConfig:
    """Load the dedicated non-administrator projector configuration."""
    from newsroom.projection.neo4j.models import Neo4jProjectorConfig

    return Neo4jProjectorConfig(
        uri=f"bolt://{NEO4J_BOLT_HOST}:{NEO4J_BOLT_PORT}",
        database=NEO4J_DATABASE,
        username=NEO4J_PROJECTOR_USERNAME,
        password=neo4j_projector_password(),
    )


def prove_openrouter_keychain() -> None:
    """Inject OPENROUTER_API and confirm OpenRouter accepts it. Never logs the secret."""
    secret = openrouter_api_key()
    if _secret_shape(secret) != "ok":
        raise BrokerError("OPENROUTER_API Keychain secret has an invalid shape")
    request = urllib.request.Request(
        f"{OPENROUTER_BASE_URL}/key",
        method="GET",
        headers={"Authorization": f"Bearer {secret}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            status = response.status
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise BrokerError(
            f"OpenRouter rejected Keychain OPENROUTER_API HTTP {exc.code}"
        ) from None
    except urllib.error.URLError as exc:
        raise BrokerError(
            "OpenRouter Keychain probe could not reach openrouter.ai"
        ) from exc
    if status != 200 or not isinstance(payload, dict):
        raise BrokerError("OpenRouter Keychain probe returned a malformed body")


def prove_neo4j_keychain(*, driver_factory: Neo4jDriverFactory) -> None:
    """Inject NEO4J_COMMUNITY_LOCAL and confirm Bolt accepts it. Never logs the secret."""
    password = neo4j_community_password()
    try:
        socket.create_connection((NEO4J_BOLT_HOST, NEO4J_BOLT_PORT), timeout=3).close()
    except OSError as exc:
        raise BrokerError("Neo4j Bolt is not listening on 127.0.0.1:7687") from exc
    driver = driver_factory(
        f"bolt://{NEO4J_BOLT_HOST}:{NEO4J_BOLT_PORT}",
        auth=("neo4j", password),
    )
    try:
        with driver.session() as session:
            value = session.run("RETURN 1 AS n").single()
            if value is None or int(value["n"]) != 1:
                raise BrokerError("Neo4j Keychain probe query failed")
    except Exception as exc:
        if isinstance(exc, BrokerError):
            raise
        raise BrokerError("Neo4j rejected Keychain NEO4J_COMMUNITY_LOCAL") from None
    finally:
        driver.close()
