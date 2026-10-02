"""Small optional diagnostic files, independent of durable engine state."""
from __future__ import annotations

import json
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread

MAX_BYTES = 1024 * 1024
BACKUP_COUNT = 2
MAX_MESSAGE = 2048
QUEUE_LIMIT = 256


class _DiagnosticFileHandler(RotatingFileHandler):
    def handleError(self, record):
        # A full disk must not create an unbounded second log on stderr.
        pass


class _QueueHandler(logging.Handler):
    def __init__(self, pending: Queue):
        super().__init__()
        self.pending = pending

    def emit(self, record):
        try:
            value = {
                'at': record.created, 'pid': record.process, 'level': record.levelname,
                'logger': record.name[:128],
                'message': record.getMessage()[:MAX_MESSAGE],
            }
            if hasattr(record, 'diagnostic_event'):
                value['event'] = str(record.diagnostic_event)[:128]
                data = record.diagnostic_data
                encoded = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
                value['data'] = data if len(encoded) <= MAX_MESSAGE else {'truncated': True, 'preview': encoded[:MAX_MESSAGE]}
            if record.exc_info and record.exc_info[1] is not None:
                value['exception'] = type(record.exc_info[1]).__name__
            compact = logging.LogRecord(record.name, record.levelno, '', 0,
                                        json.dumps(value, ensure_ascii=True), (), None)
            self.pending.put_nowait(compact)
        except Exception:
            # Diagnostics neither apply effects nor grant recovery authority.
            # Full queues and malformed diagnostic messages may be discarded.
            pass


def start_diagnostic_logging(directory: Path):
    """Return an idempotent stop callback; failed logging never stops the engine."""
    handler = None
    try:
        directory = Path(directory)
        if directory.is_symlink():
            return lambda: None
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = directory / 'native.log'
        if any(candidate.is_symlink() for candidate in (path, *(directory / f'native.log.{i}' for i in range(1, BACKUP_COUNT+1)))):
            return lambda: None
        handler = _DiagnosticFileHandler(path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding='utf-8')
        os.chmod(path, 0o600)
        handler.setFormatter(logging.Formatter('%(message)s'))
    except OSError:
        if handler is not None:
            handler.close()
        return lambda: None

    pending = Queue(maxsize=QUEUE_LIMIT)
    stopped = Event()
    logger = logging.getLogger('newsroom')
    previous = (list(logger.handlers), logger.level, logger.propagate)
    queued = _QueueHandler(pending)

    def consume():
        try:
            while not stopped.is_set() or not pending.empty():
                try:
                    record = pending.get(timeout=0.05)
                except Empty:
                    continue
                try:
                    handler.emit(record)
                except Exception:
                    pass
                finally:
                    pending.task_done()
        finally:
            handler.close()

    worker = Thread(target=consume, name='newsroom-diagnostics', daemon=True)
    try:
        worker.start()
    except Exception:
        handler.close()
        return lambda: None
    logger.handlers = [queued]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    def stop():
        if stopped.is_set():
            return
        logger.handlers, logger.level, logger.propagate = previous
        stopped.set()
        # Slow or broken diagnostic storage must not delay process recovery.
        worker.join(timeout=0.2)
        queued.close()

    return stop


def emit_diagnostic(event: str, data: dict) -> None:
    """Best-effort observation; no consumer may use it as work authority."""
    try:
        logging.getLogger('newsroom.diagnostic').info(
            event, extra={'diagnostic_event': event, 'diagnostic_data': data},
        )
    except Exception:
        pass
