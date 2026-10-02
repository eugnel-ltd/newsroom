from __future__ import annotations

import json
import logging
import threading

import pytest

from newsroom.control_plane import diagnostic_logging as diagnostics


def test_bounded_logs_are_compact_and_rotated(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, 'MAX_BYTES', 4096)
    monkeypatch.setattr(diagnostics, 'BACKUP_COUNT', 2)
    stop = diagnostics.start_diagnostic_logging(tmp_path)
    for index in range(40):
        logging.getLogger('newsroom.fixture').warning('record %s %s', index, 'x' * 10000)
    stop()
    paths = list(tmp_path.glob('native.log*'))
    assert 1 <= len(paths) <= 3
    assert sum(path.stat().st_size for path in paths) <= 3 * 4096
    for path in paths:
        for line in path.read_text().splitlines():
            value = json.loads(line)
            assert len(value['message']) <= diagnostics.MAX_MESSAGE
            assert value['logger'] == 'newsroom.fixture'


def test_slow_diagnostic_io_does_not_block_work_and_queue_is_bounded(tmp_path, monkeypatch):
    entered, release, progressed = threading.Event(), threading.Event(), threading.Event()
    queues = []
    queue_factory = diagnostics.Queue

    def tracked_queue(*args, **kwargs):
        queue = queue_factory(*args, **kwargs)
        queues.append(queue)
        return queue

    monkeypatch.setattr(diagnostics, 'Queue', tracked_queue)
    original = diagnostics.RotatingFileHandler.emit

    def slow(self, record):
        entered.set()
        release.wait(3)
        original(self, record)

    monkeypatch.setattr(diagnostics.RotatingFileHandler, 'emit', slow)
    stop = diagnostics.start_diagnostic_logging(tmp_path)
    logger = logging.getLogger('newsroom.fixture')
    logger.warning('first')
    assert entered.wait(1)

    def work():
        for _ in range(diagnostics.QUEUE_LIMIT * 3):
            logger.warning('diagnostic')
        progressed.set()

    worker = threading.Thread(target=work)
    worker.start()
    try:
        assert progressed.wait(1)
        assert queues[0].maxsize == diagnostics.QUEUE_LIMIT
        assert queues[0].qsize() <= diagnostics.QUEUE_LIMIT
    finally:
        release.set()
        worker.join(2)
        stop()


def test_unwritable_diagnostics_do_not_prevent_work(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError('fixture disk full')

    monkeypatch.setattr(diagnostics, '_DiagnosticFileHandler', unavailable)
    stop = diagnostics.start_diagnostic_logging(tmp_path)
    logging.getLogger('newsroom.fixture').warning('work continues')
    stop()


def test_logger_settings_are_restored_and_io_failure_stays_optional(tmp_path, monkeypatch):
    logger = logging.getLogger('newsroom')
    before = (list(logger.handlers), logger.level, logger.propagate)
    monkeypatch.setattr(diagnostics.RotatingFileHandler, 'emit', lambda *_: (_ for _ in ()).throw(OSError('full')))
    stop = diagnostics.start_diagnostic_logging(tmp_path)
    logging.getLogger('newsroom.fixture').warning('optional failure')
    stop()
    assert (logger.handlers, logger.level, logger.propagate) == before


@pytest.mark.parametrize('failure', ('open', 'write'))
def test_diagnostic_storage_failure_does_not_change_service_outcome(tmp_path, monkeypatch, failure):
    from newsroom.tests.test_native_service import _pipeline, _service
    from newsroom.control_plane.native_pipeline import NativePipelineReport

    def unavailable(*args, **kwargs):
        raise OSError('fixture diagnostic storage unavailable')

    if failure == 'open':
        monkeypatch.setattr(diagnostics, '_DiagnosticFileHandler', unavailable)
    else:
        monkeypatch.setattr(diagnostics.RotatingFileHandler, 'emit', unavailable)
    factory, opened = _pipeline(tmp_path, monkeypatch, lambda _: NativePipelineReport((), {}, 0))
    result = _service(tmp_path, factory).run(once=True)
    assert result.outcome == 'COMPLETE'
    assert opened == ['open', 'close']


@pytest.mark.parametrize('failed',[False,True])
@pytest.mark.parametrize('generation',[None,'fixture'])
def test_authority_projection_phase_is_completed_and_drop_safe(monkeypatch,caplog,failed,generation):
    from newsroom.authority import _increment4_projection_store as module
    clock=iter((1_000_000,124_000_000));cpu=iter((2_000_000,36_000_000))
    monkeypatch.setattr(module,'perf_counter_ns',lambda:next(clock))
    monkeypatch.setattr(module,'process_time_ns',lambda:next(cpu))
    caplog.set_level(logging.INFO,logger='newsroom.authority.projection')
    if failed:
        with pytest.raises(RuntimeError,match='original failure'):
            with module._projection_phase('CURRENT_STATE',generation_id=generation):
                raise RuntimeError('original failure')
    else:
        with module._projection_phase('CURRENT_STATE',generation_id=generation):pass
    text=caplog.text
    assert 'elapsed_ms=123' in text and 'cpu_ms=34' in text
    assert 'cpu_scope=PROCESS' in text and f'generation_id={generation}' in text
    assert ('status=FAILED' if failed else 'status=COMPLETE') in text
    monkeypatch.setattr(module._PHASE_LOG,'info',lambda *_a,**_kw:(_ for _ in ()).throw(OSError('dropped')))
    monkeypatch.setattr(module,'perf_counter_ns',lambda:0);monkeypatch.setattr(module,'process_time_ns',lambda:0)
    with pytest.raises(RuntimeError,match='original failure'):
        with module._projection_phase('CURRENT_STATE',generation_id=generation):
            raise RuntimeError('original failure')
