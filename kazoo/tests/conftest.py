import logging

from typing import Any

log = logging.getLogger(__name__)


def pytest_exception_interact(node: Any, call: Any, report: Any) -> None:
    testcase = getattr(node, "_testcase", None)
    cluster = getattr(testcase, "cluster", None)
    if cluster is None:
        return
    try:
        log.error("Zookeeper cluster logs:")
        for logs in cluster.get_logs():
            log.error(logs)
    except Exception:
        log.exception("Cannot get ZK logs:")
