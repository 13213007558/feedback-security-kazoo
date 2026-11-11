"""Unit tests for TransactionResult and the multi-response decoding path.

These tests use fake, hand-built server responses only; no Zookeeper
server is required. They follow a transaction from the serialized
multi response bytes through ``Transaction.deserialize`` and the
public ``TransactionRequest.commit`` entry point.
"""

from __future__ import annotations

import struct

import pytest

from kazoo.client import KazooClient, TransactionResult, TransactionStatus
from kazoo.exceptions import (
    BadVersionError,
    ConnectionLoss,
    MarshallingError,
    NoAuthError,
    NoNodeError,
    RolledBackError,
    SessionExpiredError,
)
from kazoo.protocol.serialization import (
    CheckVersion,
    Create,
    Delete,
    MultiHeader,
    SetData,
    Transaction,
    int_struct,
    stat_struct,
    write_string,
)
from kazoo.protocol.states import ZnodeStat

# Zookeeper error codes, see kazoo.exceptions
ROLLED_BACK = 0
NO_NODE = -101
NO_AUTH = -102
BAD_VERSION = -103

STAT_VALUES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
DONE = MultiHeader(-1, True, -1).serialize()


def _header(op_type: int) -> bytes:
    return bytes(MultiHeader(op_type, False, -1).serialize())


def _error_entry(err_code: int) -> bytes:
    return _header(-1) + int_struct.pack(err_code)


def _create_entry(path: str) -> bytes:
    return _header(Create.type) + write_string(path)


def _delete_entry() -> bytes:
    return _header(Delete.type)


def _set_data_entry() -> bytes:
    return _header(SetData.type) + stat_struct.pack(*STAT_VALUES)


def _check_entry() -> bytes:
    return _header(CheckVersion.type)


class FakeConnection:
    """Stands in for ``KazooClient._call`` + the connection reader.

    Replays the exact response handling of
    :meth:`kazoo.protocol.connection.ConnectionHandler._read_response`
    against a queue of fake server responses (raw bytes) or
    connection-level exceptions.
    """

    def __init__(self, client: KazooClient, script: list[object]):
        self.client = client
        self.script = list(script)
        self.requests: list[object] = []

    def __call__(self, request: object, async_object: object) -> None:
        self.requests.append(request)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            async_object.set_exception(item)
            return
        try:
            response = request.deserialize(item, 0)
        except Exception as exc:
            async_object.set_exception(exc)
            return
        if request.type == Transaction.type:
            response = Transaction.unchroot(self.client, response)
        async_object.set(response)


def make_client(script: list[object]) -> tuple[KazooClient, FakeConnection]:
    client = KazooClient()
    fake = FakeConnection(client, script)
    client._call = fake  # type: ignore[method-assign]
    return client, fake


def make_security_tx(client: KazooClient):
    """A multi-op security configuration update."""
    tx = client.transaction()
    tx.check("/security/config", 7)
    tx.set_data("/security/config", b'{"acl": "digest:ops"}', 7)
    tx.create("/security/audit/last-update", b"ops")
    return tx


class TestTransactionDeserialize:
    def test_success_response(self) -> None:
        payload = (
            _create_entry("/security/config")
            + _set_data_entry()
            + _check_entry()
            + DONE
        )
        results = Transaction.deserialize(payload, 0)
        assert results == ["/security/config", ZnodeStat(*STAT_VALUES), True]

    def test_error_response(self) -> None:
        payload = _error_entry(ROLLED_BACK) + _error_entry(NO_NODE) + DONE
        results = Transaction.deserialize(payload, 0)
        assert isinstance(results[0], RolledBackError)
        assert isinstance(results[1], NoNodeError)

    def test_unknown_header_type_raises(self) -> None:
        payload = _header(999) + DONE
        with pytest.raises(MarshallingError):
            Transaction.deserialize(payload, 0)

    def test_truncated_response_raises(self) -> None:
        payload = _header(SetData.type) + stat_struct.pack(*STAT_VALUES)[:10]
        with pytest.raises(struct.error):
            Transaction.deserialize(payload, 0)


class TestTransactionResult:
    def test_committed(self) -> None:
        result = TransactionResult.from_response(["/a", True], 2)
        assert result.status is TransactionStatus.COMMITTED
        assert result.is_committed
        assert not result.is_rolled_back
        assert not result.is_unknown
        assert result.error is None
        assert result.error_index is None

    def test_rolled_back_reports_real_cause(self) -> None:
        results = [RolledBackError(), BadVersionError(), RolledBackError()]
        result = TransactionResult.from_response(results, 3)
        assert result.status is TransactionStatus.ROLLED_BACK
        assert result.is_rolled_back
        assert isinstance(result.error, BadVersionError)
        assert result.error_index == 1

    def test_short_response_is_never_success(self) -> None:
        result = TransactionResult.from_response(["/a"], 3)
        assert result.status is TransactionStatus.UNKNOWN
        assert not result.is_committed

    def test_unknown_from_connection_loss(self) -> None:
        cause = ConnectionLoss()
        result = TransactionResult.unknown(2, cause)
        assert result.is_unknown
        assert result.cause is cause
        assert len(result) == 0

    def test_no_boolean_value(self) -> None:
        result = TransactionResult.from_response(["/a"], 1)
        with pytest.raises(TypeError):
            bool(result)

    def test_list_compatibility(self) -> None:
        result = TransactionResult.from_response(["/a", True], 2)
        assert len(result) == 2
        assert result[0] == "/a"
        assert list(result) == ["/a", True]
        assert result == ["/a", True]
        assert "/a" in result


class TestCommitEntryPoint:
    def test_commit_success(self) -> None:
        payload = (
            _check_entry()
            + _set_data_entry()
            + _create_entry("/security/audit/last-update")
            + DONE
        )
        client, fake = make_client([payload])
        tx = make_security_tx(client)
        assert tx.sent is False

        result = tx.commit()

        assert result.is_committed
        assert result[0] is True
        assert isinstance(result[1], ZnodeStat)
        assert result[2] == "/security/audit/last-update"
        assert tx.sent is True
        assert tx.committed is True  # "sent" only, not the outcome
        assert tx.result is result
        assert len(fake.requests) == 1

    @pytest.mark.parametrize(
        ("err_code", "exc_type"),
        [
            (BAD_VERSION, BadVersionError),
            (NO_AUTH, NoAuthError),
            (NO_NODE, NoNodeError),
        ],
    )
    def test_commit_middle_operation_fails(
        self, err_code: int, exc_type: type
    ) -> None:
        # The middle set_data fails: the server marks the other
        # operations as rolled back and reports the real error.
        payload = (
            _error_entry(ROLLED_BACK)
            + _error_entry(err_code)
            + _error_entry(ROLLED_BACK)
            + DONE
        )
        client, fake = make_client([payload])
        tx = make_security_tx(client)

        result = tx.commit()

        assert result.is_rolled_back
        assert not result.is_committed
        assert isinstance(result.error, exc_type)
        assert result.error_index == 1
        assert len(result) == 3
        assert len(fake.requests) == 1

    def test_commit_missing_result_entries_is_unknown(self) -> None:
        # Only two results for three operations, no error reported.
        payload = _check_entry() + _set_data_entry() + DONE
        client, fake = make_client([payload])
        tx = make_security_tx(client)

        result = tx.commit()

        assert result.is_unknown
        assert not result.is_committed
        assert len(fake.requests) == 1

    def test_commit_connection_loss_is_unknown_and_not_replayed(
        self,
    ) -> None:
        client, fake = make_client([ConnectionLoss()])
        tx = make_security_tx(client)

        result = tx.commit()

        assert result.is_unknown
        assert isinstance(result.cause, ConnectionLoss)
        # The transaction must not be automatically replayed.
        assert len(fake.requests) == 1

    def test_commit_session_expired_is_unknown(self) -> None:
        client, fake = make_client([SessionExpiredError()])
        tx = make_security_tx(client)

        result = tx.commit()

        assert result.is_unknown
        assert len(fake.requests) == 1

    def test_commit_malformed_response_does_not_report_success(
        self,
    ) -> None:
        payload = _header(SetData.type) + stat_struct.pack(*STAT_VALUES)[:10]
        client, fake = make_client([payload])
        tx = make_security_tx(client)

        with pytest.raises(struct.error):
            tx.commit()
        assert tx.result is None
        assert len(fake.requests) == 1

    def test_commit_twice_rejected(self) -> None:
        client, fake = make_client([_check_entry() + DONE])
        tx = client.transaction()
        tx.check("/security/config", 7)
        tx.commit()
        with pytest.raises(ValueError):
            tx.commit()
        assert len(fake.requests) == 1


class TestSecurityConfigExample:
    """The caller-side decision flow from the documentation example."""

    def run_update(
        self, client: KazooClient, audit_log: list, review_queue: list
    ) -> str:
        tx = client.transaction()
        tx.check("/security/config", 7)
        tx.set_data("/security/config", b'{"acl": "digest:ops"}', 7)
        result = tx.commit()

        if result.is_committed:
            audit_log.append(("committed", "/security/config"))
            return "audited"
        if result.is_rolled_back:
            audit_log.append(("rolled_back", "/security/config", result.error))
            return "rolled_back"
        # Unknown: never replay automatically, queue for a human.
        review_queue.append(("/security/config", result.cause))
        return "manual_review"

    def test_committed_writes_audit_record(self) -> None:
        payload = _check_entry() + _set_data_entry() + DONE
        client, _ = make_client([payload])
        audit_log: list = []
        review_queue: list = []

        assert self.run_update(client, audit_log, review_queue) == "audited"
        assert audit_log == [("committed", "/security/config")]
        assert review_queue == []

    def test_rolled_back_is_recorded_not_retried(self) -> None:
        payload = _error_entry(ROLLED_BACK) + _error_entry(NO_AUTH) + DONE
        client, _ = make_client([payload])
        audit_log: list = []
        review_queue: list = []

        outcome = self.run_update(client, audit_log, review_queue)

        assert outcome == "rolled_back"
        assert isinstance(audit_log[0][2], NoAuthError)
        assert review_queue == []

    def test_unknown_goes_to_manual_review(self) -> None:
        client, fake = make_client([ConnectionLoss()])
        audit_log: list = []
        review_queue: list = []

        assert (
            self.run_update(client, audit_log, review_queue) == "manual_review"
        )
        assert audit_log == []
        assert len(review_queue) == 1
        assert len(fake.requests) == 1
