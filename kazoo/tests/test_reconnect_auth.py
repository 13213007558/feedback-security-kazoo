"""In-process tests for CONNECTED-vs-authentication ordering.

A config service resumes business writes when Kazoo delivers the
CONNECTED event. These tests verify -- without any real ZooKeeper
cluster -- that CONNECTED is only delivered after every configured
credential has actually been accepted by the server, including across
reconnects.

A fake ZooKeeper server (see :class:`FakeZookeeper`) sits behind a
fake socket. It records one ordered log of outbound client frames and
client state events, and the tests release server responses one at a
time so they can observe exactly what the client does in between.
"""

from __future__ import annotations

import threading
import time
import types

import pytest

from kazoo.client import KazooClient
from kazoo.exceptions import AuthFailedError
from kazoo.protocol import connection as connection_module
from kazoo.protocol.serialization import (
    bool_struct,
    int_int_long_struct,
    int_long_int_long_struct,
    int_struct,
    read_string,
    reply_header_struct,
    stat_struct,
    write_buffer,
    write_string,
)
from kazoo.protocol.states import EventType, KazooState, KeeperState

AUTH_XID = -4
PING_XID = -2
SASL_TYPE = 102
CLOSE_TYPE = -11
AUTH_FAILED_CODE = -115

REQUEST_TYPES = {
    1: "create",
    2: "delete",
    3: "exists",
    4: "getdata",
    5: "setdata",
    12: "getchildren",
}

SESSION_PASSWD = b"\x01" * 16


class FakeZKSocket:
    """Socket stand-in. The client writes frames here; the test feeds
    response bytes. Every send() is one complete length-prefixed frame
    because Kazoo's `_write` loops until everything is sent and we
    always accept the whole buffer."""

    def __init__(self, server: "FakeZookeeper") -> None:
        self.server = server
        self._cond = threading.Condition()
        self._incoming = bytearray()
        self.closed = False

    # -- socket-ish API used by kazoo -------------------------------
    def setblocking(self, flag: bool) -> None:
        pass

    def settimeout(self, timeout: float | None) -> None:
        pass

    def setsockopt(self, *args: object) -> None:
        pass

    def send(self, data: bytes) -> int:
        with self._cond:
            if self.closed:
                raise OSError("socket is closed")
        self.server.frame_from_client(self, bytes(data))
        return len(data)

    def recv(self, n: int) -> bytes:
        with self._cond:
            while not self._incoming and not self.closed:
                self._cond.wait()
            if self._incoming:
                chunk = bytes(self._incoming[:n])
                del self._incoming[:n]
                return chunk
            return b""

    def close(self) -> None:
        with self._cond:
            self.closed = True
            self._cond.notify_all()

    # -- test-side API ----------------------------------------------
    def readable(self) -> bool:
        with self._cond:
            return bool(self._incoming) or self.closed

    def feed(self, data: bytes) -> None:
        with self._cond:
            self._incoming += data
            self._cond.notify_all()


class FakeZookeeper:
    """Scriptable fake ZooKeeper server.

    `log` is a single ordered record of everything observable:
    ``("frame", ...)`` entries for outbound client requests and
    ``("state", KazooState)`` entries for states delivered to the
    client's listeners. Both are recorded on the connection thread,
    so their relative order in the log is exact.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self.log: list[tuple] = []
        self.sockets: list[FakeZKSocket] = []

    # -- connection handling -----------------------------------------
    def new_socket(self) -> FakeZKSocket:
        sock = FakeZKSocket(self)
        with self._cond:
            self.sockets.append(sock)
        return sock

    @property
    def current_socket(self) -> FakeZKSocket:
        return self.sockets[-1]

    def drop(self) -> None:
        """Simulate the server side dropping the connection."""
        self.current_socket.close()

    # -- outbound frame parsing (client -> server) --------------------
    def frame_from_client(self, sock: FakeZKSocket, data: bytes) -> None:
        (length,) = int_struct.unpack_from(data, 0)
        payload = data[4:]
        assert (
            len(payload) == length
        ), "partial frame, expected one send per frame"
        if sock is self.current_socket and not getattr(
            sock, "connect_seen", False
        ):
            sock.connect_seen = True
            _, _, _, session_id = int_long_int_long_struct.unpack_from(
                payload, 0
            )
            frame: tuple = ("connect", session_id)
        else:
            (xid,) = int_struct.unpack_from(payload, 0)
            if xid == PING_XID:
                frame = ("ping",)
                sock.feed(self._wrap(reply_header_struct.pack(PING_XID, 0, 0)))
            elif xid == AUTH_XID:
                scheme, _ = read_string(payload, 12)
                frame = ("auth", scheme)
            else:
                (req_type,) = int_struct.unpack_from(payload, 4)
                if req_type == SASL_TYPE:
                    frame = ("sasl", xid)
                elif req_type == CLOSE_TYPE:
                    frame = ("close", xid)
                    sock.feed(self._wrap(reply_header_struct.pack(xid, 0, 0)))
                else:
                    name = REQUEST_TYPES.get(req_type, "type%d" % req_type)
                    frame = ("request", name, xid)
        self._record(("frame",) + frame)

    # -- scripted responses (server -> client), released one by one ---
    @staticmethod
    def _wrap(body: bytes) -> bytes:
        return int_struct.pack(len(body)) + body

    def release_connect(
        self,
        session_id: int = 0x1000,
        passwd: bytes = SESSION_PASSWD,
        timeout: int = 10000,
        read_only: bool = False,
    ) -> None:
        body = (
            int_int_long_struct.pack(0, timeout, session_id)
            + write_buffer(passwd)
            + bool_struct.pack(1 if read_only else 0)
        )
        self.current_socket.feed(self._wrap(body))

    def release_auth(self, err: int = 0) -> None:
        self.current_socket.feed(
            self._wrap(reply_header_struct.pack(AUTH_XID, 0, err))
        )

    def release_sasl(
        self, xid: int, err: int = 0, challenge: bytes = b""
    ) -> None:
        body = reply_header_struct.pack(xid, 0, err)
        if not err:
            body += write_buffer(challenge)
        self.current_socket.feed(self._wrap(body))

    def release_reply(self, xid: int, body: bytes = b"", err: int = 0) -> None:
        self.current_socket.feed(
            self._wrap(reply_header_struct.pack(xid, 1, err) + body)
        )

    # -- event log -----------------------------------------------------
    def _record(self, event: tuple) -> None:
        with self._cond:
            self.log.append(event)
            self._cond.notify_all()

    def record_state(self, state: KazooState) -> None:
        self._record(("state", state))

    def record_keeper_state(self, state: KeeperState) -> None:
        self._record(("keeper", state))

    def wait_for(self, pred, count: int = 1, timeout: float = 5.0) -> tuple:
        """Wait until at least `count` log events match; return the last."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                matches = [e for e in self.log if pred(e)]
                if len(matches) >= count:
                    return matches[-1]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(
                        "timed out waiting for event; log=%r" % (self.log,)
                    )
                self._cond.wait(min(remaining, 0.1))

    def frames(self, kind: str | None = None) -> list[tuple]:
        with self._cond:
            return [
                e[1:]
                for e in self.log
                if e[0] == "frame" and (kind is None or e[1] == kind)
            ]

    def states(self) -> list[KazooState]:
        with self._cond:
            return [e[1] for e in self.log if e[0] == "state"]


# -- log predicates ----------------------------------------------------
def is_frame(kind: str):
    return lambda e: e[0] == "frame" and e[1] == kind


def is_connect(session_id: int):
    return (
        lambda e: e[0] == "frame" and e[1] == "connect" and e[2] == session_id
    )


def is_request(name: str):
    return lambda e: e[0] == "frame" and e[1] == "request" and e[2] == name


def is_state(state: KazooState):
    return lambda e: e[0] == "state" and e[1] == state


def is_keeper(state: KeeperState):
    return lambda e: e[0] == "keeper" and e[1] == state


def assert_order(server: FakeZookeeper, *preds) -> None:
    """Assert the log contains events matching each predicate in order."""
    with server._cond:
        log = list(server.log)
    pos = 0
    for pred in preds:
        while pos < len(log) and not pred(log[pos]):
            pos += 1
        if pos == len(log):
            raise AssertionError("events out of order; log=%r" % (log,))
        pos += 1


def make_client(server: FakeZookeeper, **kwargs) -> KazooClient:
    """Build a KazooClient wired to the fake server, in-process only."""
    kwargs.setdefault("hosts", "127.0.0.1:2181")
    client = KazooClient(**kwargs)
    handler = client.handler
    handler.create_connection = lambda *a, **k: server.new_socket()
    real_select = handler.select

    def select(rlist, wlist, xlist, timeout=None):
        fake_r = [f for f in rlist if isinstance(f, FakeZKSocket)]
        fake_w = [f for f in wlist if isinstance(f, FakeZKSocket)]
        real_r = [f for f in rlist if not isinstance(f, FakeZKSocket)]
        real_w = [f for f in wlist if not isinstance(f, FakeZKSocket)]
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            r, w, x = real_select(real_r, real_w, xlist, 0.01)
            r, w = list(r), list(w)
            r.extend(f for f in fake_r if f.readable())
            w.extend(f for f in fake_w if not f.closed)
            if r or w or x:
                return r, w, x
            if deadline is not None and time.monotonic() >= deadline:
                return [], [], []

    handler.select = select
    client.add_listener(server.record_state)

    # record protocol-level keeper states as well; KazooState.LOST is the
    # client's initial state, so a failed first authentication never
    # re-notifies it -- the KeeperState stream is the reliable signal
    # for AUTH_FAILED.
    session_callback = client._session_callback

    def record_keeper_state(state):
        server.record_keeper_state(state)
        return session_callback(state)

    client._session_callback = record_keeper_state
    return client


@pytest.fixture
def server():
    return FakeZookeeper()


@pytest.fixture
def client_factory(server):
    clients = []

    def factory(**kwargs) -> KazooClient:
        client = make_client(server, **kwargs)
        clients.append(client)
        return client

    yield factory

    for sock in server.sockets:
        sock.close()
    for client in clients:
        try:
            client.stop()
        except Exception:
            pass
        try:
            client.close()
        except Exception:
            pass


@pytest.fixture
def fake_puresasl(monkeypatch):
    """Inject a minimal stand-in for the puresasl package so the real
    `_authenticate_with_sasl` code path runs without the dependency."""
    module = types.ModuleType("puresasl")

    class SASLError(Exception):
        pass

    class SASLProtocolException(Exception):
        pass

    class SASLClient:
        def __init__(self, host=None, **kwargs):
            self.complete = False

        def process(self, challenge=None):
            self.complete = True
            return b"fake-sasl-response"

        def dispose(self):
            pass

    client_module = types.ModuleType("puresasl.client")
    client_module.SASLClient = SASLClient
    module.client = client_module
    module.SASLError = SASLError
    module.SASLProtocolException = SASLProtocolException
    monkeypatch.setattr(connection_module, "puresasl", module, raising=False)
    monkeypatch.setattr(connection_module, "PURESASL_AVAILABLE", True)
    return module


SASL_OPTIONS = {
    "mechanism": "DIGEST-MD5",
    "username": "zkuser",
    "password": "zkpass",
    "service": "zookeeper",
}


class TestConnectedOnlyAfterAuthAccepted:
    """CONNECTED must not be delivered before the server has accepted
    every configured credential."""

    def test_new_session(self, server, client_factory):
        client = client_factory(auth_data=[("digest", "jsmith:jsmith")])
        client.start_async()

        server.wait_for(is_connect(0))
        server.release_connect(session_id=0x1001)
        server.wait_for(is_frame("auth"))

        # auth sent but not yet accepted: no CONNECTED may leak out
        time.sleep(0.3)
        assert KazooState.CONNECTED not in server.states()

        server.release_auth()
        server.wait_for(is_state(KazooState.CONNECTED))

        # business write resumed on CONNECTED goes out only afterwards
        result = client.create_async("/resumed-write", b"data")
        frame = server.wait_for(is_request("create"))
        server.release_reply(frame[3], body=write_string("/resumed-write"))
        assert result.get(timeout=5) == "/resumed-write"

        assert_order(
            server,
            is_frame("connect"),
            is_frame("auth"),
            is_state(KazooState.CONNECTED),
            is_request("create"),
        )

    def test_session_resume(self, server, client_factory):
        client = client_factory(auth_data=[("digest", "jsmith:jsmith")])
        client.start_async()

        server.wait_for(is_connect(0))
        server.release_connect(session_id=0xBEEF)
        server.wait_for(is_frame("auth"))
        server.release_auth()
        server.wait_for(is_state(KazooState.CONNECTED))
        assert client.client_id == (0xBEEF, SESSION_PASSWD)

        server.drop()
        server.wait_for(is_state(KazooState.SUSPENDED))

        # the client must try to resume the same session
        server.wait_for(is_connect(0xBEEF))
        server.release_connect(session_id=0xBEEF)
        server.wait_for(is_frame("auth"), count=2)

        # re-authentication is in flight: still no new CONNECTED
        time.sleep(0.3)
        assert server.states().count(KazooState.CONNECTED) == 1

        server.release_auth()
        server.wait_for(is_state(KazooState.CONNECTED), count=2)
        assert client.client_id == (0xBEEF, SESSION_PASSWD)

        assert_order(
            server,
            is_connect(0xBEEF),
            is_frame("auth"),
            is_state(KazooState.CONNECTED),
        )


class TestAuthFailureNeverSignalsRecovery:
    """A failed re-authentication must not first notify the business of
    a successful recovery and only then report the error."""

    def test_one_of_multiple_auths_fails(self, server, client_factory):
        client = client_factory(
            auth_data=[("digest", "good:good"), ("digest", "bad:bad")]
        )
        client.start_async()

        server.wait_for(is_connect(0))
        server.release_connect(session_id=0x1002)
        server.wait_for(is_frame("auth"))
        server.release_auth()
        server.wait_for(is_frame("auth"), count=2)

        # the app queues a protected write while the second credential
        # is still unacknowledged
        result = client.create_async("/protected", b"data")
        time.sleep(0.3)
        assert server.frames("request") == []
        assert KazooState.CONNECTED not in server.states()

        server.release_auth(err=AUTH_FAILED_CODE)
        server.wait_for(is_keeper(KeeperState.AUTH_FAILED))

        with pytest.raises(AuthFailedError):
            result.get(timeout=5)

        # the write was never sent anonymously, and CONNECTED never fired
        time.sleep(0.3)
        assert server.frames("request") == []
        assert KazooState.CONNECTED not in server.states()

    def test_sasl_failure(self, server, client_factory, fake_puresasl):
        client = client_factory(sasl_options=dict(SASL_OPTIONS))
        client.start_async()

        server.wait_for(is_connect(0))
        server.release_connect(session_id=0x1003)
        frame = server.wait_for(is_frame("sasl"))

        time.sleep(0.3)
        assert KazooState.CONNECTED not in server.states()

        server.release_sasl(frame[2], err=AUTH_FAILED_CODE)
        server.wait_for(is_keeper(KeeperState.AUTH_FAILED))
        assert KazooState.CONNECTED not in server.states()

    def test_sasl_unavailable(self, server, client_factory, monkeypatch):
        # puresasl missing entirely: SASLException before any handshake
        monkeypatch.setattr(connection_module, "PURESASL_AVAILABLE", False)
        client = client_factory(sasl_options=dict(SASL_OPTIONS))
        client.start_async()

        server.wait_for(is_connect(0))
        server.release_connect(session_id=0x1004)
        server.wait_for(is_keeper(KeeperState.AUTH_FAILED))
        assert KazooState.CONNECTED not in server.states()


class TestContractPreservedOnRecovery:
    """Normal recovery: queued operations and watchers keep working
    according to the existing contract."""

    def test_request_queued_during_auth_is_held_then_sent(
        self, server, client_factory
    ):
        client = client_factory(auth_data=[("digest", "jsmith:jsmith")])
        client.start_async()

        server.wait_for(is_connect(0))
        server.release_connect(session_id=0x1005)
        server.wait_for(is_frame("auth"))

        # queued while authentication is still in flight
        result = client.create_async("/protected", b"data")
        time.sleep(0.3)
        assert server.frames("request") == []
        assert KazooState.CONNECTED not in server.states()

        server.release_auth()
        server.wait_for(is_state(KazooState.CONNECTED))
        frame = server.wait_for(is_request("create"))
        server.release_reply(frame[3], body=write_string("/protected"))
        assert result.get(timeout=5) == "/protected"

        assert_order(
            server,
            is_frame("connect"),
            is_frame("auth"),
            is_state(KazooState.CONNECTED),
            is_request("create"),
        )

    def test_watchers_and_queued_ops_survive_reconnect(
        self, server, client_factory
    ):
        client = client_factory(auth_data=[("digest", "jsmith:jsmith")])
        client.start_async()

        server.wait_for(is_connect(0))
        server.release_connect(session_id=0x1006)
        server.wait_for(is_frame("auth"))
        server.release_auth()
        server.wait_for(is_state(KazooState.CONNECTED))

        watch_events = []
        stat_missing = stat_struct.pack(-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        exists_result = client.exists_async(
            "/watched", watch=watch_events.append
        )
        frame = server.wait_for(is_request("exists"))
        server.release_reply(frame[3], body=stat_missing)
        assert exists_result.get(timeout=5) is None
        assert "/watched" in client._data_watchers

        server.drop()
        server.wait_for(is_state(KazooState.SUSPENDED))

        # watchers are notified with a NONE event so the app can
        # re-register them after the reconnect
        deadline = time.monotonic() + 5
        while not any(e.type == EventType.NONE for e in watch_events):
            assert time.monotonic() < deadline
            time.sleep(0.05)

        # an operation queued while reconnection is still in progress
        # continues once the session is recovered
        server.wait_for(is_connect(0x1006))
        queued = client.get_async("/queued")
        server.release_connect(session_id=0x1006)
        server.wait_for(is_frame("auth"), count=2)
        server.release_auth()
        server.wait_for(is_state(KazooState.CONNECTED), count=2)

        frame = server.wait_for(is_request("getdata"))
        server.release_reply(
            frame[3], body=write_buffer(b"value") + stat_missing
        )
        assert queued.get(timeout=5)[0] == b"value"

        # re-registering the watcher after recovery works as before
        reregistered = client.exists_async(
            "/watched", watch=watch_events.append
        )
        frame = server.wait_for(is_request("exists"), count=2)
        server.release_reply(frame[3], body=stat_missing)
        assert reregistered.get(timeout=5) is None

    def test_sasl_success_signals_connected_after_handshake(
        self, server, client_factory, fake_puresasl
    ):
        client = client_factory(sasl_options=dict(SASL_OPTIONS))
        client.start_async()

        server.wait_for(is_connect(0))
        server.release_connect(session_id=0x1007)
        frame = server.wait_for(is_frame("sasl"))
        server.release_sasl(frame[2])
        server.wait_for(is_state(KazooState.CONNECTED))

        assert_order(
            server,
            is_frame("connect"),
            is_frame("sasl"),
            is_state(KazooState.CONNECTED),
        )
