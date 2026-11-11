"""Deterministic failure-injection tests for :class:`Lock` recovery.

The hardest recovery case for a Kazoo lock is a connection drop right
after the server created the sequential contender node but before the
create response reached the client.  These tests drive that exact state
machine against an in-memory client stand-in that records every node
and every response, so the assertions are deterministic and do not
require a ZooKeeper ensemble.

Two flavors of the failure are exercised:

* the original session is still valid, so the orphaned sequential node
  created by the lost create call still exists and must be re-adopted;
* the session has expired, so every ephemeral node owned by it is gone
  and a fresh sequential node has to be created.

In both cases the lock must only ever claim the node belonging to its
own application attempt: it must not leave duplicate contender nodes
behind, must not mistake a similarly prefixed node for its own and must
not announce ownership while a lower-sequence contender is still ahead.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, TYPE_CHECKING

import pytest

from kazoo.exceptions import (
    CancelledError,
    ConnectionLoss,
    NodeExistsError,
    NoNodeError,
)
from kazoo.protocol.states import EventType, KazooState, WatchedEvent
from kazoo.recipe.lock import Lock, ReadLock, WriteLock
from kazoo.retry import ForceRetryError, KazooRetry

if TYPE_CHECKING:
    from kazoo.interfaces import IHandler

LOCK_PATH = "/lockpath"


class GatedSleep:
    """A retry ``sleep_func`` whose progress the test can hold and release.

    KazooRetry sleeps after every retryable exception (e.g. a
    ``ConnectionLoss``).  Blocking that sleep makes retry timing
    deterministic: the test decides when the next attempt may run instead
    of racing an actual timer.  Every entry is counted so tests can wait
    until a specific retry attempt is parked.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._open = False
        self._entries = 0

    def __call__(self, _seconds: float) -> None:
        with self._cond:
            self._entries += 1
            self._cond.notify_all()
            while not self._open:
                self._cond.wait()

    def wait_for_entries(self, count: int, timeout: float = 5.0) -> None:
        with self._cond:
            if not self._cond.wait_for(
                lambda: self._entries >= count, timeout=timeout
            ):
                raise AssertionError(
                    "retry sleep entered %s times, wanted %s"
                    % (self._entries, count)
                )

    def open(self) -> None:
        with self._cond:
            self._open = True
            self._cond.notify_all()


class FakeHandler:
    """Minimal handler surface used by :class:`Lock`."""

    name = "fake_threading_handler"

    def __init__(self, sleep_func: Callable[[float], None]):
        self.sleep_func = sleep_func

    def event_object(self) -> threading.Event:
        return threading.Event()

    def lock_object(self) -> threading.Lock:
        return threading.Lock()

    def rlock_object(self) -> threading.RLock:
        return threading.RLock()


class FakeClient:
    """In-memory stand-in for the ZooKeeper-backed :class:`KazooClient`.

    Nodes are stored in a plain dict keyed by absolute path.  Sequence
    nodes get monotonically increasing, zero-padded counters just like
    ZooKeeper.  Every mutating call and every returned value is appended
    to a response log so tests can assert exactly what the recipe did.
    """

    def __init__(self, sleep_func: Callable[[float], None] | None = None):
        self._lock = threading.RLock()
        self._nodes: dict[str, bytes] = {}
        self._ephemeral: set[str] = set()
        self._watches: dict[str, list[Callable[[WatchedEvent], None]]] = {}
        self._listeners: list[Callable[[KazooState], bool | None]] = []
        self._counters: dict[str, int] = {}
        self._fail_create: str | None = None
        self.session_expired = False
        self.log: list[tuple[str, Any]] = []
        sleep_func = sleep_func or (lambda _seconds: None)
        # The stand-in only implements the handler surface Lock touches.
        self.handler: IHandler = FakeHandler(sleep_func)  # type: ignore
        self.retry = KazooRetry(max_tries=None, sleep_func=sleep_func)

    # -- fault injection -------------------------------------------------

    def fail_next_create(self, create_path: str) -> None:
        """Make the next create for ``create_path`` lose its response.

        The node is still created server-side; only the response is lost,
        mirroring the classic "request succeeded, connection dropped"
        window.
        """
        self._fail_create = create_path

    def expire_session(self) -> None:
        """Simulate session expiration: ephemerals vanish, state is LOST."""
        with self._lock:
            for path in list(self._ephemeral):
                del self._nodes[path]
            self._ephemeral.clear()
            self.session_expired = True
            listeners = list(self._listeners)
        for listener in listeners:
            listener(KazooState.LOST)

    def add_foreign_contender(
        self,
        name: str = "__lock__",
        data: bytes = b"other",
        seq: int = 1,
    ) -> str:
        """Insert a contender node that belongs to a different client.

        The node name matches the recipe's contender suffix
        (``<name><10-digit sequence>``) so lock ordering treats it as a
        real predecessor rather than filtering it out.
        """
        node = "foreign-%s%010d" % (name, seq)
        with self._lock:
            self._nodes[LOCK_PATH + "/" + node] = data
        return node

    def add_lookalike(self, prefix: str, seq: int = 99) -> str:
        """Insert an orphaned node whose name starts with ``prefix``."""
        node = "%s%010d" % (prefix, seq)
        with self._lock:
            self._nodes[LOCK_PATH + "/" + node] = b"orphan"
        return node

    # -- KazooClient surface used by the lock recipe ---------------------

    def ensure_path(self, path: str) -> bool:
        with self._lock:
            existed = path in self._nodes
            self._nodes.setdefault(path, b"")
            self.log.append(("ensure_path", path))
            return existed

    def create(
        self,
        path: str,
        data: bytes = b"",
        ephemeral: bool = False,
        sequence: bool = False,
    ) -> str:
        with self._lock:
            if sequence:
                counter = self._counters.get(path, 0) + 1
                self._counters[path] = counter
                full_path = "%s%010d" % (path, counter)
            else:
                full_path = path
            if full_path in self._nodes:
                raise NodeExistsError(full_path)
            self._nodes[full_path] = data
            if ephemeral:
                self._ephemeral.add(full_path)
            should_fail = self._fail_create == path
            if should_fail:
                self._fail_create = None
            self.log.append(
                (
                    "create",
                    full_path,
                    {"ephemeral": ephemeral, "sequence": sequence},
                )
            )
            if should_fail:
                raise ConnectionLoss(full_path)
            return full_path

    def get_children(self, path: str) -> list[str]:
        with self._lock:
            if path not in self._nodes:
                raise NoNodeError(path)
            children = [
                child[len(path) + 1 :]
                for child in self._nodes
                if child.startswith(path + "/")
            ]
            self.log.append(("get_children", path, list(children)))
            return children

    def exists(self, path: str) -> bool:
        with self._lock:
            return path in self._nodes

    def get(
        self,
        path: str,
        watch: Callable[[WatchedEvent], None] | None = None,
    ) -> tuple[bytes, Any]:
        with self._lock:
            if path not in self._nodes:
                self.log.append(("get_missing", path))
                raise NoNodeError(path)
            if watch is not None:
                self._watches.setdefault(path, []).append(watch)
            data = self._nodes[path]
            self.log.append(("get", path))
            return data, None

    def delete(self, path: str) -> bool:
        with self._lock:
            if path not in self._nodes:
                self.log.append(("delete_missing", path))
                raise NoNodeError(path)
            self._nodes.pop(path, None)
            self._ephemeral.discard(path)
            watches = list(self._watches.pop(path, ()))
            self.log.append(("delete", path))
        for watch in watches:
            watch(
                WatchedEvent(
                    type=EventType.DELETED,
                    state=KazooState.LOST,
                    path=path,
                )
            )
        return True

    def add_listener(
        self, listener: Callable[[KazooState], bool | None]
    ) -> None:
        with self._lock:
            self._listeners.append(listener)

    def remove_listener(
        self, listener: Callable[[KazooState], bool | None]
    ) -> None:
        with self._lock:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

    # -- test helpers ----------------------------------------------------

    def child_names(self) -> list[str]:
        return self.get_children(LOCK_PATH)

    def own_nodes(self, lock: Lock) -> list[str]:
        return [
            child
            for child in self.child_names()
            if child.startswith(lock.prefix)
        ]

    def create_responses(self) -> list[str]:
        return [entry[1] for entry in self.log if entry[0] == "create"]


def _make_lock(
    client: FakeClient, lock_cls: type[Lock] = Lock, identifier: str = "me"
) -> Lock:
    return lock_cls(client, LOCK_PATH, identifier)


def _wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met before timeout")
        time.sleep(0.005)


class AcquireThread(threading.Thread):
    """Runs a blocking acquire and exposes its outcome."""

    def __init__(self, lock: Lock, **kwargs: Any):
        super().__init__(daemon=True)
        self.lock = lock
        self.kwargs = kwargs
        self.result: Any = None
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            self.result = self.lock.acquire(**self.kwargs)
        except BaseException as exc:  # captured for the asserting thread
            self.error = exc


class TestCreateResponseLost:
    """Connection drop between node creation and its response."""

    def test_valid_session_recovers_own_node_and_releases(self) -> None:
        sleep = GatedSleep()
        client = FakeClient(sleep)
        lock = _make_lock(client)
        client.fail_next_create(lock.create_path)

        thread = AcquireThread(lock)
        thread.start()
        sleep.wait_for_entries(1)
        # The server created the sequential node even though the client
        # saw a ConnectionLoss; another contender is ahead of us.
        orphaned = client.own_nodes(lock)
        assert len(orphaned) == 1
        predecessor = client.add_foreign_contender(seq=0)
        assert client.own_nodes(lock) == orphaned

        sleep.open()
        # The recovered attempt adopts its node and queues behind the
        # foreign contender; deleting that predecessor wakes its watch.
        _wait_for(lambda: lock.node is not None)
        client.delete(LOCK_PATH + "/" + predecessor)
        thread.join(5)
        assert thread.error is None
        assert thread.result is True
        # Exactly one create was attempted and one node exists for us.
        assert len(client.create_responses()) == 1
        assert client.own_nodes(lock) == orphaned
        assert lock.node == orphaned[0]

        assert lock.release() is True
        assert lock.node is None
        assert client.own_nodes(lock) == []
        assert client.child_names() == []

    def test_expired_session_recreates_and_releases(self) -> None:
        sleep = GatedSleep()
        client = FakeClient(sleep)
        lock = _make_lock(client)
        client.fail_next_create(lock.create_path)

        thread = AcquireThread(lock)
        thread.start()
        sleep.wait_for_entries(1)
        first_node = client.own_nodes(lock)
        assert len(first_node) == 1

        # Session expiration removes the ephemeral contender; another
        # client's node survives because it is not ephemeral here.
        predecessor = client.add_foreign_contender(seq=0)
        client.expire_session()
        assert client.child_names() == [predecessor]

        sleep.open()
        # A fresh node is created for the new session and queues behind
        # the surviving foreign contender until that predecessor leaves.
        _wait_for(lambda: lock.node is not None)
        client.delete(LOCK_PATH + "/" + predecessor)
        thread.join(5)
        assert thread.error is None
        assert thread.result is True
        # A second sequential node had to be created for the new session.
        assert len(client.create_responses()) == 2
        own = client.own_nodes(lock)
        assert len(own) == 1
        assert own[0] != first_node[0]
        assert lock.node == own[0]

        assert lock.release() is True
        assert client.own_nodes(lock) == []
        assert client.child_names() == []

    def test_expired_session_non_blocking_reacquires(self) -> None:
        sleep = GatedSleep()
        client = FakeClient(sleep)
        lock = _make_lock(client)
        client.fail_next_create(lock.create_path)

        thread = AcquireThread(lock)
        thread.start()
        sleep.wait_for_entries(1)
        client.expire_session()
        sleep.open()
        thread.join(5)
        assert thread.error is None
        assert thread.result is True
        assert len(client.create_responses()) == 2
        assert len(client.own_nodes(lock)) == 1
        lock.release()
        assert client.own_nodes(lock) == []

    def test_valid_session_cancel_before_retry_cleans_up(self) -> None:
        sleep = GatedSleep()
        client = FakeClient(sleep)
        lock = _make_lock(client)
        client.fail_next_create(lock.create_path)

        thread = AcquireThread(lock)
        thread.start()
        sleep.wait_for_entries(1)
        created = client.own_nodes(lock)
        assert len(created) == 1
        predecessor = client.add_foreign_contender(seq=0)

        # Cancel while a retry attempt is deterministically parked: no
        # second create may run and the existing node must be removed.
        lock.cancel()
        sleep.open()
        thread.join(5)
        assert isinstance(thread.error, CancelledError)
        assert len(client.create_responses()) == 1
        assert client.own_nodes(lock) == []
        assert client.child_names() == [predecessor]

    def test_expired_session_cancel_cleans_recreated_node(self) -> None:
        sleep = GatedSleep()
        client = FakeClient(sleep)
        lock = _make_lock(client)
        client.fail_next_create(lock.create_path)

        thread = AcquireThread(lock)
        thread.start()
        sleep.wait_for_entries(1)
        predecessor = client.add_foreign_contender(seq=0)
        client.expire_session()
        assert client.child_names() == [predecessor]

        # The original ephemeral is gone, so the retry creates one fresh
        # node for the new session and queues behind the predecessor.
        sleep.open()
        _wait_for(
            lambda: len(client.create_responses()) == 2
            and lock.node is not None
        )
        recreated = client.own_nodes(lock)
        assert len(recreated) == 1

        # Cancelling that pending attempt deletes the replacement node;
        # nothing of ours may be left behind.
        lock.cancel()
        thread.join(5)
        assert isinstance(thread.error, CancelledError)
        assert len(client.create_responses()) == 2
        assert client.own_nodes(lock) == []
        assert client.child_names() == [predecessor]

    def test_cancel_while_waiting_behind_predecessor(self) -> None:
        sleep = GatedSleep()
        client = FakeClient(sleep)
        lock = _make_lock(client)
        client.fail_next_create(lock.create_path)

        thread = AcquireThread(lock)
        thread.start()
        sleep.wait_for_entries(1)
        predecessor = client.add_foreign_contender(seq=0)
        sleep.open()

        # The recovered attempt adopts its node and parks on the watch.
        _wait_for(lambda: lock.node is not None)
        assert len(client.create_responses()) == 1
        assert len(client.own_nodes(lock)) == 1
        assert thread.result is None

        lock.cancel()
        thread.join(5)
        assert isinstance(thread.error, CancelledError)
        assert len(client.create_responses()) == 1
        assert client.own_nodes(lock) == []
        assert client.child_names() == [predecessor]

    def test_predecessor_delete_wakes_recovered_lock(self) -> None:
        sleep = GatedSleep()
        client = FakeClient(sleep)
        lock = _make_lock(client)
        client.fail_next_create(lock.create_path)

        thread = AcquireThread(lock)
        thread.start()
        sleep.wait_for_entries(1)
        predecessor = client.add_foreign_contender(seq=0)
        sleep.open()

        _wait_for(lambda: lock.node is not None)
        assert thread.result is None
        client.delete(LOCK_PATH + "/" + predecessor)
        thread.join(5)
        assert thread.error is None
        assert thread.result is True
        assert lock.release() is True
        assert client.child_names() == []


class TestNoFalseOwnership:
    """The lock must not claim a node or ownership it does not own."""

    def test_find_node_requires_exact_prefix(self) -> None:
        client = FakeClient()
        lock = _make_lock(client)
        client.ensure_path(LOCK_PATH)
        other = _make_lock(client)
        other_node = client.add_lookalike(other.prefix, seq=1)
        lookalike = client.add_lookalike(lock.prefix[:-1], seq=2)

        assert lock._find_node() is None
        real = client.add_lookalike(lock.prefix, seq=3)
        assert lock._find_node() == real
        # The shorter-prefix orphan must never be selected.
        assert lock._find_node() != lookalike
        assert other_node != real

    def test_get_predecessor_ignores_higher_and_foreign_sequences(
        self,
    ) -> None:
        client = FakeClient()
        lock = _make_lock(client)
        own = "%s%010d" % (lock.prefix, 5)
        ahead = "foreign-__lock__%010d" % 4
        behind = "foreign-__lock__%010d" % 6
        client.ensure_path(LOCK_PATH)
        for child in (ahead, own, behind):
            client._nodes[LOCK_PATH + "/" + child] = b"x"

        assert lock._get_predecessor(own) == ahead

        client.delete(LOCK_PATH + "/" + ahead)
        assert lock._get_predecessor(own) is None

    def test_missing_self_forces_retry(self) -> None:
        client = FakeClient()
        lock = _make_lock(client)
        client.ensure_path(LOCK_PATH)
        ghost = "%s%010d" % (lock.prefix, 7)
        with pytest.raises(ForceRetryError):
            lock._get_predecessor(ghost)

    def test_lookalike_orphan_does_not_become_our_node(self) -> None:
        sleep = GatedSleep()
        client = FakeClient(sleep)
        lock = _make_lock(client)
        client.fail_next_create(lock.create_path)

        thread = AcquireThread(lock)
        thread.start()
        sleep.wait_for_entries(1)
        created = client.own_nodes(lock)
        assert len(created) == 1

        # A lower-sequence contender queues us; an orphan with a similar
        # prefix and a higher sequence must not be adopted as self.
        predecessor = client.add_foreign_contender(seq=0)
        orphan = client.add_lookalike(lock.prefix[:-1], seq=200)
        sleep.open()

        _wait_for(lambda: lock.node is not None)
        assert lock.node == created[0]
        assert lock.node != orphan
        assert thread.result is None
        assert len(client.create_responses()) == 1
        assert set(client.own_nodes(lock)) == {created[0]}

        client.delete(LOCK_PATH + "/" + predecessor)
        thread.join(5)
        assert thread.error is None
        assert thread.result is True
        assert lock.release() is True
        # Only the foreign orphan remains; our node was released, never
        # duplicated and never confused with the lookalike.
        assert client.child_names() == [orphan]

    def test_non_blocking_retry_does_not_duplicate_node(self) -> None:
        sleep = GatedSleep()
        client = FakeClient(sleep)
        lock = _make_lock(client)
        client.fail_next_create(lock.create_path)

        # Blocking recovery with no one ahead: the lock must adopt its
        # already-created node instead of creating a second one.
        thread = AcquireThread(lock)
        thread.start()
        sleep.wait_for_entries(1)
        created = client.own_nodes(lock)
        assert len(created) == 1
        sleep.open()
        thread.join(5)
        assert thread.error is None
        assert thread.result is True
        assert len(client.create_responses()) == 1
        assert lock.node == created[0]

        # A later non-blocking attempt while the lock is held returns
        # False and must not leave an extra contender node behind.
        assert lock.acquire(blocking=False) is False
        assert client.own_nodes(lock) == created
        lock.release()
        assert client.own_nodes(lock) == []


class TestReadWriteOrderingPreserved:
    """Recovery must keep the read/write lock sorting rules intact."""

    def test_write_lock_excludes_readers_and_writers(self) -> None:
        client = FakeClient()
        write = WriteLock(client, LOCK_PATH, "writer")
        reader = ReadLock(client, LOCK_PATH, "reader")
        client.ensure_path(LOCK_PATH)
        reader_node = "%s%010d" % (reader.prefix, 1)
        own = "%s%010d" % (write.prefix, 2)
        for child in (reader_node, own):
            client._nodes[LOCK_PATH + "/" + child] = b"x"

        assert write._get_predecessor(own) == reader_node
        assert write._find_node() == own

    def test_read_lock_only_excludes_writers(self) -> None:
        client = FakeClient()
        reader = ReadLock(client, LOCK_PATH, "reader")
        other_reader = ReadLock(client, LOCK_PATH, "reader2")
        writer = WriteLock(client, LOCK_PATH, "writer")
        client.ensure_path(LOCK_PATH)
        writer_node = "%s%010d" % (writer.prefix, 1)
        peer_node = "%s%010d" % (other_reader.prefix, 2)
        own = "%s%010d" % (reader.prefix, 3)
        for child in (writer_node, peer_node, own):
            client._nodes[LOCK_PATH + "/" + child] = b"x"

        # Readers don't block each other; the writer is the predecessor.
        assert reader._get_predecessor(own) == writer_node
        client.delete(LOCK_PATH + "/" + writer_node)
        assert reader._get_predecessor(own) is None

    def test_write_lock_recovery_adopts_own_ephemeral(self) -> None:
        sleep = GatedSleep()
        client = FakeClient(sleep)
        lock = WriteLock(client, LOCK_PATH, "writer")
        client.fail_next_create(lock.create_path)

        thread = AcquireThread(lock)
        thread.start()
        sleep.wait_for_entries(1)
        created = client.own_nodes(lock)
        assert len(created) == 1
        reader = ReadLock(client, LOCK_PATH, "reader")
        predecessor = client.add_lookalike(reader.prefix, seq=100)
        sleep.open()

        _wait_for(lambda: lock.node is not None)
        assert lock.node == created[0]
        client.delete(LOCK_PATH + "/" + predecessor)
        thread.join(5)
        assert thread.error is None
        assert thread.result is True
        assert len(client.create_responses()) == 1
        assert lock.release() is True
        assert client.own_nodes(lock) == []
