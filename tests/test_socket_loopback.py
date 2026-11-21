# SPDX-FileCopyrightText: 2026
#
# SPDX-License-Identifier: MIT

"""End-to-end tests for the MIDI-over-TCP chain.

These tests exercise the full link that the unit tests in
``test_parser.py`` / ``test_tokenizer.py`` can not see:

    TCP segment -> SocketPort._receive() -> Parser/Tokenizer
                 -> PortServer accept/multi-port fan-out

The tests use a real loopback TCP connection on an ephemeral port
(127.0.0.1:0).  Synchronisation is always expressed as an event wait
with a bounded timeout (``wait_until`` / ``queue.Queue.get``); there are
no fixed sleeps.  A negative helper (``assert_remains_empty``) proves a
queue *stays* empty within a bounded window.

Run only this module with:

    python -m pytest tests/test_socket_loopback.py -v

The deliberately broken variants that prove these tests can fail are
driven by ``tests/mutation/socket_mutations.py`` and verified by
``tests/test_socket_mutations.py``.
"""

import queue
import socket
import threading

import pytest

from mido.messages import Message
from mido.sockets import PortServer, connect

# Bounded waits.  Overridable for slow/loaded CI machines. The values
# are upper bounds, not delays: tests resume as soon as the condition is
# true, so a healthy machine finishes in milliseconds.
TIMEOUT = 2.0
IDLE_WINDOW = 0.2
POLL_INTERVAL = 0.005

SYSEX_START = 0xF0
SYSEX_END = 0xF7
REALTIME_CLOCK = 0xF8


def wait_until(predicate, timeout=TIMEOUT, interval=POLL_INTERVAL,
               reason='condition was not reached in time'):
    """Poll *predicate* until it returns a truthy value.

    Returns the truthy value.  Raises ``AssertionError`` after
    *timeout* seconds so a broken link fails the test instead of
    hanging forever.
    """
    deadline = threading.Event()
    waited = 0.0
    while True:
        value = predicate()
        if value:
            return value
        if waited >= timeout:
            raise AssertionError(
                f'{reason} (after {timeout:.1f}s)')
        deadline.wait(min(interval, timeout - waited))
        waited += interval


class _PumpThread(threading.Thread):
    """Drive ``port.receive(block=False)`` from a background thread.

    PortServer/SocketPort only parse incoming bytes when their
    ``_receive`` is polled, so every port under test needs a pump.
    Received messages are stored on a queue; pump exceptions are
    captured instead of silently killing the thread.
    """

    def __init__(self, port):
        super().__init__(daemon=True)
        self.port = port
        self.messages = []
        self._queue = queue.Queue()
        self._stop = threading.Event()
        self.error = None
        self.start()

    def run(self):
        try:
            while not self._stop.is_set():
                # PortServer.receive() may return a message directly
                # (drained from a child SocketPort inside the receive
                # loop), so the return value must be captured rather
                # than only inspecting the deque afterwards.
                msg = self.port.receive(block=False)
                if msg is not None:
                    self.messages.append(msg)
                    self._queue.put(msg)
                else:
                    self._stop.wait(POLL_INTERVAL)
        except Exception as err:  # noqa: BLE001 - surfaced in stop()
            self.error = err

    def get(self, timeout=TIMEOUT):
        return self._queue.get(timeout=timeout)

    def wait_count(self, count, timeout=TIMEOUT):
        wait_until(lambda: len(self.messages) >= count, timeout=timeout,
                   reason=f'did not receive {count} message(s)')
        # Drain the queue so later assert_remains_empty() only sees
        # messages that arrive *after* this point.
        for _ in range(count):
            self._queue.get(timeout=timeout)
        return self.messages[:count]

    def assert_remains_empty(self, window=IDLE_WINDOW):
        """Assert no message arrives within a bounded idle window."""
        try:
            msg = self._queue.get(timeout=window)
        except queue.Empty:
            return
        raise AssertionError(f'unexpected message received: {msg!r}')

    def stop(self):
        self._stop.set()
        self.join(timeout=TIMEOUT)
        assert not self.is_alive(), 'pump thread did not stop'
        if self.error is not None:
            raise self.error


class ServerHarness:
    """A PortServer plus its pump, running on an ephemeral loopback port."""

    def __init__(self):
        self.server = PortServer('127.0.0.1', 0)
        self.host, self.port = self.server._socket.getsockname()
        self.pump = _PumpThread(self.server)

    def server_port_for(self, raw_socket):
        """Find the server-side SocketPort connected to *raw_socket*.

        The accepted socket's peer address is the client socket's
        *local* address, so match against ``getsockname``.
        """
        client_local = raw_socket.getsockname()

        def match():
            for port in self.server.ports:
                if port._socket.getpeername() == client_local:
                    return port
            return None

        return wait_until(match, reason='server never accepted connection')

    def stop(self):
        self.pump.stop()
        self.server.close()


@pytest.fixture
def server():
    harness = ServerHarness()
    yield harness
    harness.stop()


@pytest.fixture
def raw_client(server):
    """A plain TCP client with Nagle disabled for deliberate fragmentation."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.connect((server.host, server.port))
    try:
        yield sock
    finally:
        sock.close()


@pytest.fixture
def make_client(server):
    """Factory producing connected, pumped mido SocketPort clients."""
    clients = []

    def _make():
        port = connect(server.host, server.port)
        pump = _PumpThread(port)
        clients.append((port, pump))
        return port, pump

    yield _make

    for port, pump in clients:
        pump.stop()
        port.close()


def _send_chunk_and_wait_drained(raw_client, server_port, chunk):
    """Send *chunk* and block until the server side has consumed it.

    Waiting for the server parser's consumed-byte counter to advance by
    exactly ``len(chunk)`` guarantees the bytes were read inside a
    SocketPort._receive() call.  (Polling socket readiness can not prove
    this: an empty socket is "not readable" too, so such a wait could
    return before the segment even arrives.)  This makes a test that
    sends one chunk at a time really exercise state carried *across*
    receive calls rather than relying on TCP keeping segments separate.
    """
    before = server_port._bytes_consumed
    raw_client.sendall(chunk)
    expected = before + len(chunk)
    wait_until(
        lambda: server_port._bytes_consumed == expected,
        reason='server never finished reading the sent chunk')


def test_status_and_data_bytes_split_across_segments(server, raw_client):
    """A status byte and its data bytes may arrive in separate packets.

    The Tokenizer must retain the running-status state between
    SocketPort._receive() calls until the full message is reassembled.
    """
    server_port = server.server_port_for(raw_client)

    # First segment: status byte only. No complete message may be
    # delivered, but the tokenizer must remember the running status.
    _send_chunk_and_wait_drained(raw_client, server_port, bytes([0x90]))
    assert server_port._parser._tok._status == 0x90
    assert server.pump.messages == []

    # Second segment: the two data bytes, each in its own segment.
    _send_chunk_and_wait_drained(raw_client, server_port, bytes([0x40]))
    _send_chunk_and_wait_drained(raw_client, server_port, bytes([0x7F]))

    received = server.pump.get()
    assert received == Message('note_on', channel=0, note=0x40,
                               velocity=0x7F)
    server.pump.assert_remains_empty()
    assert server_port.closed is False


def test_fragmented_stream_preserves_message_boundaries_and_order(
        server, raw_client):
    """Several messages, sent one byte at a time, stay ordered and whole."""
    messages = [
        Message('note_on', channel=0, note=1, velocity=10),
        Message('control_change', channel=3, control=7, value=64),
        Message('note_off', channel=0, note=1, velocity=0),
    ]
    stream = b''.join(msg.bin() for msg in messages)
    server_port = server.server_port_for(raw_client)

    # Send every byte in its own, fully acknowledged receive cycle.
    for byte in stream:
        _send_chunk_and_wait_drained(raw_client, server_port, bytes([byte]))

    got = server.pump.wait_count(len(messages))
    assert got == messages
    server.pump.assert_remains_empty()


def test_realtime_message_inside_sysex(server, raw_client):
    """A realtime clock interleaved into a fragmented SysEx survives.

    The realtime message must be delivered first while the SysEx
    continues to assemble; each chunk is acknowledged so partial SysEx
    state is provably carried across multiple receive calls.
    """
    server_port = server.server_port_for(raw_client)
    # F0 01 02 | F8 | 03 04 | F7, every chunk in its own segment.
    chunks = [
        bytes([SYSEX_START, 0x01, 0x02]),
        bytes([REALTIME_CLOCK]),
        bytes([0x03, 0x04]),
        bytes([SYSEX_END]),
    ]
    for chunk in chunks:
        _send_chunk_and_wait_drained(raw_client, server_port, chunk)

    first = server.pump.get()
    second = server.pump.get()
    assert first == Message('clock')
    assert second == Message('sysex', data=[1, 2, 3, 4])
    server.pump.assert_remains_empty()


def test_broadcast_to_multiple_clients_and_client_disconnect(
        server, make_client):
    """PortServer fans messages out; one client leaving does not break
    the others, and its server-side port is cleaned up."""
    client_a, pump_a = make_client()
    client_b, pump_b = make_client()
    client_c, pump_c = make_client()

    # The server must hold one live SocketPort per connected client.
    wait_until(lambda: len(server.server.ports) == 3,
               reason='server did not register all three clients')

    broadcast = Message('note_on', channel=2, note=60, velocity=99)
    server.server.send(broadcast)
    for pump in (pump_a, pump_b, pump_c):
        assert pump.get() == broadcast

    # Clients can also send upstream; ordering across clients is not
    # defined, so verify each message is present exactly once.
    client_a.send(Message('control_change', control=1, value=11))
    client_b.send(Message('control_change', control=2, value=22))
    uplink = server.pump.wait_count(2)
    assert sorted(uplink, key=lambda m: m.control) == [
        Message('control_change', control=1, value=11),
        Message('control_change', control=2, value=22),
    ]

    # Disconnect client B. Its server-side port must close and be pruned.
    client_b.close()
    wait_until(
        lambda: all(p is not client_b for p in server.server.ports),
        reason='disconnected client port was not removed')

    # The two remaining clients keep receiving every broadcast.
    second = Message('note_off', channel=2, note=60, velocity=0)
    server.server.send(second)
    assert pump_a.get() == second
    assert pump_c.get() == second
    pump_b.stop()
    pump_b.assert_remains_empty()

    # All surviving server-side ports are still open.
    assert all(not p.closed for p in server.server.ports)
    assert len(server.server.ports) == 2


def test_server_close_propagates_to_clients(server, make_client):
    """Closing the PortServer closes every accepted connection and the
    clients observe the hang-up on their next poll."""
    client_a, pump_a = make_client()
    client_b, pump_b = make_client()
    wait_until(lambda: len(server.server.ports) == 2,
               reason='server did not register both clients')

    server.server.close()

    wait_until(lambda: client_a.closed and client_b.closed,
               reason='clients were not closed when the server shut down')
    for port in server.server.ports:
        assert port.closed
    # Pumps must terminate cleanly instead of raising.
    pump_a.stop()
    pump_b.stop()
