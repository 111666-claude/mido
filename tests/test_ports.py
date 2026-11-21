# SPDX-FileCopyrightText: 2017 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

import collections

import pytest

import mido.ports
from mido.messages import Message
from mido.ports import BaseIOPort, MultiPort, multi_receive


class TestIOPort:
    class Port(BaseIOPort):
        def _open(self):
            self.close_called = False

        def _send(self, message):
            self._messages.append(message)

        def _close(self):
            self.close_called = True

    @pytest.fixture
    def port(self):
        with self.Port('Name') as p:
            yield p

    def test_basic(self, port):
        assert port.name == 'Name'
        assert not port.closed
        assert port._messages is port._parser.messages

        with pytest.raises(TypeError):
            port.send('not a message')

    def test_recv_non_blocking(self, port):
        message = Message('note_on')

        port.send(message)
        port.poll()
        assert port.poll() is None

    def test_send_message(self, port):
        message = Message('note_on')

        port.send(message)
        port.send(message)

    def test_port_close(self, port):
        port.close()
        assert port.close_called
        port.close_called = False
        port.close()
        assert port.closed
        assert not port.close_called

    def test_mido_port_non_blocking_recv(self, port):
        assert port.receive(block=False) is None


def test_close_inside_iteration():
    # This type of port can close when it runs out of messages.
    # (And example of this is socket ports.)
    #
    # Iteration should then stop after all messages in the
    # internal queue have been received.
    message = Message('note_on')

    class Port(BaseIOPort):
        def __init__(self, messages):
            BaseIOPort.__init__(self)
            # Simulate some messages that arrived
            # earlier.
            self._messages.extend(messages)
            self.closed = False

        def _receive(self, block=True):
            # Oops, the other end hung up.
            if self._messages:
                return self._messages.popleft()
            else:
                self.close()
                return None

    message = Message('note_on')
    with Port([message, message]) as port:
        assert len(list(port)) == 2


class FakeInputPort(BaseIOPort):
    """Controllable input port.

    The queue holds messages to deliver. A None entry means "no
    message this poll", which allows tests to decide exactly when a
    message arrives. The port never closes on its own.
    """

    def __init__(self, queue=()):
        BaseIOPort.__init__(self)
        self.queue = collections.deque(queue)
        self.poll_count = 0

    def _receive(self, block=True):
        self.poll_count += 1
        if self.queue:
            return self.queue.popleft()
        return None


class InfiniteInputPort(BaseIOPort):
    """A port with an endless supply of messages (e.g. a busy console
    port). Old multi_receive() would never get past this port."""

    def __init__(self):
        BaseIOPort.__init__(self)
        self.sent = 0

    def _receive(self, block=True):
        self.sent += 1
        return Message('note_on')


class SelfClosingPort(BaseIOPort):
    """Closes itself once its queue runs out (like a socket port whose
    peer hung up)."""

    def __init__(self, queue=()):
        BaseIOPort.__init__(self)
        self.queue = collections.deque(queue)
        self.poll_count = 0

    def _receive(self, block=True):
        self.poll_count += 1
        if self.queue:
            return self.queue.popleft()
        self.close()
        return None


class StopTest(Exception):
    pass


class TestMultiReceive:
    def test_budget_must_be_positive(self):
        with pytest.raises(ValueError):
            list(multi_receive([FakeInputPort()], budget=0))

    def test_sparse_port_max_wait_is_one_round(self):
        # A sparse port (e.g. emergency stop) sharing the rotation
        # with an infinitely chatty port must be served within one
        # full round: at most budget * (other ports) messages in
        # between. The None entry makes the message arrive right
        # after the sparse port's first turn, which is the worst case.
        estop = Message('note_off')
        sparse = FakeInputPort([None, estop])
        busy = InfiniteInputPort()
        budget = 3

        received = []
        for message in multi_receive([busy, sparse], budget=budget):
            received.append(message)
            if message is estop:
                break

        # Turn order: busy x3, sparse (empty), busy x3, sparse (estop).
        assert received[-1] is estop
        assert len(received) == 2 * budget + 1

    def test_infinite_port_cannot_starve_blocking(self):
        # With the old drain-per-port behavior this generator never
        # got past the infinite port.
        sparse = FakeInputPort([Message('note_off')])
        busy = InfiniteInputPort()
        gen = multi_receive([busy, sparse], budget=1)
        received = [next(gen) for _ in range(2)]
        assert Message('note_off') in received

    def test_per_port_order_preserved(self):
        msgs_a = [Message('note_on', note=i) for i in range(5)]
        msgs_b = [Message('note_off', note=i) for i in range(3)]
        a = FakeInputPort(msgs_a)
        b = FakeInputPort(msgs_b)

        out = []
        # Two non-blocking passes with budget=2 drain everything.
        for _ in range(2):
            out.extend(multi_receive([a, b], block=False, budget=2,
                                     yield_ports=True))
        out.extend(multi_receive([a, b], block=False, budget=2,
                                 yield_ports=True))

        assert [m for p, m in out if p is a] == msgs_a
        assert [m for p, m in out if p is b] == msgs_b

    def test_budget_limits_messages_per_pass(self):
        messages = [Message('note_on', note=i) for i in range(5)]
        port = FakeInputPort(messages)

        first = list(multi_receive([port], block=False, budget=2))
        assert first == messages[:2]
        # Nothing is swallowed: the rest is still pending.
        second = list(multi_receive([port], block=False, budget=2))
        assert second == messages[2:4]
        third = list(multi_receive([port], block=False, budget=2))
        assert third == messages[4:]

    def test_budget_none_restores_draining(self):
        msgs_a = [Message('note_on', note=i) for i in range(3)]
        msg_b = Message('note_off')
        a = FakeInputPort(msgs_a)
        b = FakeInputPort([msg_b])
        out = list(multi_receive([a, b], block=False, budget=None))
        assert out == msgs_a + [msg_b]

    def test_empty_ports_non_blocking(self):
        a, b = FakeInputPort(), FakeInputPort()
        assert list(multi_receive([a, b], block=False)) == []
        # Each port was polled exactly once (a single pass).
        assert a.poll_count == 1
        assert b.poll_count == 1

    def test_blocking_idle_sleeps_instead_of_busy_waiting(
            self, monkeypatch):
        sleeps = []

        def fake_sleep():
            sleeps.append(1)
            if len(sleeps) >= 3:
                raise StopTest

        monkeypatch.setattr(mido.ports, 'sleep', fake_sleep)
        a, b = FakeInputPort(), FakeInputPort()
        with pytest.raises(StopTest):
            list(multi_receive([a, b], block=True))
        # Exactly one poll per port per idle round: no busy-waiting.
        assert a.poll_count == 3
        assert b.poll_count == 3

    def test_non_blocking_never_sleeps(self, monkeypatch):
        def fail_sleep():
            raise AssertionError('sleep() called in non-blocking mode')

        monkeypatch.setattr(mido.ports, 'sleep', fail_sleep)
        port = FakeInputPort([Message('note_on')])
        assert len(list(multi_receive([port], block=False))) == 1

    def test_closed_port_is_drained_then_removed(self, monkeypatch):
        sleeps = []

        def fake_sleep():
            sleeps.append(1)
            if len(sleeps) >= 3:
                raise StopTest

        monkeypatch.setattr(mido.ports, 'sleep', fake_sleep)
        messages = [Message('note_on'), Message('note_on')]
        closer = SelfClosingPort(messages)
        other = FakeInputPort()

        received = []
        gen = multi_receive([closer, other], block=True)
        with pytest.raises(StopTest):
            while True:
                received.append(next(gen))

        # Pending messages were delivered before removal: not swallowed.
        assert received == messages
        # Two polls to deliver, one that saw the empty queue and closed
        # the port. After that the port was removed from the rotation,
        # so the 3 idle rounds never polled it again.
        assert closer.poll_count == 3

    def test_generator_ends_when_all_ports_closed(self):
        closer = SelfClosingPort([Message('note_on')])
        assert list(multi_receive([closer], block=True)) == \
            [Message('note_on')]


class TestMultiPort:
    def test_round_robin_receive(self):
        msgs_a = [Message('note_on', note=1), Message('note_on', note=2)]
        msg_b = Message('note_off')
        a = FakeInputPort(msgs_a)
        b = FakeInputPort([msg_b])
        port = MultiPort([a, b])

        # budget=1 interleaves the ports instead of draining a first.
        assert port.poll() == msgs_a[0]
        assert port.poll() == msg_b
        assert port.poll() == msgs_a[1]
        assert port.poll() is None

    def test_yield_ports(self):
        msg = Message('note_on')
        a = FakeInputPort([msg])
        port = MultiPort([a], yield_ports=True)
        assert port.poll() == (a, msg)

    def test_custom_budget(self):
        msgs = [Message('note_on', note=i) for i in range(3)]
        a = FakeInputPort(msgs)
        b = FakeInputPort()
        port = MultiPort([a, b], budget=10)
        for msg in msgs:
            assert port.poll() == msg

    def test_send_goes_to_all_ports(self):
        sent_a, sent_b = [], []

        class OutPort(BaseIOPort):
            def __init__(self, sink):
                BaseIOPort.__init__(self)
                self.sink = sink

            def _send(self, message):
                self.sink.append(message)

        a, b = OutPort(sent_a), OutPort(sent_b)
        port = MultiPort([a, b])
        msg = Message('note_on')
        port.send(msg)
        assert sent_a == [msg]
        assert sent_b == [msg]
