# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

import threading

import pytest

from mido.messages import Message
from mido.ports import BaseOutput, MultiPort
from mido.routing import (
    ALL_TYPES,
    CHANNEL_TYPES,
    REALTIME_TYPES,
    ClosedOutputError,
    Router,
    RoutingLoopError,
    RoutingRule,
    RoutingSendError,
    UnmatchedMessageError,
)


class FakeOutput(BaseOutput):
    """Output port that records what it receives."""

    def __init__(self, name='fake', log=None):
        self.sent = []
        self.log = log
        BaseOutput.__init__(self, name)

    def _send(self, message):
        self.sent.append(message)
        if self.log is not None:
            self.log.append((self.name, message))

    def types(self):
        return [msg.type for msg in self.sent]

    def channels(self):
        return [getattr(msg, 'channel', None) for msg in self.sent]


class FailingOutput(BaseOutput):
    def __init__(self, name='failing'):
        BaseOutput.__init__(self, name)

    def _send(self, message):
        raise RuntimeError('boom')


def make_message(type_='note_on', **kwargs):
    return Message(type_, **kwargs)


# ---------------------------------------------------------------
# Matching
# ---------------------------------------------------------------

class TestRuleMatching:
    def test_channel_rule_matches_by_channel(self):
        p = FakeOutput()
        router = Router([RoutingRule([p], channels=[2])])

        router.send(make_message(channel=2, note=1))
        router.send(make_message(channel=3, note=2))

        assert p.channels() == [2]

    def test_channel_rule_matches_by_type(self):
        p = FakeOutput()
        router = Router([RoutingRule([p], types='note_on')])

        router.send(make_message('note_on'))
        router.send(make_message('note_off'))

        assert p.types() == ['note_on']

    def test_multiple_types_and_channels(self):
        p = FakeOutput()
        router = Router([
            RoutingRule([p], types=['note_on', 'note_off'], channels=[0, 1])
        ])

        router.send(make_message('note_on', channel=0))
        router.send(make_message('note_off', channel=1))
        router.send(make_message('note_on', channel=2))
        router.send(make_message('control_change', channel=0))

        assert [(m.type, m.channel) for m in p.sent] == [
            ('note_on', 0), ('note_off', 1)]

    def test_type_group_name(self):
        p = FakeOutput()
        router = Router([RoutingRule([p], types='realtime')])

        router.send(make_message('clock'))
        router.send(make_message('note_on'))

        assert p.types() == ['clock']

    def test_match_callable_is_combined(self):
        p = FakeOutput()
        router = Router([
            RoutingRule([p], types='control_change',
                        match=lambda msg: msg.control == 7)
        ])

        router.send(make_message('control_change', control=7))
        router.send(make_message('control_change', control=1))

        assert len(p.sent) == 1
        assert p.sent[0].control == 7

    def test_multiple_outputs_receive_in_order(self):
        log = []
        a = FakeOutput('a', log)
        b = FakeOutput('b', log)
        router = Router([RoutingRule([a, b])])

        router.send(make_message())

        assert [name for name, _ in log] == ['a', 'b']

    def test_rule_accepts_a_single_port(self):
        p = FakeOutput()
        router = Router([RoutingRule(p)])
        router.send(make_message())
        assert len(p.sent) == 1

    def test_invalid_rule_arguments(self):
        p = FakeOutput()

        with pytest.raises(ValueError):
            RoutingRule([p], channels=[16])
        with pytest.raises(ValueError):
            RoutingRule([p], channels=[])
        with pytest.raises(ValueError):
            RoutingRule([p], types='no_such_type')
        with pytest.raises(ValueError):
            RoutingRule([p], types=[])
        with pytest.raises(ValueError):
            # No channel message type, so channels can never match.
            RoutingRule([p], types='clock', channels=[0])
        with pytest.raises(ValueError):
            RoutingRule([p], remap=99)
        with pytest.raises(TypeError):
            RoutingRule(['not a port'])

    def test_type_groups_are_consistent(self):
        assert CHANNEL_TYPES.isdisjoint(REALTIME_TYPES)
        assert REALTIME_TYPES < ALL_TYPES
        assert 'sysex' in ALL_TYPES
        assert 'note_on' in CHANNEL_TYPES
        assert 'songpos' not in REALTIME_TYPES


# ---------------------------------------------------------------
# Ordering and remapping
# ---------------------------------------------------------------

class TestOrderingAndRemap:
    def test_declaration_order(self):
        log = []
        a = FakeOutput('a', log)
        b = FakeOutput('b', log)
        c = FakeOutput('c', log)

        router = Router([
            RoutingRule([b]),
            RoutingRule([c]),
            RoutingRule([a]),
        ])
        router.send(make_message())

        assert [name for name, _ in log] == ['b', 'c', 'a']

    def test_explicit_priority_wins_over_declaration_order(self):
        log = []
        a = FakeOutput('a', log)
        b = FakeOutput('b', log)
        c = FakeOutput('c', log)

        router = Router([
            RoutingRule([a], priority=0),
            RoutingRule([b], priority=10),
            RoutingRule([c], priority=5),
        ])
        router.send(make_message())

        assert [name for name, _ in log] == ['b', 'c', 'a']

    def test_equal_priority_keeps_declaration_order(self):
        log = []
        a = FakeOutput('a', log)
        b = FakeOutput('b', log)

        router = Router([
            RoutingRule([a], priority=1),
            RoutingRule([b], priority=1),
        ])
        router.send(make_message())

        assert [name for name, _ in log] == ['a', 'b']

    def test_rules_property_keeps_declaration_order(self):
        a, b = FakeOutput(), FakeOutput()
        first = RoutingRule([a], priority=0)
        second = RoutingRule([b], priority=10)

        router = Router([first, second])

        assert router.rules == (first, second)
        assert router.ordered_rules == (second, first)

    def test_stop_ends_processing(self):
        log = []
        a = FakeOutput('a', log)
        b = FakeOutput('b', log)

        router = Router([
            RoutingRule([a], stop=True),
            RoutingRule([b]),
        ])
        router.send(make_message())

        assert [name for name, _ in log] == ['a']

    def test_stop_with_no_match_does_not_end_processing(self):
        log = []
        a = FakeOutput('a', log)
        b = FakeOutput('b', log)

        router = Router([
            RoutingRule([a], channels=[9], stop=True),
            RoutingRule([b]),
        ])
        router.send(make_message(channel=0))

        assert [name for name, _ in log] == ['b']

    def test_remap_int_sends_a_copy(self):
        p = FakeOutput()
        router = Router([RoutingRule([p], channels=[0], remap=5)])

        original = make_message(channel=0, note=60)
        router.send(original)

        assert p.sent[0].channel == 5
        assert original.channel == 0

    def test_remap_callable(self):
        p = FakeOutput()
        router = Router([
            RoutingRule([p], channels=[0, 1, 2], remap=lambda ch: ch + 8)
        ])

        router.send(make_message(channel=0))
        router.send(make_message(channel=2))

        assert p.channels() == [8, 10]

    def test_remap_callable_can_leave_channel_unchanged(self):
        p = FakeOutput()
        router = Router([RoutingRule([p], remap=lambda ch: None)])

        router.send(make_message(channel=3))

        assert p.channels() == [3]

    def test_remap_ignores_time_and_other_attributes(self):
        p = FakeOutput()
        router = Router([RoutingRule([p], remap=1)])

        message = make_message(channel=0, note=60, velocity=100, time=12)
        router.send(message)

        assert p.sent[0].dict() == message.copy(channel=1).dict()

    def test_original_message_is_never_modified(self):
        a, b = FakeOutput(), FakeOutput()
        router = Router([
            RoutingRule([a], channels=[0], remap=1),
            RoutingRule([b], channels=[0], remap=2),
        ])

        original = make_message('note_on', channel=0, note=60)
        snapshot = original.dict()
        router.send(original)

        assert original.dict() == snapshot
        assert a.sent[0].channel == 1
        assert b.sent[0].channel == 2


# ---------------------------------------------------------------
# Default paths for SysEx / system messages / unmatched messages
# ---------------------------------------------------------------

class TestDefaultPaths:
    def test_sysex_is_not_matched_by_channel_rules(self):
        channel_port = FakeOutput()
        sysex_port = FakeOutput()
        router = Router(
            [RoutingRule([channel_port])],
            sysex=[sysex_port])

        router.send(make_message('sysex', data=[1, 2, 3]))

        assert channel_port.sent == []
        assert sysex_port.types() == ['sysex']

    def test_realtime_is_not_matched_by_channel_rules(self):
        channel_port = FakeOutput()
        realtime_port = FakeOutput()
        router = Router(
            [RoutingRule([channel_port])],
            realtime=[realtime_port])

        router.send(make_message('clock'))
        router.send(make_message('start'))

        assert channel_port.sent == []
        assert realtime_port.types() == ['clock', 'start']

    def test_system_messages_fall_back_to_default(self):
        default = FakeOutput()
        router = Router(default=[default])

        router.send(make_message('songpos', pos=1))
        router.send(make_message('tune_request'))

        assert default.types() == ['songpos', 'tune_request']

    def test_sysex_and_realtime_fall_back_to_default(self):
        default = FakeOutput()
        router = Router(default=[default])

        router.send(make_message('sysex', data=[1]))
        router.send(make_message('clock'))

        assert default.types() == ['sysex', 'clock']

    def test_specific_path_wins_over_default(self):
        default = FakeOutput('default')
        sysex_port = FakeOutput('sysex')
        router = Router(default=[default], sysex=[sysex_port])

        router.send(make_message('sysex', data=[1]))
        router.send(make_message('songpos'))

        assert sysex_port.types() == ['sysex']
        assert default.types() == ['songpos']

    def test_explicit_rule_beats_default(self):
        rule_port = FakeOutput('rule')
        sysex_port = FakeOutput('sysex')
        default = FakeOutput('default')
        router = Router(
            [RoutingRule([rule_port], types='sysex')],
            sysex=[sysex_port],
            default=[default])

        router.send(make_message('sysex', data=[1]))
        router.send(make_message('clock'))

        assert rule_port.types() == ['sysex']
        assert sysex_port.sent == []
        assert default.types() == ['clock']

    def test_explicit_rule_can_match_clock(self):
        p = FakeOutput()
        router = Router([RoutingRule([p], types='clock')])

        router.send(make_message('clock'))
        router.send(make_message('start'))

        assert p.types() == ['clock']

    def test_empty_match_is_dropped_by_default(self):
        p = FakeOutput()
        router = Router([RoutingRule([p], channels=[9])])

        router.send(make_message(channel=0))  # Must not raise.

        assert p.sent == []

    def test_empty_match_can_raise(self):
        router = Router([], on_unmatched='raise')

        with pytest.raises(UnmatchedMessageError):
            router.send(make_message())

    def test_explicitly_empty_path_suppresses_default(self):
        default = FakeOutput()
        router = Router(
            [RoutingRule([], channels=[0])],
            default=[default])

        router.send(make_message(channel=0))
        router.send(make_message(channel=1))

        # The rule matched (so no default) but routes nowhere; the
        # second message did not match and takes the default path.
        assert len(default.sent) == 1
        assert default.sent[0].channel == 1

    def test_empty_sysex_path_suppresses_default(self):
        default = FakeOutput()
        router = Router(sysex=[], default=[default])

        router.send(make_message('sysex', data=[1]))

        assert default.sent == []

    def test_invalid_option_values(self):
        with pytest.raises(ValueError):
            Router(on_unmatched='ignore')
        with pytest.raises(ValueError):
            Router(on_error='ignore')
        with pytest.raises(ValueError):
            Router(on_closed='ignore')


# ---------------------------------------------------------------
# Closed ports and send failures
# ---------------------------------------------------------------

class TestFailures:
    def test_closed_output_is_skipped_by_default(self):
        closed = FakeOutput('closed')
        open_port = FakeOutput('open')
        closed.close()

        router = Router([RoutingRule([closed, open_port])])
        router.send(make_message())

        assert closed.sent == []
        assert len(open_port.sent) == 1

    def test_closed_output_can_raise(self):
        closed = FakeOutput('closed')
        closed.close()

        router = Router([RoutingRule([closed])], on_closed='raise')

        with pytest.raises(ClosedOutputError):
            router.send(make_message())

    def test_send_error_is_wrapped_by_default(self):
        failing = FailingOutput()
        later = FakeOutput('later')
        router = Router([RoutingRule([failing, later])])

        with pytest.raises(RoutingSendError) as info:
            router.send(make_message())

        assert info.value.port is failing
        assert isinstance(info.value.original, RuntimeError)
        # Delivery stops at the failing port.
        assert later.sent == []

    def test_send_error_can_be_skipped(self):
        failing = FailingOutput()
        later = FakeOutput('later')
        router = Router([RoutingRule([failing, later])], on_error='skip')

        router.send(make_message())

        assert len(later.sent) == 1

    def test_send_error_after_partial_fanout_keeps_earlier_deliveries(self):
        first = FakeOutput('first')
        failing = FailingOutput()
        router = Router([RoutingRule([first, failing])])

        with pytest.raises(RoutingSendError):
            router.send(make_message())

        assert len(first.sent) == 1

    def test_nested_send_error_is_not_double_wrapped(self):
        failing = FailingOutput('failing')
        inner = Router([RoutingRule([failing])], name='inner')
        outer = Router([RoutingRule([inner])], name='outer')

        with pytest.raises(RoutingSendError) as info:
            outer.send(make_message())

        assert info.value.port is failing
        assert not isinstance(info.value.original, RoutingSendError)

    def test_outer_router_can_skip_a_failing_inner_router(self):
        failing = FailingOutput('failing')
        later = FakeOutput('later')
        inner = Router([RoutingRule([failing])], name='inner')
        outer = Router(
            [RoutingRule([inner, later])], name='outer', on_error='skip')

        outer.send(make_message())

        assert len(later.sent) == 1

    def test_router_closed_itself_raises(self):
        router = Router()
        router.close()

        with pytest.raises(ValueError):
            router.send(make_message())

    def test_closing_router_does_not_close_outputs_by_default(self):
        p = FakeOutput()
        router = Router([RoutingRule([p])])
        router.close()

        assert not p.closed

    def test_close_outputs_closes_downstream_ports(self):
        p = FakeOutput()
        router = Router([RoutingRule([p])], close_outputs=True)
        router.close()

        assert p.closed


# ---------------------------------------------------------------
# Cycles
# ---------------------------------------------------------------

class TestCycles:
    def test_self_cycle_is_rejected(self):
        router = Router()

        with pytest.raises(RoutingLoopError):
            router.set_rules([RoutingRule([router])])

        assert router.rules == ()

    def test_two_router_cycle_is_rejected(self):
        a = Router(name='a')
        b = Router(name='b')
        a.set_rules([RoutingRule([b])])

        with pytest.raises(RoutingLoopError):
            b.set_rules([RoutingRule([a])])

        assert b.rules == ()
        a.send(make_message())  # a -> b, b has no rules: dropped.

    def test_cycle_through_default_path_is_rejected(self):
        a = Router(name='a')
        b = Router(name='b', default=[a])

        with pytest.raises(RoutingLoopError):
            a.set_rules([RoutingRule([b])])

    def test_failed_rule_replacement_keeps_old_rules(self):
        p = FakeOutput()
        old = RoutingRule([p])
        router = Router([old])

        with pytest.raises(RoutingLoopError):
            router.set_rules([RoutingRule([router])])

        assert router.rules == (old,)
        router.send(make_message())
        assert len(p.sent) == 1

    def test_hidden_cycle_is_detected_at_run_time(self):
        # The eager check cannot see through the plain forwarding port,
        # so the run-time guard has to catch it.
        class ForwardingPort(BaseOutput):
            def __init__(self, target):
                self.target = target
                BaseOutput.__init__(self, 'forward')

            def _send(self, message):
                self.target.send(message)

        forward = ForwardingPort(None)
        a = Router(name='a')
        b = Router(name='b')
        forward.target = b
        a.set_rules([RoutingRule([forward])])
        b.set_rules([RoutingRule([a])])

        with pytest.raises(RoutingLoopError):
            a.send(make_message())

    def test_non_cyclic_shared_output_router_is_allowed(self):
        shared = Router(name='shared')
        p = FakeOutput()
        shared.set_rules([RoutingRule([p])])

        a = Router([RoutingRule([shared])])
        b = Router([RoutingRule([shared])])

        a.send(make_message())
        b.send(make_message())

        assert len(p.sent) == 2


# ---------------------------------------------------------------
# Rule replacement
# ---------------------------------------------------------------

class TestRuleReplacement:
    def test_rules_can_be_replaced(self):
        a, b = FakeOutput(), FakeOutput()
        router = Router([RoutingRule([a])])

        router.rules = [RoutingRule([b])]
        router.send(make_message())

        assert a.sent == []
        assert len(b.sent) == 1

    def test_set_rules_accepts_none_and_single_rule(self):
        p = FakeOutput()
        router = Router()
        router.set_rules(RoutingRule([p]))
        router.send(make_message())
        router.set_rules(None)
        router.send(make_message())

        assert len(p.sent) == 1

    def test_add_and_clear_rules(self):
        a, b = FakeOutput(), FakeOutput()
        router = Router([RoutingRule([a])])
        router.add_rule(RoutingRule([b]))
        router.send(make_message())

        assert (len(a.sent), len(b.sent)) == (1, 1)

        router.clear_rules()
        router.send(make_message())
        assert (len(a.sent), len(b.sent)) == (1, 1)

    def test_set_rules_rejects_non_rules(self):
        router = Router()

        with pytest.raises(TypeError):
            router.set_rules(['not a rule'])
        with pytest.raises(TypeError):
            router.set_rules(42)


def test_rules_are_immutable():
    rule = RoutingRule([FakeOutput()])

    with pytest.raises(AttributeError):
        rule.channels = [1]


def test_router_is_a_normal_output_port():
    router = Router()
    assert router.is_output
    assert not router.is_input
    assert not router.closed

    with router:
        pass
    assert router.closed


def test_router_works_inside_multiport():
    a, b = FakeOutput(), FakeOutput()
    router = Router([RoutingRule([a], channels=[0]),
                     RoutingRule([b], channels=[1])])
    multi = MultiPort([router])

    multi.send(make_message(channel=0))
    multi.send(make_message(channel=1))
    multi.close()

    assert len(a.sent) == len(b.sent) == 1


def test_frozen_messages_can_be_routed():
    # Frozen messages reject attribute assignment; the router must not
    # try to modify them in place when remapping.
    from mido.frozen import freeze_message

    p = FakeOutput()
    router = Router([RoutingRule([p], remap=1)])

    frozen = freeze_message(make_message(channel=0))
    router.send(frozen)

    assert p.channels() == [1]
    assert frozen.channel == 0


# ---------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------

class TestConcurrentRuleUpdates:
    def test_message_is_processed_with_one_rule_snapshot(self):
        """Replacing rules mid-message must not split a message."""
        entered = threading.Event()
        release = threading.Event()

        old_a = FakeOutput('old_a')
        old_b = FakeOutput('old_b')
        new_a = FakeOutput('new_a')
        new_b = FakeOutput('new_b')

        def blocking_match(message):
            entered.set()
            assert release.wait(5)
            return True

        old_rules = [
            RoutingRule([old_a], match=blocking_match),
            RoutingRule([old_b]),
        ]
        new_rules = [
            RoutingRule([new_a]),
            RoutingRule([new_b]),
        ]

        router = Router(old_rules)
        errors = []

        def send():
            try:
                router.send(make_message())
            except Exception as exc:  # pragma: no cover - test failure path
                errors.append(exc)

        thread = threading.Thread(target=send)
        thread.start()

        assert entered.wait(5)
        router.set_rules(new_rules)
        release.set()
        thread.join(5)

        assert not thread.is_alive()
        assert errors == []

        # The in-flight message used the old snapshot completely.
        assert len(old_a.sent) == len(old_b.sent) == 1
        assert new_a.sent == new_b.sent == []

        # The next message uses the new snapshot completely and the
        # ports from the old snapshot are not touched again.
        router.send(make_message())
        assert len(old_a.sent) == len(old_b.sent) == 1
        assert len(new_a.sent) == len(new_b.sent) == 1

    def test_concurrent_updates_and_sends_never_observe_mixed_rules(self):
        left = FakeOutput('left')
        right = FakeOutput('right')

        rules_left = [RoutingRule([left]), RoutingRule([left])]
        rules_right = [RoutingRule([right]), RoutingRule([right])]

        router = Router(rules_left)
        errors = []
        stop = threading.Event()

        def sender():
            try:
                while not stop.is_set():
                    router.send(make_message())
            except Exception as exc:  # pragma: no cover - test failure path
                errors.append(exc)

        def updater():
            try:
                for index in range(500):
                    router.set_rules(rules_left if index % 2 else rules_right)
            except Exception as exc:  # pragma: no cover - test failure path
                errors.append(exc)

        senders = [threading.Thread(target=sender) for _ in range(2)]
        update_thread = threading.Thread(target=updater)
        for thread in senders:
            thread.start()
        update_thread.start()
        update_thread.join(10)
        stop.set()
        for thread in senders:
            thread.join(10)

        assert errors == []
        assert all(not thread.is_alive() for thread in senders + [update_thread])

        # Every message went to exactly one snapshot's pair of ports,
        # never to a mixture.
        assert len(left.sent) % 2 == 0
        assert len(right.sent) % 2 == 0
        assert left.sent or right.sent
