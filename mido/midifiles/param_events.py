# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

"""
RPN/NRPN parameter event extraction and write-back.

Registered Parameter Numbers (RPNs) and Non-Registered Parameter Numbers
(NRPNs) are not single MIDI messages. They are assembled from a sequence of
Control Change (CC) messages:

* CC 99 / CC 98 select an NRPN (MSB / LSB)
* CC 101 / CC 100 select an RPN (MSB / LSB)
* CC 6 / CC 38 set the selected parameter's value (MSB / LSB)
* CC 96 / CC 97 increment / decrement the selected parameter

Because the messages are spread out and interleaved with ordinary CC
messages, they are hard to edit directly. This module turns the raw CC
stream into structured :class:`ParameterEvent` objects and can write edited
events back into a :class:`~mido.midifiles.midifiles.MidiFile`.

A selector pair can serve several data messages in a row without being
re-sent. Such events share their :attr:`ParameterEvent.group`; on
write-back the selector CCs are emitted only for the first event of the
group.

References:

* https://www.midi.org/specifications-old/table-of-midi-messages-1-0
"""

from dataclasses import dataclass
from itertools import count
from numbers import Integral

from ..messages import Message

# Selector controls.
NRPN_MSB = 99
NRPN_LSB = 98
RPN_MSB = 101
RPN_LSB = 100

# Data entry controls.
DATA_ENTRY_MSB = 6
DATA_ENTRY_LSB = 38
DATA_INCREMENT = 96
DATA_DECREMENT = 97

#: The RPN null selector (MSB=127, LSB=127) deselects the current RPN.
RPN_NULL = 0x3FFF  # 127 << 7 | 127

_SELECTOR_CONTROLS = frozenset((RPN_MSB, RPN_LSB, NRPN_MSB, NRPN_LSB))
_DATA_CONTROLS = frozenset(
    (DATA_ENTRY_MSB, DATA_ENTRY_LSB, DATA_INCREMENT, DATA_DECREMENT)
)

#: All CC numbers that participate in RPN/NRPN assembly.
PARAMETER_CONTROLS = frozenset(_SELECTOR_CONTROLS | _DATA_CONTROLS)

MAX_7BIT = 0x7F
MAX_14BIT = 0x3FFF


def _check_int(name, value, maximum):
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f'{name} must be an integer')
    if not 0 <= value <= maximum:
        raise ValueError(f'{name} must be in range 0..{maximum}')


@dataclass(frozen=True)
class ParameterEvent:
    """A single RPN/NRPN parameter event.

    Attributes:
        track: index of the track the event was extracted from (or will be
            written to).
        channel: MIDI channel (0..15).
        tick: absolute tick within the track. For a value set this is the
            tick of the final data entry message; for a null selector it is
            the tick of the selector's last message.
        kind: ``'rpn'`` or ``'nrpn'``.
        operation: ``'set'``, ``'increment'``, ``'decrement'`` or
            ``'null'``. A ``'null'`` event represents the RPN null
            selector (127, 127), which deselects the current RPN.
        number: 14-bit parameter number (0..16383). It is ``None`` for
            ``'null'`` events.
        value: parameter value. For ``'set'`` events it is a 7-bit
            (0..127) or 14-bit (0..16383) value. For
            ``'increment'``/``'decrement'`` events it is the 7-bit step
            carried by the CC message. It is ``None`` for ``'null'``
            events.
        width: ``7`` or ``14`` for ``'set'`` events, ``7`` for
            increment/decrement events and ``None`` for ``'null'`` events.
        group: identifier shared by every event assembled under the same
            selector pair. Write-back emits the selector CCs only for the
            first event of each group. ``None`` for user-created events,
            which always emit a fresh selector pair.
    """

    track: int
    channel: int
    tick: int
    kind: str
    operation: str
    number: object = None
    value: object = None
    width: object = None
    group: object = None
    # Positions (indices into the source track) of the CC messages that
    # make up the event. Selector positions may be shared by all events of
    # a group. ``None`` for events created by the user. Not part of the
    # public data model; only used internally for write-back.
    _sources: object = None
    # True for every event except the first of a shared-selector group.
    _selector_shared: bool = False

    def __post_init__(self):
        _check_int('track', self.track, 2 ** 31 - 1)
        _check_int('channel', self.channel, 15)
        _check_int('tick', self.tick, 2 ** 31 - 1)

        if self.kind not in ('rpn', 'nrpn'):
            raise ValueError("kind must be 'rpn' or 'nrpn'")
        if self.operation not in ('set', 'increment', 'decrement', 'null'):
            raise ValueError(
                "operation must be 'set', 'increment', "
                "'decrement' or 'null'")

        if self.operation == 'null':
            if self.kind != 'rpn':
                raise ValueError("only RPN selectors can be null")
            if self.number is not None or self.value is not None \
                    or self.width is not None:
                raise ValueError(
                    "null events have no number, value or width")
            return

        _check_int('number', self.number, MAX_14BIT)

        if self.operation == 'set':
            if self.width not in (7, 14):
                raise ValueError("width must be 7 or 14 for set events")
            _check_int('value', self.value,
                       MAX_7BIT if self.width == 7 else MAX_14BIT)
        else:
            if self.width != 7:
                raise ValueError(
                    "increment/decrement events must have width 7")
            _check_int('value', self.value, MAX_7BIT)

    def with_value(self, value, width=None):
        """Return a copy of the event with a new absolute value.

        ``width`` may be passed to change the encoding width (``7`` or
        ``14``). By default the event's current width is kept; values that
        do not fit the requested width raise :class:`ValueError`.
        Increment/decrement and null events cannot be converted this way.
        """
        if self.operation == 'null':
            raise ValueError("null events do not carry a value")
        if self.operation != 'set':
            raise ValueError("only set events carry an absolute value")

        if width is None:
            width = self.width
        if width not in (7, 14):
            raise ValueError("width must be 7 or 14")
        _check_int('value', value,
                   MAX_7BIT if width == 7 else MAX_14BIT)

        return ParameterEvent(
            track=self.track,
            channel=self.channel,
            tick=self.tick,
            kind=self.kind,
            operation='set',
            number=self.number,
            value=value,
            width=width,
            group=self.group,
            _sources=self._sources,
            _selector_shared=self._selector_shared,
        )


class _ChannelState:
    """Per-channel selector assembly state (RPN and NRPN independent)."""

    def __init__(self, group_ids):
        self._group_ids = group_ids
        # Pending selector halves per kind:
        # {'rpn': {'msb': (index, value), 'lsb': ...}, 'nrpn': {...}}
        self.pending = {
            'rpn': {'msb': None, 'lsb': None},
            'nrpn': {'msb': None, 'lsb': None},
        }
        # Currently selected parameter:
        # (kind, number, [selector source indices], group_id) or None
        self.selection = None
        # Most recent Data Entry MSB awaiting a possible LSB:
        # (index, value, tick) or None
        self.pending_data_msb = None

    def new_group(self):
        return next(self._group_ids)


def extract_parameter_events(mid):
    """Extract RPN/NRPN parameter events from a MIDI file.

    Every track is scanned independently while keeping separate assembly
    state per MIDI channel, so channels can never interfere with each
    other. Messages that do not belong to a complete, selected parameter
    sequence (ordinary CCs, half-finished selectors, data messages without
    a selection, ...) do not produce events and are left untouched.

    A Data Entry MSB (CC 6) becomes a 7-bit ``set`` event as soon as it is
    clear no LSB (CC 38) follows: when another parameter control arrives,
    when the selection changes, or at the end of the track. The event is
    stamped with the MSB's own tick regardless of when the flush happens.

    Returns a list of :class:`ParameterEvent` objects ordered by track and
    then by completion order inside the track.
    """
    events = []

    for track_index, track in enumerate(mid.tracks):
        group_ids = count()
        states = [_ChannelState(group_ids) for _ in range(16)]
        tick = 0

        for msg_index, msg in enumerate(track):
            tick += msg.time

            if msg.type != 'control_change':
                continue

            state = states[msg.channel]
            control = msg.control
            value = msg.value

            if control in (RPN_MSB, RPN_LSB):
                _flush_data_msb(state, track_index, msg.channel,
                                tick, events)
                _selector_half(state, 'rpn', control == RPN_MSB,
                               msg_index, value, track_index, msg.channel,
                               tick, events)
            elif control in (NRPN_MSB, NRPN_LSB):
                _flush_data_msb(state, track_index, msg.channel,
                                tick, events)
                _selector_half(state, 'nrpn', control == NRPN_MSB,
                               msg_index, value, track_index, msg.channel,
                               tick, events)
            elif control == DATA_ENTRY_MSB:
                if state.selection is not None:
                    # A second CC 6 finalizes the previous 7-bit entry
                    # instead of silently replacing it.
                    _flush_data_msb(state, track_index, msg.channel,
                                    tick, events)
                    state.pending_data_msb = (msg_index, value, tick)
                # Without a selection CC 6 is an ordinary CC.
            elif control == DATA_ENTRY_LSB:
                _data_entry_lsb(state, msg_index, value, track_index,
                                msg.channel, tick, events)
            elif control in (DATA_INCREMENT, DATA_DECREMENT):
                _inc_dec(state, control, msg_index, value, track_index,
                         msg.channel, tick, events)
            # Any other CC is ordinary and never affects parameter state.

        # Anything still pending at the end of the track is a 7-bit set.
        for channel, state in enumerate(states):
            _flush_data_msb(state, track_index, channel, tick, events)

    return events


def _flush_data_msb(state, track_index, channel, tick, events):
    """Complete a Data Entry MSB without an LSB as a 7-bit set event.

    The ``tick`` argument is only used to decide *when* the flush happens;
    the event itself is stamped with the MSB's own tick, which is when the
    value actually takes effect.
    """
    if state.pending_data_msb is None or state.selection is None:
        return

    msb_index, msb_value, msb_tick = state.pending_data_msb
    state.pending_data_msb = None
    kind, number, selector_sources, group_id = state.selection

    events.append(ParameterEvent(
        track=track_index,
        channel=channel,
        tick=msb_tick,
        kind=kind,
        operation='set',
        number=number,
        value=msb_value,
        width=7,
        group=group_id,
        _sources=list(selector_sources) + [msb_index],
        _selector_shared=state.group_used,
    ))
    state.group_used = True


def _selector_half(state, kind, is_msb, msg_index, value, track_index,
                   channel, tick, events):
    half = 'msb' if is_msb else 'lsb'
    state.pending[kind][half] = (msg_index, value)

    msb = state.pending[kind]['msb']
    lsb = state.pending[kind]['lsb']
    if msb is None or lsb is None:
        # Half-finished selector: wait for the other half.
        return

    number = (msb[1] << 7) | lsb[1]
    sources = [msb[0], lsb[0]]
    state.pending[kind] = {'msb': None, 'lsb': None}
    state.pending_data_msb = None

    if kind == 'rpn' and number == RPN_NULL:
        # Null RPN: deselect whatever is currently selected.
        state.selection = None
        events.append(ParameterEvent(
            track=track_index,
            channel=channel,
            tick=tick,
            kind='rpn',
            operation='null',
            group=state.new_group(),
            _sources=sources,
        ))
    else:
        state.selection = (kind, number, sources, state.new_group())
        state.group_used = False


def _data_entry_lsb(state, msg_index, value, track_index, channel, tick,
                    events):
    if state.selection is None or state.pending_data_msb is None:
        # CC 38 without a preceding CC 6 (or without a selection) is an
        # ordinary CC.
        return

    msb_index, msb_value, _msb_tick = state.pending_data_msb
    state.pending_data_msb = None
    kind, number, selector_sources, group_id = state.selection

    events.append(ParameterEvent(
        track=track_index,
        channel=channel,
        tick=tick,
        kind=kind,
        operation='set',
        number=number,
        value=(msb_value << 7) | value,
        width=14,
        group=group_id,
        _sources=list(selector_sources) + [msb_index, msg_index],
        _selector_shared=state.group_used,
    ))
    state.group_used = True


def _inc_dec(state, control, msg_index, value, track_index, channel, tick,
             events):
    if state.selection is None:
        # No parameter selected: CC 96/97 are ordinary CCs.
        return

    _flush_data_msb(state, track_index, channel, tick, events)
    kind, number, selector_sources, group_id = state.selection
    operation = ('increment' if control == DATA_INCREMENT
                 else 'decrement')

    events.append(ParameterEvent(
        track=track_index,
        channel=channel,
        tick=tick,
        kind=kind,
        operation=operation,
        number=number,
        value=value,
        width=7,
        group=group_id,
        _sources=list(selector_sources) + [msg_index],
        _selector_shared=state.group_used,
    ))
    state.group_used = True


def _cc(channel, control, value):
    return Message('control_change', channel=channel,
                   control=control, value=value, time=0)


def _selector_messages(event):
    if event.operation == 'null':
        return [
            _cc(event.channel, RPN_MSB, MAX_7BIT),
            _cc(event.channel, RPN_LSB, MAX_7BIT),
        ]

    msb_control = RPN_MSB if event.kind == 'rpn' else NRPN_MSB
    lsb_control = RPN_LSB if event.kind == 'rpn' else NRPN_LSB
    return [
        _cc(event.channel, msb_control, (event.number >> 7) & MAX_7BIT),
        _cc(event.channel, lsb_control, event.number & MAX_7BIT),
    ]


def _data_messages(event):
    if event.operation == 'null':
        return []
    if event.operation == 'set':
        if event.width == 14:
            return [
                _cc(event.channel, DATA_ENTRY_MSB,
                    (event.value >> 7) & MAX_7BIT),
                _cc(event.channel, DATA_ENTRY_LSB,
                    event.value & MAX_7BIT),
            ]
        return [_cc(event.channel, DATA_ENTRY_MSB, event.value)]
    if event.operation == 'increment':
        return [_cc(event.channel, DATA_INCREMENT, event.value)]
    return [_cc(event.channel, DATA_DECREMENT, event.value)]


def event_messages(event):
    """Build the canonical CC message sequence for an event.

    The selector pair is always included; use :func:`event_data_messages`
    for events that reuse an already-emitted selector. The messages are
    returned with ``time=0``; write-back assigns timing.
    """
    return _selector_messages(event) + _data_messages(event)


def event_data_messages(event):
    """Build only the data CC messages for an event (no selector pair)."""
    return _data_messages(event)


def write_parameter_events(mid, events):
    """Write parameter events back into a MIDI file.

    The tracks of ``mid`` are rebuilt in place. Events extracted from the
    file replace exactly the CC messages they were assembled from; events
    created without source messages are inserted into their target track
    at their absolute tick. Every other message (ordinary CCs, meta
    messages, notes, ...) keeps its original order and timing, and the
    relative timing of same-tick interleaved messages is preserved.

    Events sharing a :attr:`ParameterEvent.group` reuse one selector pair:
    only the first event of each group emits selector CCs.

    All events are validated before anything is written, so an invalid
    event leaves the file untouched. :class:`ValueError` is raised for
    unknown tracks, out-of-range ticks or events whose channel does not
    match the channel of the messages they replace.
    """
    events_by_track = _group_and_validate(mid, events)

    for track_index, track_events in events_by_track.items():
        mid.tracks[track_index] = _rewrite_track(
            mid.tracks[track_index], track_events)


def _group_and_validate(mid, events):
    events_by_track = {}
    for event in events:
        if not isinstance(event, ParameterEvent):
            raise TypeError(
                'events must be ParameterEvent instances, got '
                f'{type(event).__name__}')
        if not 0 <= event.track < len(mid.tracks):
            raise ValueError(
                f'event references track {event.track} but the file only '
                f'has {len(mid.tracks)} track(s)')
        events_by_track.setdefault(event.track, []).append(event)

    for track_index, track_events in events_by_track.items():
        _validate_track_events(mid.tracks[track_index], track_events)

    return events_by_track


def _expected_source_controls(event):
    """The CC control numbers expected at each of the event's sources.

    Extracted events always record the selector pair first (even for
    shared groups), followed by their own data message(s). Null events
    only carry the selector pair. When a set event's width is edited the
    source list may be shorter than the new width requires; such extra
    data messages are generated during write-back, so this function is
    only consulted for the positions that actually exist.
    """
    if event.kind == 'rpn':
        selector = [RPN_MSB, RPN_LSB]
    else:
        selector = [NRPN_MSB, NRPN_LSB]
    if event.operation == 'null':
        return selector
    if event.operation == 'set':
        data = ([DATA_ENTRY_MSB, DATA_ENTRY_LSB] if event.width == 14
                else [DATA_ENTRY_MSB])
    elif event.operation == 'increment':
        data = [DATA_INCREMENT]
    else:
        data = [DATA_DECREMENT]
    return selector + data


def _validate_track_events(track, events):
    # Data message positions must be claimed by exactly one event;
    # selector positions may be shared within a group.
    data_claimed = set()
    selector_claims = {}
    for event in events:
        if event._sources is None:
            continue
        selector_positions = set(event._sources[:2])
        expected_controls = _expected_source_controls(event)
        # Every sourced event at least carries its selector pair.
        if len(event._sources) < 2:
            raise ValueError(
                'event source count does not match its operation and width')
        for offset, position in enumerate(event._sources):
            expected_control = (
                expected_controls[offset]
                if offset < len(expected_controls)
                else None)
            if not isinstance(position, Integral) \
                    or not 0 <= position < len(track):
                raise ValueError(
                    'event source message no longer exists in the track')

            msg = track[position]
            if msg.type != 'control_change' \
                    or msg.channel != event.channel:
                raise ValueError(
                    'event channel does not match its source messages')
            if expected_control is not None \
                    and msg.control != expected_control:
                raise ValueError(
                    'event source messages do not match its operation '
                    f'(expected CC {expected_control}, found '
                    f'CC {msg.control})')

            if position in selector_positions:
                if position in selector_claims \
                        and selector_claims[position] != event.group:
                    raise ValueError(
                        'selector source reused by a different group')
                selector_claims[position] = event.group
            else:
                if position in data_claimed:
                    raise ValueError(
                        'two events claim the same data message')
                data_claimed.add(position)

    last_tick = sum(msg.time for msg in track)
    for event in events:
        if event.tick < 0 or event.tick > last_tick:
            raise ValueError(
                f'event tick {event.tick} is outside the track '
                f'(0..{last_tick})')

    # Multiple new (sourceless) events at the same tick/channel would have
    # ambiguous ordering.
    seen = set()
    for event in events:
        if event._sources is not None:
            continue
        key = (event.channel, event.tick)
        if key in seen:
            raise ValueError(
                'multiple new events share a tick on the same channel; '
                'give them distinct ticks')
        seen.add(key)


def _rewrite_track(track, events):
    # The first event of each group owns the selector pair and emits it
    # at the selector's original tick; every event emits its data at its
    # own primary data tick.
    primary_by_group = {}
    for event in events:
        if event._sources is not None:
            primary_by_group.setdefault(event.group, event)

    # Every claimed source position is dropped from the output.
    claimed = set()
    # Selector emission tick per group (tick of the last selector CC).
    selector_tick = {}
    # Data emission: primary data position -> event.
    data_at = {}
    for event in events:
        if event._sources is None:
            continue
        primary = primary_by_group[event.group]
        claimed.update(event._sources)
        selector_tick[event.group] = _tick_of(
            track, primary._sources[1])
        data_at[event._sources[-1]] = event

    # Groups whose selector still needs to be emitted while walking the
    # track. New (sourceless) events carry their own selector.
    pending_groups = {id(primary): primary
                      for primary in primary_by_group.values()}

    # New (sourceless) events, grouped by insertion tick.
    new_by_tick = {}
    for event in events:
        if event._sources is None:
            new_by_tick.setdefault(event.tick, []).append(event)

    # Build an absolute-tick stream of (tick, message) pairs.
    stream = []
    tick = 0
    for index, msg in enumerate(track):
        tick += msg.time

        # Insertions whose tick has been passed without an exact match
        # go here, immediately before the current message.
        for due_tick in sorted(t for t in new_by_tick if t <= tick):
            for new_event in new_by_tick.pop(due_tick):
                for generated_msg in event_messages(new_event):
                    stream.append((due_tick, generated_msg))

        # Emit any group selector whose tick has arrived.
        for primary in list(pending_groups.values()):
            if selector_tick[primary.group] == tick:
                for generated_msg in _selector_messages(primary):
                    stream.append((tick, generated_msg))
                del pending_groups[id(primary)]

        if index in claimed:
            if index in data_at:
                # Data messages are emitted without a selector: the
                # selector was (or will be) emitted at its own tick.
                event = data_at[index]
                for generated_msg in _data_messages(event):
                    stream.append((tick, generated_msg))
            # Selector slots and shared positions produce nothing.
        else:
            stream.append((tick, msg))

        # Insertions exactly at this tick are appended after the original
        # message(s), which keeps every delta non-negative.
        for new_event in new_by_tick.pop(tick, []):
            for generated_msg in event_messages(new_event):
                stream.append((tick, generated_msg))

    # New events placed at or after the last original message.
    for late_tick in sorted(new_by_tick):
        for new_event in new_by_tick[late_tick]:
            for generated_msg in event_messages(new_event):
                stream.append((late_tick, generated_msg))

    # Convert the absolute-tick stream into delta times.
    result = track.__class__()
    previous_tick = 0
    for absolute_tick, msg in stream:
        result.append(msg.copy(time=absolute_tick - previous_tick))
        previous_tick = absolute_tick

    return result


def _tick_of(track, position):
    tick = 0
    for msg in track[:position + 1]:
        tick += msg.time
    return tick
