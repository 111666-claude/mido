# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

import io
from copy import deepcopy

import pytest

from mido import Message, MetaMessage, MidiFile, MidiTrack
from mido.midifiles.param_events import (
    DATA_DECREMENT,
    DATA_ENTRY_LSB,
    DATA_ENTRY_MSB,
    DATA_INCREMENT,
    NRPN_LSB,
    NRPN_MSB,
    RPN_LSB,
    RPN_MSB,
    RPN_NULL,
    ParameterEvent,
    event_messages,
    extract_parameter_events,
    write_parameter_events,
)


def cc(channel, control, value, time=0):
    return Message('control_change', channel=channel, control=control,
                   value=value, time=time)


def build_fixture():
    """A two-track, two-channel file covering RPN, NRPN, 7/14-bit and
    increment/decrement sequences plus ordinary CC and meta messages."""
    track0 = MidiTrack([
        MetaMessage('track_name', name='Param Track', time=0),
        # Channel 0: RPN 0 (pitch bend range), 14-bit value 0x0200 = 512,
        # spread over several ticks.
        cc(0, RPN_MSB, 0x00, time=0),
        cc(0, RPN_LSB, 0x00, time=10),
        cc(0, DATA_ENTRY_MSB, 0x04, time=5),
        cc(0, DATA_ENTRY_LSB, 0x00, time=7),
        # An ordinary, unrelated CC interleaved at the same tick as the
        # next selector.
        cc(1, 7, 64, time=20),  # channel 1 volume at tick 42
        # Channel 0: NRPN 1 (0x0001), 7-bit value 100.
        cc(0, NRPN_MSB, 0x00, time=0),
        cc(0, NRPN_LSB, 0x01, time=3),
        cc(0, DATA_ENTRY_MSB, 100, time=11),
        # Channel 0: increment the still-selected NRPN by 2.
        cc(0, DATA_INCREMENT, 2, time=8),
        # Null RPN selector deselects.
        cc(0, RPN_MSB, 127, time=4),
        cc(0, RPN_LSB, 127, time=0),
        # Data entry after null selection must stay an ordinary CC.
        cc(0, DATA_ENTRY_MSB, 99, time=6),
        MetaMessage('end_of_track', time=100),
    ])

    track1 = MidiTrack([
        MetaMessage('track_name', name='Other Track', time=0),
        # Channel 2 (different channel, same tick numbers, must not
        # cross-talk): NRPN 0x1000, 14-bit value 0x1234.
        cc(2, NRPN_MSB, 0x20, time=0),
        cc(2, NRPN_LSB, 0x00, time=10),
        cc(2, DATA_ENTRY_MSB, 0x24, time=0),
        cc(2, DATA_ENTRY_LSB, 0x34, time=0),
        # Channel 2: decrement the selected NRPN by 1.
        cc(2, DATA_DECREMENT, 1, time=30),
        # Channel 3: half-finished selector, never completed.
        cc(3, RPN_MSB, 0x01, time=5),
        cc(3, DATA_ENTRY_MSB, 50, time=5),  # no selection -> ordinary CC
        # Orphan data entry LSB, no preceding MSB.
        cc(3, DATA_ENTRY_LSB, 7, time=5),
        # Note messages and an ordinary CC must be preserved untouched.
        Message('note_on', channel=2, note=60, velocity=64, time=10),
        cc(2, 64, 127, time=0),  # sustain pedal at same tick
        Message('note_off', channel=2, note=60, velocity=64, time=40),
        MetaMessage('end_of_track', time=0),
    ])

    return MidiFile(type=1, ticks_per_beat=480, tracks=[track0, track1])


def save_and_reload(mid):
    buffer = io.BytesIO()
    mid.save(file=buffer)
    buffer.seek(0)
    return MidiFile(file=buffer)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def test_extract_basic_events():
    mid = build_fixture()
    events = extract_parameter_events(mid)

    # Track 0: RPN set, NRPN set, NRPN increment, null RPN.
    track0 = [e for e in events if e.track == 0]
    assert len(track0) == 4

    rpn_set, nrpn_set, nrpn_inc, null = track0

    assert rpn_set.kind == 'rpn'
    assert rpn_set.operation == 'set'
    assert rpn_set.channel == 0
    assert rpn_set.number == 0
    assert rpn_set.value == 0x0200
    assert rpn_set.width == 14
    # Tick of the final data entry LSB: 10+5+7 after tick 0.
    assert rpn_set.tick == 22

    assert nrpn_set.kind == 'nrpn'
    assert nrpn_set.operation == 'set'
    assert nrpn_set.number == 1
    assert nrpn_set.value == 100
    assert nrpn_set.width == 7
    assert nrpn_set.tick == 22 + 20 + 3 + 11

    assert nrpn_inc.operation == 'increment'
    assert nrpn_inc.kind == 'nrpn'
    assert nrpn_inc.number == 1
    assert nrpn_inc.value == 2
    assert nrpn_inc.width == 7

    assert null.operation == 'null'
    assert null.kind == 'rpn'
    assert null.number is None
    assert null.value is None
    assert null.width is None


def test_extract_second_track_and_channel_isolation():
    mid = build_fixture()
    events = extract_parameter_events(mid)

    track1 = [e for e in events if e.track == 1]
    assert len(track1) == 2

    nrpn_set, nrpn_dec = track1
    assert nrpn_set.channel == 2
    assert nrpn_set.kind == 'nrpn'
    assert nrpn_set.number == 0x1000
    assert nrpn_set.value == 0x1234
    assert nrpn_set.width == 14
    assert nrpn_set.tick == 10

    assert nrpn_dec.operation == 'decrement'
    assert nrpn_dec.channel == 2
    assert nrpn_dec.number == 0x1000
    assert nrpn_dec.value == 1

    # Channel 3 produced nothing: half selector and orphan data stay CCs.
    assert all(e.channel != 3 for e in events)


def test_ordinary_ccs_are_not_recognized():
    mid = build_fixture()
    events = extract_parameter_events(mid)
    claimed = {
        (event.track, position)
        for event in events
        for position in (event._sources or [])
    }

    for track_index, track in enumerate(mid.tracks):
        for index, msg in enumerate(track):
            if msg.type != 'control_change':
                continue
            if msg.control in (RPN_MSB, RPN_LSB, NRPN_MSB, NRPN_LSB,
                               DATA_ENTRY_MSB, DATA_ENTRY_LSB,
                               DATA_INCREMENT, DATA_DECREMENT):
                # Parameter-control messages are either claimed or
                # deliberately left as ordinary CCs.
                continue
            # Non-parameter CCs must never be claimed.
            assert (track_index, index) not in claimed

    # Explicit ordinary messages from the fixture.
    track0 = mid.tracks[0]
    assert any(m.is_cc(7) and m.channel == 1 for m in track0)
    assert any(m.is_cc(DATA_ENTRY_MSB) and m.value == 99
               and m.channel == 0 for m in track0)
    track1 = mid.tracks[1]
    assert any(m.is_cc(64) for m in track1)


def test_half_selector_and_missing_data_produce_no_events():
    track = MidiTrack([
        cc(0, RPN_MSB, 1),                 # only MSB, no LSB
        cc(0, DATA_ENTRY_MSB, 10),         # no complete selection
        cc(0, RPN_LSB, 2),                 # completes selector 0x82
        cc(0, DATA_ENTRY_LSB, 5),          # orphan LSB, no MSB
        MetaMessage('end_of_track', time=0),
    ])
    mid = MidiFile(tracks=[track])
    events = extract_parameter_events(mid)
    assert events == []


def test_data_entry_without_selection_is_ordinary_cc():
    track = MidiTrack([
        cc(0, DATA_ENTRY_MSB, 11, time=0),
        cc(0, DATA_ENTRY_LSB, 22, time=0),
        cc(0, DATA_INCREMENT, 1, time=0),
        cc(0, DATA_DECREMENT, 1, time=0),
        MetaMessage('end_of_track', time=0),
    ])
    mid = MidiFile(tracks=[track])
    assert extract_parameter_events(mid) == []
    # All four messages survive write-back untouched.
    write_parameter_events(mid, [])
    saved = save_and_reload(mid)
    ccs = [m for m in saved.tracks[0] if m.type == 'control_change']
    assert [(m.control, m.value) for m in ccs] == [
        (DATA_ENTRY_MSB, 11),
        (DATA_ENTRY_LSB, 22),
        (DATA_INCREMENT, 1),
        (DATA_DECREMENT, 1),
    ]


def test_seven_bit_set_flushed_on_selector_change():
    # A Data Entry MSB followed by a new selector (no LSB ever arrives)
    # must be recognized as a 7-bit set at the MSB's own tick.
    track = MidiTrack([
        cc(0, NRPN_MSB, 0, time=0),
        cc(0, NRPN_LSB, 5, time=0),       # NRPN 5
        cc(0, DATA_ENTRY_MSB, 42, time=10),
        cc(0, NRPN_MSB, 0, time=10),      # new selector flushes the set
        cc(0, NRPN_LSB, 6, time=0),       # NRPN 6
        MetaMessage('end_of_track', time=0),
    ])
    mid = MidiFile(tracks=[track])
    events = extract_parameter_events(mid)

    assert len(events) == 1
    event = events[0]
    assert event.operation == 'set'
    assert event.kind == 'nrpn'
    assert event.number == 5
    assert event.value == 42
    assert event.width == 7
    # Stamped at the Data Entry MSB's own tick, not the flush tick.
    assert event.tick == 10


def test_seven_bit_set_flushed_at_end_of_track():
    track = MidiTrack([
        cc(0, RPN_MSB, 0, time=0),
        cc(0, RPN_LSB, 1, time=0),
        cc(0, DATA_ENTRY_MSB, 77, time=15),
        MetaMessage('end_of_track', time=0),
    ])
    mid = MidiFile(tracks=[track])
    events = extract_parameter_events(mid)
    assert len(events) == 1
    assert events[0].value == 77
    assert events[0].width == 7
    assert events[0].tick == 15


def test_rpn_and_nrpn_selectors_are_independent():
    # RPN MSB pending while a full NRPN sequence completes; the pending
    # RPN half must not be consumed by the NRPN.
    track = MidiTrack([
        cc(0, RPN_MSB, 3, time=0),         # pending RPN MSB
        cc(0, NRPN_MSB, 0, time=0),
        cc(0, NRPN_LSB, 9, time=0),        # NRPN 9
        cc(0, DATA_ENTRY_MSB, 1, time=0),  # 7-bit set on NRPN 9
        cc(0, RPN_LSB, 4, time=0),         # completes RPN 0x184
        cc(0, DATA_ENTRY_MSB, 2, time=0),  # 7-bit set on RPN 0x184
        MetaMessage('end_of_track', time=0),
    ])
    mid = MidiFile(tracks=[track])
    events = extract_parameter_events(mid)

    assert len(events) == 2
    nrpn, rpn = events
    assert (nrpn.kind, nrpn.number, nrpn.value) == ('nrpn', 9, 1)
    assert (rpn.kind, rpn.number, rpn.value) == ('rpn', 0x184, 2)


def test_same_tick_interleaving_two_channels():
    # Both channels run complete sequences at identical ticks; state must
    # stay strictly per channel.
    track = MidiTrack([
        cc(0, NRPN_MSB, 0, time=0),
        cc(1, NRPN_MSB, 0, time=0),
        cc(0, NRPN_LSB, 1, time=0),
        cc(1, NRPN_LSB, 2, time=0),
        cc(0, DATA_ENTRY_MSB, 10, time=0),
        cc(1, DATA_ENTRY_MSB, 20, time=0),
        cc(0, DATA_ENTRY_LSB, 0, time=0),
        cc(1, DATA_ENTRY_LSB, 0, time=0),
        MetaMessage('end_of_track', time=0),
    ])
    mid = MidiFile(tracks=[track])
    events = extract_parameter_events(mid)

    by_channel = {e.channel: e for e in events}
    assert by_channel[0].number == 1
    assert by_channel[0].value == (10 << 7)
    assert by_channel[1].number == 2
    assert by_channel[1].value == (20 << 7)


def test_null_selector_deselects():
    track = MidiTrack([
        cc(0, RPN_MSB, 0, time=0),
        cc(0, RPN_LSB, 0, time=0),        # RPN 0 selected
        cc(0, DATA_ENTRY_MSB, 5, time=0),  # 7-bit set
        cc(0, RPN_MSB, 127, time=0),
        cc(0, RPN_LSB, 127, time=0),      # null
        cc(0, DATA_ENTRY_MSB, 6, time=0),  # ordinary now
        MetaMessage('end_of_track', time=0),
    ])
    mid = MidiFile(tracks=[track])
    events = extract_parameter_events(mid)

    assert len(events) == 2
    set_event, null_event = events
    assert set_event.operation == 'set'
    assert set_event.value == 5
    assert null_event.operation == 'null'
    assert null_event.tick == 0


# ---------------------------------------------------------------------------
# ParameterEvent validation
# ---------------------------------------------------------------------------

def test_invalid_event_construction():
    kwargs = {'track': 0, 'channel': 0, 'tick': 0, 'kind': 'rpn',
              'operation': 'set', 'number': 0, 'value': 0, 'width': 7}

    with pytest.raises(ValueError):
        ParameterEvent(**{**kwargs, 'channel': 16})
    with pytest.raises(ValueError):
        ParameterEvent(**{**kwargs, 'value': 128})  # too big for 7-bit
    with pytest.raises(ValueError):
        ParameterEvent(**{**kwargs, 'kind': 'nope'})
    with pytest.raises(ValueError):
        ParameterEvent(**{**kwargs, 'operation': 'frobnicate'})
    with pytest.raises(ValueError):
        ParameterEvent(track=0, channel=0, tick=0, kind='nrpn',
                       operation='null')
    with pytest.raises(ValueError):
        ParameterEvent(**{**kwargs, 'width': 14, 'value': 20000})
    with pytest.raises(TypeError):
        ParameterEvent(**{**kwargs, 'value': 1.5})
    with pytest.raises(TypeError):
        ParameterEvent(**{**kwargs, 'value': True})  # bool is not an int here


def test_event_messages_canonical_sequence():
    event = ParameterEvent(track=0, channel=3, tick=10, kind='nrpn',
                           operation='set', number=0x1234,
                           value=0x3678, width=14)
    messages = event_messages(event)
    assert [(m.control, m.value) for m in messages] == [
        (NRPN_MSB, 0x24),
        (NRPN_LSB, 0x34),
        (DATA_ENTRY_MSB, 0x6C),
        (DATA_ENTRY_LSB, 0x78),
    ]
    assert all(m.channel == 3 for m in messages)

    null = ParameterEvent(track=0, channel=0, tick=0, kind='rpn',
                          operation='null')
    assert [(m.control, m.value) for m in event_messages(null)] == [
        (RPN_MSB, 127),
        (RPN_LSB, 127),
    ]


def test_with_value():
    event = ParameterEvent(track=0, channel=0, tick=0, kind='rpn',
                           operation='set', number=0, value=10, width=7)
    wider = event.with_value(300, width=14)
    assert wider.width == 14
    assert wider.value == 300
    assert wider._sources == event._sources

    narrow = wider.with_value(42, width=7)
    assert narrow.width == 7
    assert narrow.value == 42

    # A value that does not fit the requested width is rejected.
    with pytest.raises(ValueError):
        event.with_value(128, width=7)

    null = ParameterEvent(track=0, channel=0, tick=0, kind='rpn',
                          operation='null')
    with pytest.raises(ValueError):
        null.with_value(1)


# ---------------------------------------------------------------------------
# Write-back
# ---------------------------------------------------------------------------

def test_round_trip_preserves_file_when_unmodified():
    mid = build_fixture()
    original = deepcopy(mid)

    events = extract_parameter_events(mid)
    write_parameter_events(mid, events)

    # Track structure and every non-replaced message is unchanged.
    assert len(mid.tracks) == len(original.tracks)
    for new_track, old_track in zip(mid.tracks, original.tracks):
        assert [m.type for m in new_track] == [m.type for m in old_track]
        assert sum(m.time for m in new_track) == \
            sum(m.time for m in old_track)

    reloaded = save_and_reload(mid)
    again = extract_parameter_events(reloaded)
    assert [(e.track, e.channel, e.tick, e.kind, e.operation,
             e.number, e.value, e.width) for e in again] == \
        [(e.track, e.channel, e.tick, e.kind, e.operation,
          e.number, e.value, e.width) for e in events]


def test_modify_values_and_write_back():
    mid = build_fixture()
    events = extract_parameter_events(mid)

    modified = []
    for event in events:
        if event.operation == 'set' and event.width == 14:
            modified.append(event.with_value(0x3FFF))
        elif event.operation == 'set':
            modified.append(event.with_value(127))
        else:
            modified.append(event)

    write_parameter_events(mid, modified)
    reloaded = save_and_reload(mid)
    new_events = extract_parameter_events(reloaded)

    assert len(new_events) == len(events)
    for old, new in zip(events, new_events):
        assert (new.track, new.channel, new.tick, new.kind,
                new.operation, new.number) == \
            (old.track, old.channel, old.tick, old.kind,
             old.operation, old.number)
        if old.operation == 'set':
            assert new.value == 0x3FFF if old.width == 14 else 127
            assert new.width == old.width
        else:
            assert new.value == old.value


def test_width_change_expands_and_shrinks_message_groups():
    mid = build_fixture()
    events = extract_parameter_events(mid)

    # The NRPN 7-bit set becomes 14-bit (one extra CC).
    target = next(e for e in events
                  if e.operation == 'set' and e.width == 7)
    changed = target.with_value(1000, width=14)
    assert changed.width == 14

    new_events = [changed if e is target else e for e in events]
    write_parameter_events(mid, new_events)

    reloaded = save_and_reload(mid)
    again = extract_parameter_events(reloaded)
    found = next(e for e in again
                 if e.track == target.track and e.number == target.number
                 and e.operation == 'set')
    assert found.width == 14
    assert found.value == 1000

    # And shrinking back to 7-bit restores the original message count.
    original_cc_count = sum(
        1 for t in build_fixture().tracks for m in t
        if m.type == 'control_change')
    shrunk = found.with_value(5, width=7)
    assert shrunk.width == 7
    write_parameter_events(
        reloaded,
        [shrunk if e is found else e for e in again])
    reloaded_twice = save_and_reload(reloaded)
    cc_count = sum(
        1 for t in reloaded_twice.tracks for m in t
        if m.type == 'control_change')
    assert cc_count == original_cc_count


def test_unrelated_messages_and_meta_are_preserved():
    mid = build_fixture()
    original_bytes = io.BytesIO()
    build_fixture().save(file=original_bytes)

    events = extract_parameter_events(mid)
    write_parameter_events(mid, events)
    reloaded = save_and_reload(mid)

    for track_index, track in enumerate(reloaded.tracks):
        original_track = build_fixture().tracks[track_index]
        # Notes and meta messages compare equal including delta time.
        for new_msg, old_msg in zip(
                (m for m in track if m.type != 'control_change'),
                (m for m in original_track
                 if m.type != 'control_change')):
            assert new_msg == old_msg

        # Ordinary CCs (volume, sustain, post-null data, orphans) keep
        # their control, value and delta time.
        ordinary_new = [
            m for m in track if m.type == 'control_change'
            and m.control not in (RPN_MSB, RPN_LSB, NRPN_MSB, NRPN_LSB,
                                  DATA_ENTRY_MSB, DATA_ENTRY_LSB,
                                  DATA_INCREMENT, DATA_DECREMENT)]
        ordinary_old = [
            m for m in original_track if m.type == 'control_change'
            and m.control not in (RPN_MSB, RPN_LSB, NRPN_MSB, NRPN_LSB,
                                  DATA_ENTRY_MSB, DATA_ENTRY_LSB,
                                  DATA_INCREMENT, DATA_DECREMENT)]
        assert ordinary_new == ordinary_old

        # Parameter-control CCs left unrecognized (post-null CC 6,
        # orphan CC 38, half-selector CC 101) must also survive verbatim.
        def unrecognized(messages):
            return [
                m for m in messages
                if m.type == 'control_change'
                and (m.control == DATA_ENTRY_MSB and m.value == 99
                     or m.control == DATA_ENTRY_LSB and m.value == 7
                     or m.control == RPN_MSB and m.channel == 3
                     or m.control == DATA_ENTRY_MSB and m.channel == 3
                     and m.value == 50)]

        assert unrecognized(track) == unrecognized(original_track)


def test_insert_new_event():
    mid = build_fixture()
    new_event = ParameterEvent(
        track=1, channel=5, tick=25, kind='nrpn', operation='set',
        number=0x0042, value=0x1122, width=14)

    write_parameter_events(mid, extract_parameter_events(mid)
                           + [new_event])
    reloaded = save_and_reload(mid)
    found = next(e for e in extract_parameter_events(reloaded)
                 if e.track == 1 and e.channel == 5)
    assert found.number == 0x0042
    assert found.value == 0x1122
    assert found.tick == 25


def test_write_back_validation_errors():
    mid = build_fixture()
    events = extract_parameter_events(mid)

    # Unknown track.
    bad = ParameterEvent(track=9, channel=0, tick=0, kind='rpn',
                         operation='set', number=0, value=1, width=7)
    with pytest.raises(ValueError):
        write_parameter_events(mid, events + [bad])

    # Tick outside the track.
    bad_tick = ParameterEvent(
        track=0, channel=0, tick=10 ** 9, kind='rpn', operation='set',
        number=0, value=1, width=7)
    with pytest.raises(ValueError):
        write_parameter_events(mid, events + [bad_tick])

    # Two new events at the same tick/channel.
    a = ParameterEvent(track=0, channel=9, tick=5, kind='nrpn',
                       operation='set', number=1, value=1, width=7)
    b = ParameterEvent(track=0, channel=9, tick=5, kind='nrpn',
                       operation='set', number=2, value=2, width=7)
    with pytest.raises(ValueError):
        write_parameter_events(mid, [a, b])

    # Non ParameterEvent object.
    with pytest.raises(TypeError):
        write_parameter_events(mid, events + ['not an event'])

    # File untouched after the failed calls.
    fresh = build_fixture()
    for t1, t2 in zip(mid.tracks, fresh.tracks):
        assert t1 == t2


def test_channel_mismatch_with_sources_rejected():
    mid = build_fixture()
    events = extract_parameter_events(mid)
    target = events[0]
    stolen = ParameterEvent(
        track=target.track, channel=15, tick=target.tick,
        kind=target.kind, operation=target.operation,
        number=target.number, value=target.value, width=target.width,
        _sources=target._sources)
    with pytest.raises(ValueError):
        write_parameter_events(mid, [stolen])


def test_same_tick_interleaving_survives_round_trip():
    track = MidiTrack([
        cc(0, NRPN_MSB, 0, time=0),
        cc(1, NRPN_MSB, 0, time=0),
        cc(0, NRPN_LSB, 1, time=0),
        cc(1, NRPN_LSB, 2, time=0),
        cc(0, DATA_ENTRY_MSB, 10, time=0),
        cc(1, DATA_ENTRY_MSB, 20, time=0),
        cc(0, DATA_ENTRY_LSB, 0, time=5),
        cc(1, DATA_ENTRY_LSB, 0, time=0),
        Message('note_on', channel=0, note=60, velocity=64, time=5),
        MetaMessage('end_of_track', time=0),
    ])
    mid = MidiFile(tracks=[track])
    events = extract_parameter_events(mid)
    write_parameter_events(mid, events)
    reloaded = save_and_reload(mid)

    # The note keeps its absolute tick (10) and the track length is
    # unchanged despite the rewritten CC stream.
    def absolute_ticks(track):
        ticks = []
        running = 0
        for message in track:
            running += message.time
            ticks.append(running)
        return ticks

    original = MidiFile(tracks=[MidiTrack([
        cc(0, NRPN_MSB, 0, time=0),
        cc(1, NRPN_MSB, 0, time=0),
        cc(0, NRPN_LSB, 1, time=0),
        cc(1, NRPN_LSB, 2, time=0),
        cc(0, DATA_ENTRY_MSB, 10, time=0),
        cc(1, DATA_ENTRY_MSB, 20, time=0),
        cc(0, DATA_ENTRY_LSB, 0, time=5),
        cc(1, DATA_ENTRY_LSB, 0, time=0),
        Message('note_on', channel=0, note=60, velocity=64, time=5),
        MetaMessage('end_of_track', time=0),
    ])])
    note = next(m for m in reloaded.tracks[0] if m.type == 'note_on')
    note_tick = absolute_ticks(reloaded.tracks[0])[
        list(reloaded.tracks[0]).index(note)]
    original_note = next(m for m in original.tracks[0]
                         if m.type == 'note_on')
    assert note_tick == absolute_ticks(original.tracks[0])[
        list(original.tracks[0]).index(original_note)]
    assert sum(m.time for m in reloaded.tracks[0]) == \
        sum(m.time for m in original.tracks[0])
    again = extract_parameter_events(reloaded)
    assert {e.channel: e.value for e in again} == {0: 10 << 7,
                                                   1: 20 << 7}


def test_increment_decrement_round_trip():
    track = MidiTrack([
        cc(0, NRPN_MSB, 0, time=0),
        cc(0, NRPN_LSB, 8, time=0),
        cc(0, DATA_INCREMENT, 3, time=12),
        cc(0, DATA_DECREMENT, 1, time=8),
        MetaMessage('end_of_track', time=0),
    ])
    mid = MidiFile(tracks=[track])
    events = extract_parameter_events(mid)
    assert [(e.operation, e.value, e.tick) for e in events] == [
        ('increment', 3, 12),
        ('decrement', 1, 20),
    ]

    write_parameter_events(mid, events)
    reloaded = save_and_reload(mid)
    again = extract_parameter_events(reloaded)
    assert [(e.operation, e.value, e.tick) for e in again] == [
        ('increment', 3, 12),
        ('decrement', 1, 20),
    ]
    # The shared selector pair is emitted once; both data messages reuse
    # it, exactly as in the source file.
    controls = [m.control for m in reloaded.tracks[0]
                if m.type == 'control_change']
    assert controls == [NRPN_MSB, NRPN_LSB, DATA_INCREMENT,
                        DATA_DECREMENT]


def test_rpn_null_value_is_illegal():
    # 14-bit value with both bytes 127 means number 16383, *not* null;
    # only the exact (127, 127) RPN selector is null.
    event = ParameterEvent(track=0, channel=0, tick=0, kind='rpn',
                           operation='set', number=RPN_NULL,
                           value=1, width=7)
    assert event.number == RPN_NULL
    messages = event_messages(event)
    assert [(m.control, m.value) for m in messages] == [
        (RPN_MSB, 127), (RPN_LSB, 127), (DATA_ENTRY_MSB, 1)]


def test_consecutive_data_entry_msb_yields_two_seven_bit_events():
    track = MidiTrack([
        cc(0, NRPN_MSB, 0, time=0),
        cc(0, NRPN_LSB, 5, time=0),       # NRPN 5
        cc(0, DATA_ENTRY_MSB, 10, time=4),
        cc(0, DATA_ENTRY_MSB, 20, time=6),
        MetaMessage('end_of_track', time=0),
    ])
    mid = MidiFile(tracks=[track])
    events = extract_parameter_events(mid)

    assert len(events) == 2
    assert [(e.value, e.width, e.tick) for e in events] == [
        (10, 7, 4),
        (20, 7, 10),
    ]
    # Both belong to the same selector group.
    assert events[0].group == events[1].group

    write_parameter_events(mid, events)
    reloaded = save_and_reload(mid)
    again = extract_parameter_events(reloaded)
    assert [(e.value, e.tick) for e in again] == [(10, 4), (20, 10)]


def test_ordinary_cc_does_not_break_fourteen_bit_pairing():
    track = MidiTrack([
        cc(0, RPN_MSB, 0, time=0),
        cc(0, RPN_LSB, 0, time=0),
        cc(0, DATA_ENTRY_MSB, 0x10, time=0),
        cc(0, 7, 64, time=3),              # unrelated volume CC in between
        Message('note_on', channel=0, note=60, velocity=64, time=2),
        cc(0, DATA_ENTRY_LSB, 0x20, time=4),
        MetaMessage('end_of_track', time=0),
    ])
    mid = MidiFile(tracks=[track])
    events = extract_parameter_events(mid)

    assert len(events) == 1
    assert events[0].width == 14
    assert events[0].value == (0x10 << 7) | 0x20
    assert events[0].tick == 9


def test_stale_sources_are_rejected():
    mid = build_fixture()
    events = extract_parameter_events(mid)
    target = events[0]

    # Fabricate an event whose sources point at the wrong messages
    # (an ordinary volume CC instead of the selector pair).
    bogus_sources = [0, 1] + target._sources[2:]
    bogus = ParameterEvent(
        track=target.track, channel=target.channel, tick=target.tick,
        kind=target.kind, operation=target.operation,
        number=target.number, value=target.value, width=target.width,
        _sources=bogus_sources)
    with pytest.raises(ValueError):
        write_parameter_events(mid, [bogus])

    # A source list shorter than the selector pair is rejected outright.
    truncated = ParameterEvent(
        track=target.track, channel=target.channel, tick=target.tick,
        kind=target.kind, operation=target.operation,
        number=target.number, value=target.value, width=target.width,
        _sources=target._sources[:1])
    with pytest.raises(ValueError):
        write_parameter_events(mid, [truncated])


def test_public_api_exported_at_top_level():
    import mido
    from mido.midifiles import param_events

    for name in ('ParameterEvent', 'extract_parameter_events',
                 'write_parameter_events', 'PARAMETER_CONTROLS',
                 'RPN_NULL'):
        assert hasattr(mido, name)
    assert hasattr(param_events, 'event_messages')

    assert mido.PARAMETER_CONTROLS == frozenset(
        (RPN_MSB, RPN_LSB, NRPN_MSB, NRPN_LSB, DATA_ENTRY_MSB,
         DATA_ENTRY_LSB, DATA_INCREMENT, DATA_DECREMENT))
