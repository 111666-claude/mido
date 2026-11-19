# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

import io

import pytest

from mido import (
    Message,
    MetaMessage,
    MidiFile,
    MidiTrack,
    split_instrument_lanes,
    split_track_lanes,
)
from mido.midifiles.lanes import DRUM_CHANNEL, build_tempo_map, tick_at

TICKS_PER_BEAT = 480


def make_two_channel_tempo_fixture():
    """Two tracks, two channels, one tempo change and every edge case.

    Track 0 (channel 0): piano (program 0) switches to strings
    (program 48); note 60 is started before the switch and closed after
    it; note 64 has no note_off and must be synthesized.

    Track 1 (channel 1): flute (program 73) switches to violin
    (program 40) with another event at the same tick, plus a drum
    channel (9) that must never be split.
    """
    mid = MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)
    mid.tracks.append(MidiTrack([
        MetaMessage('track_name', name='piano+strings', time=0),
        MetaMessage('set_tempo', tempo=500000, time=0),              # 0
        Message('control_change', channel=0, control=64, value=127,
                time=0),                                              # 0 sustain
        Message('note_on', channel=0, note=60, velocity=80, time=0),  # 0
        MetaMessage('set_tempo', tempo=250000, time=TICKS_PER_BEAT),  # 480
        Message('program_change', channel=0, program=48, time=0),     # 480
        Message('note_on', channel=0, note=62, velocity=80,
                time=TICKS_PER_BEAT),                                 # 960
        Message('note_off', channel=0, note=60, velocity=0, time=0),  # 960
        Message('control_change', channel=0, control=7, value=100,
                time=TICKS_PER_BEAT),                                 # 1440
        Message('note_off', channel=0, note=62, velocity=0, time=0),  # 1440
        Message('note_on', channel=0, note=64, velocity=70,
                time=TICKS_PER_BEAT),                                 # 1920
        MetaMessage('end_of_track', time=0),
    ]))
    mid.tracks.append(MidiTrack([
        Message('program_change', channel=1, program=73, time=0),     # 0
        Message('pitchwheel', channel=1, pitch=512, time=0),          # 0
        Message('note_on', channel=1, note=70, velocity=90,
                time=TICKS_PER_BEAT),                                 # 480
        Message('program_change', channel=9, program=20, time=0),     # 480
        Message('note_on', channel=9, note=36, velocity=110, time=0),  # 480
        Message('note_off', channel=9, note=36, velocity=0,
                time=TICKS_PER_BEAT),                                 # 960
        Message('note_off', channel=1, note=70, velocity=0, time=0),  # 960
        Message('program_change', channel=1, program=40,
                time=TICKS_PER_BEAT),                                 # 1440
        Message('note_on', channel=1, note=72, velocity=90, time=0),  # 1440
        Message('note_off', channel=1, note=72, velocity=0,
                time=TICKS_PER_BEAT),                                 # 1920
        MetaMessage('end_of_track', time=0),
    ]))
    return mid


def lane(lanes, channel, program):
    return next(lane for lane in lanes
                if lane.channel == channel and lane.program == program)


def test_lane_identity_and_order():
    lanes = make_two_channel_tempo_fixture().instrument_lanes
    keys = [(lane.channel, lane.program) for lane in lanes]
    # First appearance tick, then channel; drum program is None.
    assert keys == [(0, 0), (1, 73), (0, 48), (9, None), (1, 40)]


def test_notes_are_owned_by_their_attack_lane():
    lanes = make_two_channel_tempo_fixture().instrument_lanes

    piano = lane(lanes, 0, 0)
    assert [(e.tick, e.message.type, getattr(e.message, 'note', None))
            for e in piano.events] == [
        (0, 'control_change', None),
        (0, 'note_on', 60),
        (960, 'note_off', 60),  # closed on the piano lane after switch
    ]
    assert all(e.track == 0 for e in piano.events)

    strings = lane(lanes, 0, 48)
    assert [(e.tick, e.message.type,
             getattr(e.message, 'note', getattr(e.message, 'control', None)))
            for e in strings.events] == [
        (480, 'program_change', None),
        (960, 'note_on', 62),
        (1440, 'control_change', 7),
        (1440, 'note_off', 62),
        (1920, 'note_on', 64),
        (1920, 'note_off', 64),  # synthesized closing event
    ]


def test_missing_note_off_is_synthesized_at_track_end():
    lanes = make_two_channel_tempo_fixture().instrument_lanes
    strings = lane(lanes, 0, 48)
    synthesized = [e for e in strings.events if e.synthesized]
    assert len(synthesized) == 1
    event = synthesized[0]
    assert event.message.type == 'note_off'
    assert event.message.note == 64
    assert event.tick == 1920  # end of the source track


def test_controllers_pitchwheel_and_sustain_follow_active_program():
    lanes = make_two_channel_tempo_fixture().instrument_lanes

    # Sustain (CC 64) at tick 0 lands on piano, volume (CC 7) at 1440
    # lands on strings.
    piano, strings = lane(lanes, 0, 0), lane(lanes, 0, 48)
    sustain = [e for e in piano.events if e.message.type == 'control_change']
    volume = [e for e in strings.events if e.message.type == 'control_change']
    assert [(e.tick, e.message.control) for e in sustain] == [(0, 64)]
    assert [(e.tick, e.message.control) for e in volume] == [(1440, 7)]

    # Pitchwheel goes to the flute lane active at that tick.
    flute = lane(lanes, 1, 73)
    assert [(e.tick, e.message.type) for e in flute.events] == [
        (0, 'program_change'),
        (0, 'pitchwheel'),
        (480, 'note_on'),
        (960, 'note_off'),
    ]


def test_drum_channel_is_a_single_lane():
    lanes = make_two_channel_tempo_fixture().instrument_lanes
    drum = next(lane for lane in lanes if lane.is_drum)
    assert drum.channel == DRUM_CHANNEL
    assert drum.program is None
    assert all(e.message.channel == DRUM_CHANNEL for e in drum.events)
    # The program change on channel 9 is kept but does not split it.
    assert any(e.message.type == 'program_change' for e in drum.events)
    assert [(e.tick, e.message.type, getattr(e.message, 'note', None))
            for e in drum.events] == [
        (480, 'program_change', None),
        (480, 'note_on', 36),
        (960, 'note_off', 36),
    ]


def test_same_tick_switch_routes_later_event_to_new_lane():
    track = MidiTrack([
        Message('program_change', channel=1, program=40, time=100),
        Message('note_on', channel=1, note=72, velocity=90, time=0),
        Message('note_off', channel=1, note=72, velocity=0, time=50),
        MetaMessage('end_of_track', time=0),
    ])
    violin = lane(split_track_lanes(track), 1, 40)
    # Both same-tick events end up on the new lane, in stored order.
    assert [(e.tick, e.message.type) for e in violin.events] == [
        (100, 'program_change'),
        (100, 'note_on'),
        (150, 'note_off'),
    ]


def test_seconds_use_the_tempo_map():
    mid = make_two_channel_tempo_fixture()
    lanes = mid.instrument_lanes
    piano = lane(lanes, 0, 0)
    flute = lane(lanes, 1, 73)

    # tick 0 -> 0 s; tick 480 at 500000 us -> 0.5 s;
    # tick 960 adds 480 ticks at 250000 us -> 0.75 s.
    assert piano.events[0].second == 0.0
    closing = next(e for e in piano.events if e.message.type == 'note_off')
    assert closing.second == pytest.approx(0.75)
    assert flute.events[2].second == pytest.approx(0.5)


def test_tick_at_helper():
    tempo_map = build_tempo_map(make_two_channel_tempo_fixture().tracks)
    assert tick_at(0, tempo_map, TICKS_PER_BEAT) == 0.0
    assert tick_at(480, tempo_map, TICKS_PER_BEAT) == pytest.approx(0.5)
    assert tick_at(960, tempo_map, TICKS_PER_BEAT) == pytest.approx(0.75)


def test_source_track_is_preserved():
    mid = MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)
    mid.tracks.append(MidiTrack([
        Message('note_on', channel=0, note=60, time=0),
        Message('note_off', channel=0, note=60, time=100),
        MetaMessage('end_of_track', time=0),
    ]))
    mid.tracks.append(MidiTrack([
        Message('note_on', channel=0, note=62, time=50),
        Message('note_off', channel=0, note=62, time=100),
        MetaMessage('end_of_track', time=0),
    ]))
    only_lane = split_instrument_lanes(mid)[0]
    assert only_lane.source_tracks == [0, 1]
    assert [(e.tick, e.track, e.message.note) for e in only_lane.events] == [
        (0, 0, 60),
        (50, 1, 62),
        (100, 0, 60),
        (150, 1, 62),
    ]


def test_round_trip_through_midi_file_preserves_lanes():
    mid = make_two_channel_tempo_fixture()
    buffer = io.BytesIO()
    mid.save(file=buffer)
    buffer.seek(0)
    reloaded = MidiFile(file=buffer)

    before = [(item.channel, item.program,
               [(e.tick, e.message.type) for e in item.events])
              for item in mid.instrument_lanes]
    after = [(item.channel, item.program,
              [(e.tick, e.message.type) for e in item.events])
             for item in reloaded.instrument_lanes]
    assert before == after


def test_to_midi_track_uses_delta_times():
    lanes = make_two_channel_tempo_fixture().instrument_lanes
    piano = lane(lanes, 0, 0)
    track = piano.to_midi_track()
    assert isinstance(track, MidiTrack)
    assert track[-1].type == 'end_of_track'
    # Delta times reproduce the original absolute ticks.
    assert [msg.time for msg in track[:-1]] == [0, 0, 960]


def test_type_2_rejected_for_file_but_supported_per_track():
    mid = MidiFile(type=2, ticks_per_beat=TICKS_PER_BEAT, tracks=[
        MidiTrack([
            Message('note_on', channel=0, note=60, time=0),
            Message('note_off', channel=0, note=60, time=10),
            MetaMessage('end_of_track', time=0),
        ]),
    ])
    with pytest.raises(TypeError):
        split_instrument_lanes(mid)
    assert len(split_track_lanes(mid.tracks[0])) == 1


def test_illegal_program_raises():
    track = MidiTrack([
        Message('program_change', channel=0, program=999,
                skip_checks=True, time=0),
    ])
    with pytest.raises(ValueError, match='0..127'):
        split_track_lanes(track)


def test_velocity_zero_note_on_and_fifo_pairing():
    track = MidiTrack([
        Message('note_on', channel=0, note=60, velocity=80, time=0),
        Message('note_on', channel=0, note=60, velocity=80, time=0),
        Message('program_change', channel=0, program=48, time=10),
        Message('note_on', channel=0, note=60, velocity=0, time=0),
        Message('note_off', channel=0, note=60, time=0),
        MetaMessage('end_of_track', time=0),
    ])
    lanes = split_track_lanes(track)
    piano = lane(lanes, 0, 0)
    strings = lane(lanes, 0, 48)
    assert [(e.tick, e.message.type) for e in piano.events] == [
        (0, 'note_on'), (0, 'note_on'),
        (10, 'note_on'), (10, 'note_off'),
    ]
    assert [e.message.type for e in strings.events] == ['program_change']


def test_stray_note_off_uses_current_lane():
    track = MidiTrack([
        Message('program_change', channel=0, program=25, time=0),
        Message('note_off', channel=0, note=70, time=20),
        MetaMessage('end_of_track', time=0),
    ])
    lanes = split_track_lanes(track)
    assert len(lanes) == 1
    assert lanes[0].program == 25
    assert lanes[0].events[-1].message.type == 'note_off'


def test_meta_and_sysex_messages_are_excluded():
    track = MidiTrack([
        Message('polytouch', channel=0, note=60, value=10, time=0),
        Message('aftertouch', channel=0, value=5, time=0),
        Message('sysex', data=[1, 2, 3], time=0),
        MetaMessage('track_name', name='x', time=0),
        MetaMessage('end_of_track', time=0),
    ])
    events = split_track_lanes(track)[0].events
    assert [e.message.type for e in events] == ['polytouch', 'aftertouch']


def test_instrument_lanes_cache_invalidated_on_add_track():
    mid = MidiFile(type=1, tracks=[MidiTrack([
        Message('note_on', channel=0, note=60, time=0),
        MetaMessage('end_of_track', time=0),
    ])])
    assert mid.instrument_lanes is mid.instrument_lanes
    mid.add_track('extra')
    assert mid._instrument_lanes is None
