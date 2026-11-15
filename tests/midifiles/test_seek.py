# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

import io

from pytest import raises

from mido.messages import Message
from mido.midifiles.meta import MetaMessage
from mido.midifiles.midifiles import MidiFile, MidiTrack
from mido.midifiles.seek import ChannelState, PlaybackPosition

# With ticks_per_beat=1000 and tempo=1000000 (one second per quarter
# note) one tick is exactly one millisecond, which keeps the expected
# times in these tests easy to read.
TICKS_PER_BEAT = 1000
TEMPO = 1000000


def make_file(tracks, type=1):
    return MidiFile(type=type, ticks_per_beat=TICKS_PER_BEAT, tracks=tracks)


def make_simple_file():
    """One track, constant tempo:

    t=0.0  note_on  60 (vel 100)
    t=1.0  note_off 60
    t=1.5  note_on  64 (vel 90)
    t=2.5  note_off 64
    """
    track = MidiTrack([
        MetaMessage('set_tempo', tempo=TEMPO, time=0),
        Message('note_on', channel=0, note=60, velocity=100, time=0),
        Message('note_off', channel=0, note=60, velocity=0, time=1000),
        Message('note_on', channel=0, note=64, velocity=90, time=500),
        Message('note_off', channel=0, note=64, velocity=0, time=1000),
    ])
    return make_file([track])


def channel_events(position):
    """Non-meta events of a position as (type, note, time) tuples.

    Times are rounded to nanoseconds to avoid float noise in the
    comparisons.
    """
    return [(msg.type, msg.note, round(msg.time, 9))
            for msg in position.events if not msg.is_meta]


def test_seek_returns_playback_position():
    position = make_simple_file().seek(0.5)
    assert isinstance(position, PlaybackPosition)
    assert isinstance(position.state, ChannelState)
    assert position.time == 0.5


def test_seek_between_events():
    position = make_simple_file().seek(1.25)
    # Note 60 is off, note 64 has not started yet.
    assert position.state.notes == {}
    assert channel_events(position) == [
        ('note_on', 64, 0.25),
        ('note_off', 64, 1.25),
    ]


def test_seek_event_times_are_relative_and_ordered():
    position = make_simple_file().seek(0.25)
    times = [msg.time for msg in position.events]
    assert times == sorted(times)
    assert times[0] >= 0
    # Absolute times of the whole song shifted by the seek position.
    assert channel_events(position) == [
        ('note_off', 60, 0.75),
        ('note_on', 64, 1.25),
        ('note_off', 64, 2.25),
    ]


def test_seek_exactly_on_event():
    # A message exactly at the position is returned in the events
    # with time 0 and is not yet reflected in the state.
    position = make_simple_file().seek(1.0)
    assert position.state.notes == {0: {60: 100}}
    first = position.events[0]
    assert first.type == 'note_off' and first.note == 60
    assert first.time == 0.0


def test_seek_at_start():
    position = make_simple_file().seek(0)
    assert position.state.notes == {}
    assert position.events[0].time == 0.0
    assert channel_events(position)[0] == ('note_on', 60, 0.0)


def test_seek_past_end():
    position = make_simple_file().seek(100.0)
    assert position.events == []
    assert position.state.notes == {}


def test_seek_past_end_keeps_sounding_note():
    track = MidiTrack([
        MetaMessage('set_tempo', tempo=TEMPO, time=0),
        Message('note_on', channel=0, note=60, velocity=100, time=0),
        # No note_off: the note is still sounding at the end.
    ])
    position = make_file([track]).seek(100.0)
    assert position.events == []
    assert position.state.active_notes() == [(0, 60, 100)]


def test_seek_empty_file():
    position = make_file([]).seek(0)
    assert position.state.notes == {}
    assert position.state.controllers == {}
    # Only the end of track message remains.
    assert [msg.type for msg in position.events] == ['end_of_track']
    assert make_file([]).seek(1.0).events == []


def test_seek_empty_track():
    track = MidiTrack([MetaMessage('end_of_track', time=0)])
    position = make_file([track]).seek(0.5)
    assert position.events == []
    assert position.state == ChannelState()


def test_seek_negative_seconds():
    with raises(ValueError):
        make_simple_file().seek(-0.1)


def test_seek_type_2_file():
    mid = make_file([MidiTrack(), MidiTrack()], type=2)
    with raises(TypeError):
        mid.seek(0)


def test_tempo_change_on_separate_track():
    # Type 1 file: tempo map on track 0, notes on track 1.
    # Tempo is 1 s/beat until tick 1000 (t=1.0), then 0.5 s/beat.
    tempo_track = MidiTrack([
        MetaMessage('set_tempo', tempo=1000000, time=0),
        MetaMessage('set_tempo', tempo=500000, time=1000),
    ])
    note_track = MidiTrack([
        Message('note_on', channel=0, note=60, velocity=100, time=0),
        # tick 1500 -> t = 1.0 + 500 * 0.0005 = 1.25
        Message('note_on', channel=0, note=62, velocity=90, time=1500),
        # tick 2000 -> t = 1.5
        Message('note_off', channel=0, note=62, velocity=0, time=500),
        Message('note_off', channel=0, note=60, velocity=0, time=0),
    ])
    mid = make_file([tempo_track, note_track])

    position = mid.seek(1.3)
    # Both notes are sounding; the tempo change sped the song up.
    assert position.state.active_notes() == [(0, 60, 100), (0, 62, 90)]
    assert channel_events(position) == [
        ('note_off', 62, 0.2),
        ('note_off', 60, 0.2),
    ]


def test_same_tick_order_is_stable():
    # Messages on the same absolute tick keep their merge order
    # (track order, then order within the track).
    track_a = MidiTrack([
        MetaMessage('set_tempo', tempo=TEMPO, time=0),
        Message('note_on', channel=0, note=60, velocity=100, time=500),
    ])
    track_b = MidiTrack([
        Message('note_on', channel=1, note=62, velocity=90, time=500),
        Message('note_on', channel=2, note=64, velocity=80, time=0),
    ])
    mid = make_file([track_a, track_b])

    position = mid.seek(0)
    notes = [(msg.channel, msg.note) for msg in position.events
             if msg.type == 'note_on']
    assert notes == [(0, 60), (1, 62), (2, 64)]
    # All three land on the same tick, hence the same time.
    assert [msg.time for msg in position.events
            if msg.type == 'note_on'] == [0.5, 0.5, 0.5]


def test_overlapping_notes():
    track = MidiTrack([
        MetaMessage('set_tempo', tempo=TEMPO, time=0),
        Message('note_on', channel=0, note=60, velocity=100, time=0),
        # Same note struck again before the first one is off.
        Message('note_on', channel=0, note=60, velocity=80, time=500),
        Message('note_off', channel=0, note=60, velocity=0, time=500),
        # note_on with velocity 0 acts as note off (nothing left).
        Message('note_on', channel=0, note=60, velocity=0, time=500),
    ])
    mid = make_file([track])

    # The re-strike replaces the velocity of the sounding note.
    assert mid.seek(0.75).state.active_notes() == [(0, 60, 80)]
    assert mid.seek(1.25).state.active_notes() == []
    assert mid.seek(1.75).state.active_notes() == []


def test_notes_are_tracked_per_channel():
    track = MidiTrack([
        MetaMessage('set_tempo', tempo=TEMPO, time=0),
        Message('note_on', channel=0, note=60, velocity=100, time=0),
        Message('note_on', channel=1, note=60, velocity=70, time=0),
        Message('note_off', channel=0, note=60, velocity=0, time=500),
    ])
    position = make_file([track]).seek(0.75)
    assert position.state.active_notes() == [(1, 60, 70)]
    assert position.state.active_notes(channel=0) == []


def test_sustain_and_controllers():
    track = MidiTrack([
        MetaMessage('set_tempo', tempo=TEMPO, time=0),
        Message('control_change', channel=0, control=64, value=127, time=0),
        Message('control_change', channel=0, control=7, value=100, time=100),
        Message('control_change', channel=1, control=64, value=20, time=100),
        Message('control_change', channel=0, control=64, value=0, time=1800),
    ])
    mid = make_file([track])

    position = mid.seek(0.5)
    assert position.state.is_sustained(0)
    # Sustain value 20 is below the halfway point: pedal is up.
    assert not position.state.is_sustained(1)
    assert position.state.control(0, 7) == 100
    assert position.state.control(0, 64) == 127
    # Controller never set: default is returned.
    assert position.state.control(0, 11) == 0
    assert position.state.control(5, 7) == 0

    # Pedal released at t=2.0; volume is still at its latest value.
    position = mid.seek(3.0)
    assert not position.state.is_sustained(0)
    assert position.state.control(0, 64) == 0
    assert position.state.control(0, 7) == 100


def saved_bytes(mid):
    outfile = io.BytesIO()
    mid.save(file=outfile)
    return outfile.getvalue()


def test_repeated_seek_does_not_modify_file():
    mid = make_simple_file()
    before = saved_bytes(mid)
    track_times_before = [[msg.time for msg in track] for track in mid.tracks]

    for seconds in [0, 0.5, 1.0, 1.25, 2.5, 100.0, 0.5]:
        mid.seek(seconds)

    assert saved_bytes(mid) == before
    assert [[msg.time for msg in track] for track in mid.tracks] == \
        track_times_before


def test_repeated_seek_is_deterministic():
    mid = make_simple_file()
    first = mid.seek(0.9)
    second = mid.seek(0.9)
    assert [(msg.type, msg.time) for msg in first.events] == \
        [(msg.type, msg.time) for msg in second.events]
    assert first.state == second.state


def test_returned_events_are_copies():
    mid = make_simple_file()
    position = mid.seek(0)
    for msg in position.events:
        msg.time = 999

    # The file is unaffected by modifying the returned messages.
    again = mid.seek(0)
    assert [msg.time for msg in again.events] != []
    assert max(msg.time for msg in again.events) < 999


def test_position_is_iterable():
    position = make_simple_file().seek(0)
    assert list(position) == position.events
    assert len(position) == len(position.events)


def test_channel_state_copy_is_independent():
    mid = make_simple_file()
    state = mid.seek(0.5).state
    copied = state.copy()
    assert copied == state
    copied.notes[0][60] = 1
    assert state.notes[0][60] == 100
