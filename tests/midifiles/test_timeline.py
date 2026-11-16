# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

import copy
import io

import pytest

from mido import (
    END_REASON_END_OF_TRACK,
    END_REASON_NOTE_OFF,
    Message,
    MetaMessage,
    MidiFile,
    MidiTrack,
    NoteTimeline,
    SoundingNote,
    get_note_timeline,
    merge_tracks,
)


def make_file(tracks, ticks_per_beat=480, type=1):
    return MidiFile(type=type, ticks_per_beat=ticks_per_beat,
                    tracks=[MidiTrack(messages) for messages in tracks])


def pedal(value, time=0):
    return Message('control_change', control=64, value=value, time=time)


# ---------------------------------------------------------------------
# Basic extraction and time conversion
# ---------------------------------------------------------------------

def test_basic_note_seconds():
    # 480 ticks at 480 tpb and the default tempo = one beat = 0.5 s.
    mid = make_file([[
        Message('note_on', note=60, velocity=100, time=0),
        Message('note_off', note=60, velocity=64, time=480),
        MetaMessage('end_of_track', time=0),
    ]])

    timeline = mid.note_timeline()

    assert isinstance(timeline, NoteTimeline)
    assert len(timeline) == 1
    note = timeline[0]
    assert isinstance(note, SoundingNote)
    assert note.track == 0
    assert note.channel == 0
    assert note.note == 60
    assert note.velocity == 100
    assert note.start_time == pytest.approx(0.0)
    assert note.end_time == pytest.approx(0.5)
    assert note.end_reason == END_REASON_NOTE_OFF
    assert timeline.unmatched_note_offs == []


def test_seconds_follow_tempo_changes():
    # First beat at 120 BPM (0.5 s), second beat at 240 BPM (0.25 s).
    mid = make_file([[
        Message('note_on', note=60, velocity=100, time=0),
        MetaMessage('set_tempo', tempo=250000, time=480),
        Message('note_off', note=60, velocity=64, time=480),
        MetaMessage('end_of_track', time=0),
    ]])

    note = mid.note_timeline()[0]

    assert note.start_time == pytest.approx(0.0)
    assert note.end_time == pytest.approx(0.75)


def test_tempo_change_at_time_zero():
    mid = make_file([[
        MetaMessage('set_tempo', tempo=1_000_000, time=0),
        Message('note_on', note=60, velocity=100, time=0),
        Message('note_off', note=60, velocity=64, time=480),
        MetaMessage('end_of_track', time=0),
    ]])

    note = mid.note_timeline()[0]

    # One beat at 1 000 000 us/beat = 1.0 s.
    assert note.end_time == pytest.approx(1.0)


def test_several_tempo_changes():
    mid = make_file([[
        Message('note_on', note=60, velocity=100, time=0),       # tick 0
        MetaMessage('set_tempo', tempo=250000, time=240),        # tick 240
        MetaMessage('set_tempo', tempo=125000, time=240),        # tick 480
        Message('note_off', note=60, velocity=64, time=240),     # tick 720
        MetaMessage('end_of_track', time=0),
    ]])

    note = mid.note_timeline()[0]

    # 240 ticks @500000 + 240 @250000 + 240 @125000
    # = 0.25 + 0.125 + 0.0625
    assert note.end_time == pytest.approx(0.4375)


def test_multi_track_preserves_track_indices():
    mid = make_file([
        [
            Message('note_on', channel=0, note=60, velocity=100, time=0),
            Message('note_off', channel=0, note=60, velocity=64, time=480),
            MetaMessage('end_of_track', time=0),
        ],
        [
            Message('note_on', channel=1, note=72, velocity=90, time=0),
            Message('note_off', channel=1, note=72, velocity=64, time=960),
            MetaMessage('end_of_track', time=0),
        ],
    ])

    timeline = mid.note_timeline()

    assert len(timeline) == 2
    first, second = timeline
    # Both start at tick 0; the track index breaks the tie.
    assert (first.track, first.note, first.channel) == (0, 60, 0)
    assert (second.track, second.note, second.channel) == (1, 72, 1)
    assert first.end_time == pytest.approx(0.5)
    assert second.end_time == pytest.approx(1.0)


def test_events_at_same_tick_keep_track_order():
    mid = make_file([
        [
            Message('note_on', note=60, velocity=100, time=100),
            Message('note_off', note=60, velocity=64, time=380),
            MetaMessage('end_of_track', time=0),
        ],
        [
            Message('note_on', note=62, velocity=100, time=100),
            Message('note_off', note=62, velocity=64, time=380),
            MetaMessage('end_of_track', time=0),
        ],
    ])

    timeline = mid.note_timeline()

    assert [n.track for n in timeline] == [0, 1]
    assert [n.note for n in timeline] == [60, 62]


def test_type_0_single_track():
    mid = make_file([[
        Message('note_on', note=60, velocity=100, time=0),
        Message('note_off', note=60, velocity=64, time=480),
        MetaMessage('end_of_track', time=0),
    ]], type=0)

    assert len(mid.note_timeline()) == 1


# ---------------------------------------------------------------------
# Note-on / note-off pairing rules
# ---------------------------------------------------------------------

def test_zero_velocity_note_on_acts_as_note_off():
    mid = make_file([[
        Message('note_on', note=60, velocity=100, time=0),
        Message('note_on', note=60, velocity=0, time=480),
        MetaMessage('end_of_track', time=0),
    ]])

    note = mid.note_timeline()[0]

    assert note.velocity == 100
    assert note.end_time == pytest.approx(0.5)
    assert note.end_reason == END_REASON_NOTE_OFF


def test_overlapping_same_note_uses_fifo_pairing():
    mid = make_file([[
        Message('note_on', note=60, velocity=100, time=0),    # tick 0, A
        Message('note_on', note=60, velocity=80, time=100),    # tick 100, B
        Message('note_off', note=60, velocity=64, time=100),   # tick 200 -> A
        Message('note_off', note=60, velocity=64, time=100),   # tick 300 -> B
        MetaMessage('end_of_track', time=0),
    ]])

    timeline = mid.note_timeline()
    a, b = timeline

    assert a.velocity == 100
    assert a.start_time == pytest.approx(0.0)
    assert a.end_time == pytest.approx(200 * 0.5 / 480)
    assert b.velocity == 80
    assert b.start_time == pytest.approx(100 * 0.5 / 480)
    assert b.end_time == pytest.approx(300 * 0.5 / 480)


def test_retrigger_while_sustained():
    # A released into the pedal, then a new note on the same pitch
    # starts and ends normally while A is still sounding.
    mid = make_file([[
        Message('note_on', note=60, velocity=100, time=0),   # tick 0 A
        pedal(127, time=100),                                # tick 100
        Message('note_off', note=60, velocity=64, time=100),  # 200: A sustained
        Message('note_on', note=60, velocity=90, time=100),  # tick 300 B
        Message('note_off', note=60, velocity=64, time=100),  # 400: B sustained
        pedal(0, time=100),                                  # tick 500 up
        MetaMessage('end_of_track', time=0),
    ]])

    timeline = mid.note_timeline()

    assert len(timeline) == 2
    by_velocity = {n.velocity: n for n in timeline}
    assert by_velocity[100].start_time == pytest.approx(0.0)
    assert by_velocity[90].start_time == pytest.approx(300 * 0.5 / 480)
    for note in timeline:
        assert note.end_time == pytest.approx(500 * 0.5 / 480)
        assert note.end_reason == END_REASON_NOTE_OFF


def test_pairing_is_per_channel_and_pitch_across_tracks():
    # Same channel and pitch in two tracks share one pairing space.
    mid = make_file([
        [
            Message('note_on', channel=0, note=60, velocity=100, time=0),
            MetaMessage('end_of_track', time=480),
        ],
        [
            Message('note_off', channel=0, note=60, velocity=64, time=240),
            MetaMessage('end_of_track', time=240),
        ],
    ])

    timeline = mid.note_timeline()

    assert len(timeline) == 1
    note = timeline[0]
    assert note.track == 0
    assert note.end_time == pytest.approx(240 * 0.5 / 480)


def test_channels_are_independent():
    mid = make_file([[
        Message('note_on', channel=0, note=60, velocity=100, time=0),
        Message('note_on', channel=1, note=60, velocity=90, time=0),
        Message('note_off', channel=0, note=60, velocity=64, time=240),
        Message('note_off', channel=1, note=60, velocity=64, time=480),
        MetaMessage('end_of_track', time=0),
    ]])

    timeline = mid.note_timeline()

    assert {n.channel: n.end_time for n in timeline} == {
        0: pytest.approx(240 * 0.5 / 480),
        1: pytest.approx(720 * 0.5 / 480),
    }


# ---------------------------------------------------------------------
# Sustain pedal (CC64)
# ---------------------------------------------------------------------

def test_sustain_pedal_holds_released_note():
    mid = make_file([[
        Message('note_on', note=60, velocity=100, time=0),     # tick 0
        pedal(127, time=240),                                  # tick 240 down
        Message('note_off', note=60, velocity=64, time=240),   # tick 480 released
        pedal(0, time=480),                                    # tick 960 up
        MetaMessage('end_of_track', time=0),
    ]])

    note = mid.note_timeline()[0]

    assert note.end_time == pytest.approx(1.0)
    assert note.end_reason == END_REASON_NOTE_OFF


def test_pedal_release_does_not_affect_held_notes():
    mid = make_file([[
        Message('note_on', note=60, velocity=100, time=0),     # tick 0
        pedal(127, time=100),                                  # tick 100
        pedal(0, time=100),                                    # tick 200
        Message('note_off', note=60, velocity=64, time=100),   # tick 300
        MetaMessage('end_of_track', time=0),
    ]])

    note = mid.note_timeline()[0]

    assert note.end_time == pytest.approx(300 * 0.5 / 480)
    assert note.end_reason == END_REASON_NOTE_OFF


def test_pedal_holds_several_released_notes():
    mid = make_file([[
        Message('note_on', note=60, velocity=100, time=0),
        Message('note_on', note=64, velocity=100, time=0),
        pedal(127, time=100),
        Message('note_off', note=60, velocity=64, time=100),
        Message('note_off', note=64, velocity=64, time=100),
        pedal(0, time=100),
        MetaMessage('end_of_track', time=0),
    ]])

    timeline = mid.note_timeline()

    assert len(timeline) == 2
    for note in timeline:
        assert note.end_time == pytest.approx(400 * 0.5 / 480)
        assert note.end_reason == END_REASON_NOTE_OFF


def test_pedal_threshold_value_64():
    # 63 is pedal up, 64 is pedal down.
    mid_up = make_file([[
        Message('note_on', note=60, velocity=100, time=0),
        pedal(63, time=100),
        Message('note_off', note=60, velocity=64, time=100),
        MetaMessage('end_of_track', time=100),
    ]])
    assert mid_up.note_timeline()[0].end_time == \
        pytest.approx(200 * 0.5 / 480)

    mid_down = make_file([[
        Message('note_on', note=60, velocity=100, time=0),
        pedal(64, time=100),
        Message('note_off', note=60, velocity=64, time=100),
        pedal(0, time=100),
        MetaMessage('end_of_track', time=0),
    ]])
    assert mid_down.note_timeline()[0].end_time == \
        pytest.approx(300 * 0.5 / 480)


def test_repeated_pedal_down_is_idempotent():
    mid = make_file([[
        Message('note_on', note=60, velocity=100, time=0),
        pedal(127, time=100),
        pedal(100, time=100),
        Message('note_off', note=60, velocity=64, time=100),
        pedal(0, time=100),
        MetaMessage('end_of_track', time=0),
    ]])

    note = mid.note_timeline()[0]

    assert note.end_time == pytest.approx(400 * 0.5 / 480)


def test_pedal_crosses_tempo_change():
    mid = make_file([[
        Message('note_on', note=60, velocity=100, time=0),      # tick 0
        pedal(127, time=240),                                   # tick 240
        MetaMessage('set_tempo', tempo=250000, time=240),       # tick 480
        Message('note_off', note=60, velocity=64, time=0),      # tick 480
        pedal(0, time=480),                                     # tick 960
        MetaMessage('end_of_track', time=0),
    ]])

    note = mid.note_timeline()[0]

    # 480 ticks @500000 (0.5 s) + 480 ticks @250000 (0.25 s)
    assert note.end_time == pytest.approx(0.75)
    assert note.start_time == pytest.approx(0.0)
    assert note.end_reason == END_REASON_NOTE_OFF


def test_pedal_state_is_per_channel():
    mid = make_file([[
        Message('note_on', channel=0, note=60, velocity=100, time=0),
        Message('note_on', channel=1, note=60, velocity=100, time=0),
        pedal(127, time=100),  # channel 0 pedal down
        Message('control_change', channel=1, control=64,
                value=127, time=0),
        Message('note_off', channel=0, note=60, velocity=64, time=100),
        Message('note_off', channel=1, note=60, velocity=64, time=100),
        Message('control_change', channel=0, control=64,
                value=0, time=100),
        Message('control_change', channel=1, control=64,
                value=0, time=0),
        MetaMessage('end_of_track', time=0),
    ]])

    timeline = mid.note_timeline()

    for note in timeline:
        # Both pedals are released at tick 400.
        assert note.end_time == pytest.approx(400 * 0.5 / 480)


# ---------------------------------------------------------------------
# End of file and unmatched events
# ---------------------------------------------------------------------

def test_hanging_note_ends_at_end_of_track():
    mid = make_file([[
        Message('note_on', note=60, velocity=100, time=0),
        MetaMessage('end_of_track', time=480),
    ]])

    note = mid.note_timeline()[0]

    assert note.end_time == pytest.approx(0.5)
    assert note.end_reason == END_REASON_END_OF_TRACK


def test_hanging_note_while_pedal_down():
    mid = make_file([[
        Message('note_on', note=60, velocity=100, time=0),
        pedal(127, time=100),
        Message('note_off', note=60, velocity=64, time=100),
        MetaMessage('end_of_track', time=200),
    ]])

    note = mid.note_timeline()[0]

    assert note.end_time == pytest.approx(400 * 0.5 / 480)
    assert note.end_reason == END_REASON_END_OF_TRACK


def test_hanging_note_uses_longest_track_end():
    mid = make_file([
        [
            Message('note_on', note=60, velocity=100, time=0),
            MetaMessage('end_of_track', time=960),
        ],
        [
            MetaMessage('end_of_track', time=480),
        ],
    ])

    note = mid.note_timeline()[0]

    assert note.end_time == pytest.approx(1.0)
    assert note.end_reason == END_REASON_END_OF_TRACK


def test_unmatched_note_off_is_recorded():
    mid = make_file([[
        Message('note_off', note=60, velocity=64, time=100),
        MetaMessage('end_of_track', time=0),
    ]])

    timeline = mid.note_timeline()

    assert list(timeline) == []
    assert timeline.unmatched_note_offs == [
        (0, 0, 60, pytest.approx(100 * 0.5 / 480)),
    ]


def test_unmatched_zero_velocity_note_on_is_recorded():
    mid = make_file([[
        Message('note_on', note=60, velocity=0, time=100),
        MetaMessage('end_of_track', time=0),
    ]])

    timeline = mid.note_timeline()

    assert list(timeline) == []
    assert len(timeline.unmatched_note_offs) == 1
    assert timeline.unmatched_note_offs[0][:3] == (0, 0, 60)


def test_extra_note_off_after_all_notes_closed():
    mid = make_file([[
        Message('note_on', note=60, velocity=100, time=0),
        Message('note_off', note=60, velocity=64, time=100),
        Message('note_off', note=60, velocity=64, time=100),
        MetaMessage('end_of_track', time=0),
    ]])

    timeline = mid.note_timeline()

    assert len(timeline) == 1
    assert len(timeline.unmatched_note_offs) == 1


def test_empty_file():
    mid = MidiFile()

    timeline = mid.note_timeline()

    assert isinstance(timeline, NoteTimeline)
    assert list(timeline) == []
    assert timeline.unmatched_note_offs == []


# ---------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------

def test_type_2_file_raises():
    mid = MidiFile(type=2)

    with pytest.raises(ValueError, match='type 2'):
        mid.note_timeline()


@pytest.mark.parametrize('ppq', [0, -1, -480])
def test_invalid_ppq_raises(ppq):
    mid = make_file([[
        Message('note_on', note=60, velocity=100, time=0),
        Message('note_off', note=60, velocity=64, time=10),
    ]], ticks_per_beat=ppq)

    with pytest.raises(ValueError, match='ticks_per_beat'):
        mid.note_timeline()


def test_negative_delta_time_raises():
    mid = make_file([[
        Message('note_on', note=60, velocity=100, time=100),
        Message('note_off', note=60, velocity=64, time=-200),
    ]])

    with pytest.raises(ValueError, match='negative absolute time'):
        mid.note_timeline()


# ---------------------------------------------------------------------
# Ordering, immutability and compatibility
# ---------------------------------------------------------------------

def test_timeline_is_sorted():
    mid = make_file([[
        Message('note_on', note=72, velocity=100, time=480),
        Message('note_off', note=72, velocity=64, time=100),
        Message('note_on', note=60, velocity=100, time=100),
        Message('note_off', note=60, velocity=64, time=100),
        MetaMessage('end_of_track', time=0),
    ]])

    timeline = mid.note_timeline()

    starts = [n.start_time for n in timeline]
    assert starts == sorted(starts)
    # Note 72 starts at tick 480, note 60 starts later at tick 680.
    assert [n.note for n in timeline] == [72, 60]


def test_input_midifile_is_not_modified():
    mid = make_file([[
        Message('note_on', note=60, velocity=100, time=0),
        pedal(127, time=240),
        Message('note_off', note=60, velocity=64, time=240),
        pedal(0, time=480),
        MetaMessage('end_of_track', time=0),
    ]])
    snapshot = copy.deepcopy(mid)

    mid.note_timeline()

    assert len(mid.tracks) == len(snapshot.tracks)
    for track, snapshot_track in zip(mid.tracks, snapshot.tracks):
        assert len(track) == len(snapshot_track)
        for message, snapshot_message in zip(track, snapshot_track):
            assert message == snapshot_message
            assert message.time == snapshot_message.time
    # The merged track cache must not have been created as a side effect.
    assert mid._merged_track is None


def test_save_bytes_identical_before_and_after():
    mid = make_file([[
        Message('note_on', note=60, velocity=100, time=0),
        pedal(127, time=240),
        Message('note_off', note=60, velocity=64, time=240),
        pedal(0, time=480),
        MetaMessage('end_of_track', time=0),
    ]])

    buffer_before = io.BytesIO()
    mid.save(file=buffer_before)
    mid.note_timeline()
    buffer_after = io.BytesIO()
    mid.save(file=buffer_after)

    assert buffer_before.getvalue() == buffer_after.getvalue()


def test_round_trip_through_file():
    mid = make_file([[
        Message('note_on', note=60, velocity=100, time=0),
        pedal(127, time=240),
        Message('note_off', note=60, velocity=64, time=240),
        pedal(0, time=480),
        MetaMessage('end_of_track', time=0),
    ]])
    buffer = io.BytesIO()
    mid.save(file=buffer)
    buffer.seek(0)

    loaded = MidiFile(file=buffer)
    note = loaded.note_timeline()[0]

    assert note.note == 60
    assert note.end_time == pytest.approx(1.0)


def test_function_entry_matches_method():
    mid = make_file([[
        Message('note_on', note=60, velocity=100, time=0),
        Message('note_off', note=60, velocity=64, time=480),
        MetaMessage('end_of_track', time=0),
    ]])

    from_function = get_note_timeline(mid)
    from_method = mid.note_timeline()

    assert from_function == from_method
    assert (from_function.unmatched_note_offs
            == from_method.unmatched_note_offs)


def test_merge_tracks_behavior_unchanged():
    mid = make_file([
        [
            Message('note_on', channel=0, note=60, velocity=100, time=100),
            Message('note_off', channel=0, note=60, velocity=64, time=380),
            MetaMessage('end_of_track', time=0),
        ],
        [
            MetaMessage('set_tempo', tempo=250000, time=0),
            Message('note_on', channel=1, note=72, velocity=90, time=200),
            Message('note_off', channel=1, note=72, velocity=64, time=280),
            MetaMessage('end_of_track', time=0),
        ],
    ])

    merged = merge_tracks(mid.tracks)
    merged_duration = sum(message.time for message in merged)
    track_durations = [
        sum(message.time for message in track) for track in mid.tracks
    ]

    assert merged_duration == max(track_durations)
    # Delta times are still in ticks and messages stay untouched.
    assert all(isinstance(message.time, int) for message in merged)


def test_sounding_note_equality_and_repr():
    note = SoundingNote(0, 1, 60, 100, 0.0, 0.5, END_REASON_NOTE_OFF)

    assert note == SoundingNote(0, 1, 60, 100, 0.0, 0.5,
                                END_REASON_NOTE_OFF)
    assert note != SoundingNote(0, 1, 60, 100, 0.0, 0.75,
                                END_REASON_NOTE_OFF)
    assert 'SoundingNote(' in repr(note)
