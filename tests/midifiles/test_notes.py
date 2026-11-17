# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

import io

import mido
from mido.midifiles.notes import Note, get_note_intervals


def make_file(tracks, type=1, ticks_per_beat=480):
    mid = mido.MidiFile(type=type, ticks_per_beat=ticks_per_beat)
    for messages in tracks:
        track = mido.MidiTrack()
        for msg in messages:
            track.append(msg)
        mid.tracks.append(track)
    return mid


def on(note, velocity=100, channel=0, time=0):
    return mido.Message('note_on', note=note, velocity=velocity,
                        channel=channel, time=time)


def off(note, velocity=0, channel=0, time=0):
    return mido.Message('note_off', note=note, velocity=velocity,
                        channel=channel, time=time)


def cc(control, value, channel=0, time=0):
    return mido.Message('control_change', control=control, value=value,
                        channel=channel, time=time)


def pedal_file():
    """Fixed multi-track sample with a sustain pedal on track 1.

    Track 1 (pedal, channel 0):  down at tick 0, up at tick 1000.
    Track 2 (notes, channel 0):  60 released at 400 (sustained),
                                 64 released at 800 (sustained),
                                 67 played 1100-1300 (pedal up).
    Track 3 (notes, channel 1):  60 played 100-600 (other channel,
                                 not affected by the pedal).
    """
    return make_file([
        [cc(64, 127, time=0),
         cc(64, 0, time=1000)],
        [on(60, velocity=100, time=0),
         off(60, time=400),
         on(64, velocity=90, time=100),
         off(64, time=300),
         on(67, velocity=80, time=300),
         off(67, time=200)],
        [on(60, velocity=70, channel=1, time=100),
         off(60, channel=1, time=500)],
    ])


PEDAL_FILE_NOTES = [
    Note(channel=0, note=60, velocity=100, start=0, end=1000),
    Note(channel=1, note=60, velocity=70, start=100, end=600),
    Note(channel=0, note=64, velocity=90, start=500, end=1000),
    Note(channel=0, note=67, velocity=80, start=1100, end=1300),
]


def test_pedal_release_extends_notes_multitrack():
    # Notes released while the pedal is down end at the pedal
    # release (tick 1000), not at their note-off. Notes played
    # after the release end at their own note-off.
    assert get_note_intervals(pedal_file()) == PEDAL_FILE_NOTES


def test_pedal_file_roundtrip_through_save_load():
    # The same sample saved to a real MIDI file and read back must
    # give the same intervals.
    outfile = io.BytesIO()
    pedal_file().save(file=outfile)
    outfile.seek(0)
    mid = mido.MidiFile(file=outfile)
    assert get_note_intervals(mid) == PEDAL_FILE_NOTES
    assert mid.note_intervals() == PEDAL_FILE_NOTES


def test_simple_interval():
    mid = make_file([[on(60, time=0), off(60, time=240)]])
    assert get_note_intervals(mid) == [
        Note(channel=0, note=60, velocity=100, start=0, end=240),
    ]


def test_note_on_velocity_zero_is_note_off():
    mid = make_file([[on(60, time=0), on(60, velocity=0, time=240)]])
    assert get_note_intervals(mid) == [
        Note(channel=0, note=60, velocity=100, start=0, end=240),
    ]


def test_repeated_note_on_retriggers():
    # A second note-on on a sounding key ends the first interval.
    mid = make_file([[on(60, velocity=100, time=0),
                      on(60, velocity=80, time=100),
                      off(60, time=100)]])
    assert get_note_intervals(mid) == [
        Note(channel=0, note=60, velocity=100, start=0, end=100),
        Note(channel=0, note=60, velocity=80, start=100, end=200),
    ]


def test_retrigger_of_sustained_note():
    # Note released under the pedal, then restruck before the pedal
    # comes up: the sustained interval ends at the retrigger.
    mid = make_file([[cc(64, 127, time=0),
                      on(60, velocity=100, time=0),
                      off(60, time=100),
                      on(60, velocity=80, time=100),
                      cc(64, 0, time=200),
                      off(60, time=0)]])
    assert get_note_intervals(mid) == [
        Note(channel=0, note=60, velocity=100, start=0, end=200),
        Note(channel=0, note=60, velocity=80, start=200, end=400),
    ]


def test_orphan_note_off_is_ignored():
    mid = make_file([[off(60, time=10),
                      on(60, time=10),
                      off(60, time=100),
                      off(60, time=10)]])
    assert get_note_intervals(mid) == [
        Note(channel=0, note=60, velocity=100, start=20, end=120),
    ]


def test_note_held_at_end_of_file():
    # A note that is never released ends at the last tick of the file.
    mid = make_file([[on(60, time=0), on(64, time=100), off(64, time=100)]])
    assert get_note_intervals(mid) == [
        Note(channel=0, note=60, velocity=100, start=0, end=200),
        Note(channel=0, note=64, velocity=100, start=100, end=200),
    ]


def test_sustained_note_held_at_end_of_file():
    mid = make_file([[cc(64, 127, time=0),
                      on(60, time=0),
                      off(60, time=100)]])
    assert get_note_intervals(mid) == [
        Note(channel=0, note=60, velocity=100, start=0, end=100),
    ]


def test_all_notes_off_overrides_pedal():
    mid = make_file([[cc(64, 127, time=0),
                      on(60, time=0),
                      off(60, time=100),
                      on(64, time=0),
                      cc(123, 0, time=100),
                      cc(64, 0, time=100)]])
    assert get_note_intervals(mid) == [
        Note(channel=0, note=60, velocity=100, start=0, end=200),
        Note(channel=0, note=64, velocity=100, start=100, end=200),
    ]


def test_all_notes_off_is_per_channel():
    mid = make_file([[on(60, channel=0, time=0),
                      on(60, channel=1, time=0),
                      cc(123, 0, channel=0, time=100),
                      off(60, channel=1, time=100)]])
    assert get_note_intervals(mid) == [
        Note(channel=0, note=60, velocity=100, start=0, end=100),
        Note(channel=1, note=60, velocity=100, start=0, end=200),
    ]


def test_same_tick_pedal_down_before_note_off():
    # Pedal down comes first in the stream, so the note-off at the
    # same tick is sustained.
    mid = make_file([[on(60, time=0),
                      cc(64, 127, time=100),
                      off(60, time=0),
                      cc(64, 0, time=100)]])
    assert get_note_intervals(mid) == [
        Note(channel=0, note=60, velocity=100, start=0, end=200),
    ]


def test_same_tick_note_off_before_pedal_down():
    # Note-off comes first in the stream, so the note ends there
    # even though the pedal goes down at the same tick.
    mid = make_file([[on(60, time=0),
                      off(60, time=100),
                      cc(64, 127, time=0),
                      cc(64, 0, time=100)]])
    assert get_note_intervals(mid) == [
        Note(channel=0, note=60, velocity=100, start=0, end=100),
    ]


def test_same_tick_note_off_and_pedal_up():
    # Both orders end the note at the same tick.
    mid = make_file([[cc(64, 127, time=0),
                      on(60, time=0),
                      off(60, time=100),
                      cc(64, 0, time=0)]])
    assert get_note_intervals(mid) == [
        Note(channel=0, note=60, velocity=100, start=0, end=100),
    ]


def test_no_tempo_messages():
    # Intervals are in ticks, so a file without any tempo or other
    # meta messages works fine.
    mid = make_file([[on(60, time=0), off(60, time=480)]])
    assert get_note_intervals(mid) == [
        Note(channel=0, note=60, velocity=100, start=0, end=480),
    ]


def test_intervals_are_sorted_stably():
    mid = make_file([
        [on(64, channel=0, time=0), off(64, channel=0, time=300)],
        [on(60, channel=1, time=0), off(60, channel=1, time=100)],
        [on(60, channel=0, time=0), off(60, channel=0, time=200)],
    ])
    assert get_note_intervals(mid) == [
        Note(channel=1, note=60, velocity=100, start=0, end=100),
        Note(channel=0, note=60, velocity=100, start=0, end=200),
        Note(channel=0, note=64, velocity=100, start=0, end=300),
    ]


def test_empty_file():
    assert get_note_intervals(mido.MidiFile()) == []


def test_note_duration():
    note = Note(channel=0, note=60, velocity=100, start=100, end=250)
    assert note.duration == 150
