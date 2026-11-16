# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

import io

from pytest import raises

from mido.messages import Message
from mido.midifiles.meta import MetaMessage
from mido.midifiles.midifiles import MidiFile, MidiTrack
from mido.midifiles.mpe import (
    MPENote,
    MPEZone,
    extract_mpe_notes,
    write_mpe_notes,
)

# A lower zone with master channel 0 and two member channels.
ZONE = MPEZone.lower(2)


def two_bend_file():
    """Fixture: two simultaneous notes bending in different directions.

    One track, ticks_per_beat=480. Note 60 on member channel 1 bends
    up while note 64 on member channel 2 bends down. A CC message, a
    master-channel message and meta messages share the same ticks and
    must survive a read/write round trip untouched and in order.
    """
    track = MidiTrack()
    track.append(MetaMessage('track_name', name='mpe', time=0))
    track.append(Message('note_on', channel=1, note=60, velocity=100, time=0))
    track.append(Message('note_on', channel=2, note=64, velocity=90, time=0))
    track.append(Message('pitchwheel', channel=1, pitch=1000, time=240))
    track.append(Message('pitchwheel', channel=2, pitch=-1000, time=0))
    track.append(Message('control_change', channel=1, control=74,
                         value=64, time=0))
    track.append(Message('aftertouch', channel=1, value=50, time=240))
    track.append(Message('pitchwheel', channel=1, pitch=2000, time=0))
    track.append(Message('pitchwheel', channel=2, pitch=-2000, time=0))
    # Zone-wide message on the master channel: not per-note data.
    track.append(Message('pitchwheel', channel=0, pitch=0, time=0))
    track.append(Message('note_off', channel=1, note=60, velocity=0,
                         time=240))
    track.append(Message('note_off', channel=2, note=64, velocity=0,
                         time=0))
    # A third note reuses member channel 1 after the first note ends.
    track.append(Message('note_on', channel=1, note=67, velocity=80,
                         time=120))
    track.append(Message('pitchwheel', channel=1, pitch=500, time=120))
    track.append(Message('note_off', channel=1, note=67, velocity=0,
                         time=120))
    track.append(MetaMessage('end_of_track', time=0))
    return MidiFile(type=1, ticks_per_beat=480, tracks=[track])


def test_zone_layout():
    assert ZONE.master_channel == 0
    assert ZONE.member_channels == (1, 2)
    upper = MPEZone.upper(2)
    assert upper.master_channel == 15
    assert upper.member_channels == (13, 14)


def test_zone_invalid_layout():
    with raises(ValueError):
        MPEZone(0, [])  # no member channels
    with raises(ValueError):
        MPEZone(1, [1, 2])  # master is also a member
    with raises(ValueError):
        MPEZone(0, [16])  # channel out of range


def test_extract_associates_expression_per_note():
    notes = extract_mpe_notes(two_bend_file(), [ZONE])
    assert len(notes) == 3

    first, second, third = notes
    # Each note keeps its own pitch bend trajectory even though the
    # two bends are interleaved in the file.
    assert (first.note, first.channel) == (60, 1)
    assert first.start == 0 and first.end == 720
    assert first.pitch == [(240, 1000), (480, 2000)]
    assert first.pressure == [(480, 50)]

    assert (second.note, second.channel) == (64, 2)
    assert second.start == 0 and second.end == 720
    assert second.pitch == [(240, -1000), (480, -2000)]
    assert second.pressure == []

    # The third note reuses channel 1 without inheriting anything.
    assert (third.note, third.channel) == (67, 1)
    assert third.start == 840 and third.end == 1080
    assert third.pitch == [(960, 500)]


def test_extract_master_channel_is_not_per_note():
    notes = extract_mpe_notes(two_bend_file(), [ZONE])
    # The pitchwheel on master channel 0 belongs to no note.
    assert all((480, 0) not in note.pitch for note in notes)


def test_extract_requires_zones():
    with raises(ValueError):
        extract_mpe_notes(two_bend_file(), [])


def test_extract_overlapping_zones():
    with raises(ValueError):
        extract_mpe_notes(two_bend_file(), [ZONE, MPEZone.lower(3)])


def test_extract_conflict_closes_sounding_note():
    track = MidiTrack([
        Message('note_on', channel=1, note=60, velocity=100, time=0),
        Message('note_on', channel=1, note=62, velocity=100, time=480),
        Message('note_off', channel=1, note=62, velocity=0, time=480),
    ])
    mid = MidiFile(type=0, tracks=[track])
    notes = extract_mpe_notes(mid, [ZONE])
    assert [(n.note, n.start, n.end) for n in notes] == [
        (60, 0, 480), (62, 480, 960)]


def test_extract_conflict_error():
    track = MidiTrack([
        Message('note_on', channel=1, note=60, velocity=100, time=0),
        Message('note_on', channel=1, note=62, velocity=100, time=480),
    ])
    mid = MidiFile(type=0, tracks=[track])
    with raises(ValueError):
        extract_mpe_notes(mid, [ZONE], on_conflict='error')


def test_extract_orphan_note_off_ignored_by_default():
    track = MidiTrack([
        Message('note_off', channel=1, note=60, velocity=0, time=0),
        Message('note_on', channel=1, note=60, velocity=100, time=480),
        Message('note_off', channel=1, note=60, velocity=0, time=480),
    ])
    mid = MidiFile(type=0, tracks=[track])
    notes = extract_mpe_notes(mid, [ZONE])
    assert [(n.note, n.start, n.end) for n in notes] == [(60, 480, 960)]


def test_extract_orphan_note_off_error():
    track = MidiTrack([
        Message('note_off', channel=1, note=60, velocity=0, time=0),
    ])
    mid = MidiFile(type=0, tracks=[track])
    with raises(ValueError):
        extract_mpe_notes(mid, [ZONE], on_orphan='error')


def test_extract_note_on_velocity_zero_is_note_off():
    track = MidiTrack([
        Message('note_on', channel=1, note=60, velocity=100, time=0),
        Message('note_on', channel=1, note=60, velocity=0, time=480),
    ])
    mid = MidiFile(type=0, tracks=[track])
    (note,) = extract_mpe_notes(mid, [ZONE])
    assert (note.start, note.end) == (0, 480)


def test_extract_unclosed_note_ends_with_track():
    track = MidiTrack([
        Message('note_on', channel=1, note=60, velocity=100, time=0),
        Message('pitchwheel', channel=1, pitch=100, time=480),
    ])
    mid = MidiFile(type=0, tracks=[track])
    (note,) = extract_mpe_notes(mid, [ZONE])
    assert note.end == 480


def test_round_trip_preserves_notes_and_other_messages():
    mid = two_bend_file()
    notes = extract_mpe_notes(mid, [ZONE])

    # Edit: bend the first note further and swap nothing else.
    notes[0].pitch = [(tick, value * 2) for tick, value in notes[0].pitch]

    write_mpe_notes(mid, [ZONE], notes)

    # Non-MPE messages survive in order, with absolute ticks intact.
    kept = []
    tick = 0
    for msg in mid.tracks[0]:
        tick += msg.time
        if msg.is_meta or msg.type == 'control_change' or msg.channel == 0:
            kept.append((tick, msg.type))
    assert kept == [
        (0, 'track_name'),
        (240, 'control_change'),
        (480, 'pitchwheel'),  # master channel message
        (1080, 'end_of_track'),
    ]

    # Re-extraction sees the edited trajectories.
    new_notes = extract_mpe_notes(mid, [ZONE])
    assert new_notes == notes
    assert new_notes[0].pitch == [(240, 2000), (480, 4000)]
    assert new_notes[1].pitch == [(240, -1000), (480, -2000)]


def test_round_trip_through_saved_file():
    mid = two_bend_file()
    notes = extract_mpe_notes(mid, [ZONE])
    write_mpe_notes(mid, [ZONE], notes)

    outfile = io.BytesIO()
    mid.save(file=outfile)
    outfile.seek(0)
    reread = MidiFile(file=outfile)

    assert extract_mpe_notes(reread, [ZONE]) == notes


def test_write_allocates_channels_without_crosstalk():
    # Three simultaneous notes in a two-member-channel zone cannot fit.
    notes = [
        MPENote(60, channel=1, start=0, end=960),
        MPENote(62, channel=2, start=0, end=960),
        MPENote(64, channel=1, start=0, end=960),
    ]
    with raises(ValueError):
        write_mpe_notes(MidiFile(type=0, tracks=[MidiTrack()]), [ZONE], notes)

    # A note whose original channel is busy is moved to a free one.
    notes = [
        MPENote(60, channel=1, start=0, end=960),
        MPENote(64, channel=1, start=0, end=960),
    ]
    mid = MidiFile(type=0, tracks=[MidiTrack()])
    write_mpe_notes(mid, [ZONE], notes)
    assert [note.channel for note in notes] == [1, 2]
    channels = {msg.note: msg.channel for msg in mid.tracks[0]
                if msg.type == 'note_on'}
    assert channels == {60: 1, 64: 2}


def test_write_channel_reuse_at_same_tick():
    # A channel freed at tick 480 can be reused at tick 480: the
    # note-off is written before the new note-on.
    notes = [
        MPENote(60, channel=1, start=0, end=480),
        MPENote(62, channel=1, start=480, end=960),
    ]
    mid = MidiFile(type=0, tracks=[MidiTrack()])
    write_mpe_notes(mid, [ZONE], notes)
    assert [note.channel for note in notes] == [1, 1]
    types = [msg.type for msg in mid.tracks[0]]
    assert types == ['note_on', 'note_off', 'note_on', 'note_off']
    assert extract_mpe_notes(mid, [ZONE]) == notes


def test_write_clips_out_of_range_values():
    notes = [MPENote(60, velocity=200, channel=1, start=0, end=480,
                     pitch=[(0, 99999)], pressure=[(240, -5)])]
    mid = MidiFile(type=0, tracks=[MidiTrack()])
    write_mpe_notes(mid, [ZONE], notes)
    (note,) = extract_mpe_notes(mid, [ZONE])
    assert note.velocity == 127
    assert note.pitch == [(0, 8191)]
    assert note.pressure == [(240, 0)]


def test_write_strict_raises_on_out_of_range():
    notes = [MPENote(60, channel=1, start=0, end=480, pitch=[(0, 99999)])]
    mid = MidiFile(type=0, tracks=[MidiTrack()])
    with raises(ValueError):
        write_mpe_notes(mid, [ZONE], notes, strict=True)


def test_write_requires_zones_and_valid_track():
    notes = [MPENote(60, channel=1, start=0, end=480)]
    with raises(ValueError):
        write_mpe_notes(MidiFile(type=0, tracks=[MidiTrack()]), [], notes)
    with raises(ValueError):
        write_mpe_notes(MidiFile(type=0, tracks=[MidiTrack()]), [ZONE],
                        [MPENote(60, channel=1, track=5, start=0, end=480)])


def test_write_keeps_tracks_and_same_tick_order():
    # Notes live in track 1; track 0 holds tempo/meta only.
    meta_track = MidiTrack([
        MetaMessage('set_tempo', tempo=400000, time=0),
        MetaMessage('track_name', name='conductor', time=0),
    ])
    note_track = MidiTrack([
        Message('control_change', channel=3, control=7, value=100, time=0),
        Message('control_change', channel=3, control=10, value=64, time=0),
        Message('note_on', channel=1, note=60, velocity=100, time=0),
        Message('note_off', channel=1, note=60, velocity=0, time=480),
    ])
    mid = MidiFile(type=1, tracks=[meta_track, note_track])

    notes = extract_mpe_notes(mid, [ZONE])
    assert len(notes) == 1 and notes[0].track == 1
    notes[0].pitch = [(240, 1000)]
    write_mpe_notes(mid, [ZONE], notes)

    # Track structure and non-member messages are untouched.
    assert len(mid.tracks) == 2
    assert mid.tracks[0] == meta_track
    ccs = [msg for msg in mid.tracks[1] if msg.type == 'control_change']
    assert [(msg.control, msg.time) for msg in ccs] == [(7, 0), (10, 0)]
    # Same-tick order of the two CCs is preserved (stable).
    assert [msg.control for msg in ccs] == [7, 10]
