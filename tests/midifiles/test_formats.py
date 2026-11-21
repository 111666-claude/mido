# SPDX-FileCopyrightText: 2026 Mido contributors
#
# SPDX-License-Identifier: MIT

"""Tests for conversion between type 0 and type 1 MIDI files."""

import copy
import io

import pytest

from mido import (
    Message,
    MetaMessage,
    MidiFile,
    MidiTrack,
    UnknownMetaMessage,
    to_type0,
    to_type1,
)

# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------

def absolute_ticks(mid):
    """Return a multiset of ``(tick, message)`` pairs across all tracks.

    Message times are normalized to 0 so only the absolute tick position
    and the message itself are compared. ``end_of_track`` is excluded,
    since every track receives a fresh one on save.
    """
    events = []
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type != 'end_of_track':
                events.append((tick, msg.copy(time=0)))
    return sorted(events, key=lambda event: repr(event))


def absolute_timeline(mid):
    """Return ``(tick, message)`` pairs in merged playback order."""
    tick = 0
    result = []
    for msg in mid.merged_track:
        tick += msg.time
        if msg.type != 'end_of_track':
            result.append((tick, msg.copy(time=0)))
    return result


def save_and_read(mid):
    """Round-trip a MidiFile through bytes."""
    buffer = io.BytesIO()
    mid.save(file=buffer)
    buffer.seek(0)
    return MidiFile(file=buffer, charset=mid.charset)


@pytest.fixture
def type1_file():
    """A type 1 file exercising every edge case at once.

    * tempo changes (variable tempo) on the global track,
    * several channels with events landing on the same tick,
    * a SysEx dump and an unknown meta message,
    * a completely empty track,
    * duplicate end-of-track-less tracks and delta times spanning
      several beats.
    """
    mid = MidiFile(type=1, ticks_per_beat=480)

    tempo_track = MidiTrack()
    tempo_track.name = 'Tempo'
    tempo_track.append(MetaMessage('set_tempo', tempo=500000, time=0))
    tempo_track.append(
        MetaMessage('time_signature', numerator=4, denominator=4, time=480))
    tempo_track.append(MetaMessage('set_tempo', tempo=250000, time=480))
    tempo_track.append(MetaMessage('key_signature', key='Gm', time=0))

    piano = MidiTrack()
    piano.name = 'Piano'
    piano.append(Message('note_on', channel=0, note=60, velocity=64, time=0))
    piano.append(Message('control_change', channel=0,
                         control=7, value=100, time=0))
    piano.append(Message('note_off', channel=0, note=60, velocity=64, time=480))

    bass = MidiTrack()
    bass.name = 'Bass'
    # Same-tick event as the piano note above.
    bass.append(Message('note_on', channel=1, note=36, velocity=64, time=0))
    bass.append(Message('sysex', data=[1, 2, 3], time=120))
    bass.append(Message('note_off', channel=1, note=36, velocity=64, time=360))
    # Unknown meta message must survive the conversion untouched.
    bass.append(UnknownMetaMessage(0x4d, [7, 8], time=0))

    drums = MidiTrack()
    drums.name = 'Drums'
    drums.append(Message('note_on', channel=9, note=35, velocity=127, time=0))
    drums.append(Message('note_off', channel=9, note=35, velocity=0, time=240))

    empty = MidiTrack()  # Deliberately empty.

    mid.tracks = [tempo_track, piano, bass, drums, empty]
    return mid


@pytest.fixture
def type0_file(type1_file):
    return type1_file.to_type0()


# ---------------------------------------------------------------------------
# Type 1 -> type 0
# ---------------------------------------------------------------------------

def test_type1_to_type0_structure(type1_file):
    mid = to_type0(type1_file)

    assert mid.type == 0
    assert len(mid.tracks) == 1
    assert mid.ticks_per_beat == type1_file.ticks_per_beat
    assert mid is not type1_file


def test_type1_to_type0_preserves_absolute_ticks(type1_file):
    assert absolute_ticks(type1_file) == absolute_ticks(type1_file.to_type0())


def test_type1_to_type0_same_tick_order(type1_file):
    # Same-tick events must keep their source track order after merging.
    merged = type1_file.to_type0().tracks[0]
    tick = 0
    zero_tick = []
    for msg in merged:
        tick += msg.time
        if tick == 0:
            zero_tick.append(msg)

    # note_on before the same-channel control_change that follows it.
    types_at_zero = [msg.type for msg in zero_tick]
    assert types_at_zero.index('note_on') < types_at_zero.index(
        'control_change')

    # Piano (source track 1, channel 0) before bass (track 2, ch 1)
    # before drums (track 3, ch 9).
    channels = [msg.channel for msg in zero_tick if msg.type == 'note_on']
    assert channels == [0, 1, 9]


def test_type1_to_type0_single_end_of_track(type0_file):
    track = type0_file.tracks[0]
    eot = [msg for msg in track if msg.type == 'end_of_track']
    assert len(eot) == 1
    assert eot[0] is track[-1]


def test_type1_to_type0_handles_empty_tracks():
    mid = MidiFile(type=1)
    mid.tracks = [MidiTrack(), MidiTrack([
        Message('note_on', channel=0, note=60, time=0),
        Message('note_off', channel=0, note=60, time=10),
    ]), MidiTrack()]

    converted = mid.to_type0()
    assert len(converted.tracks) == 1
    notes = [msg for msg in converted.tracks[0] if msg.type.startswith('note')]
    assert len(notes) == 2


def test_type1_to_type0_sysex_and_unknown_meta_preserved(type1_file):
    merged = type1_file.to_type0().tracks[0]
    sysex = [msg for msg in merged if msg.type == 'sysex']
    unknown = [msg for msg in merged if isinstance(msg, UnknownMetaMessage)]
    assert len(sysex) == 1
    assert sysex[0].data == (1, 2, 3)
    assert len(unknown) == 1
    assert unknown[0].type_byte == 0x4d
    assert unknown[0].data == (7, 8)


def test_type0_input_to_type0_returns_independent_copy():
    mid = MidiFile(type=0)
    mid.tracks = [MidiTrack([
        Message('note_on', channel=0, note=60, time=0),
        Message('note_off', channel=0, note=60, time=10),
    ])]

    converted = mid.to_type0()
    assert converted is not mid
    assert converted.tracks[0] is not mid.tracks[0]
    # merge_tracks normalizes the end_of_track marker.
    converted_events = [msg for msg in converted.tracks[0]
                        if msg.type != 'end_of_track']
    assert converted_events == list(mid.tracks[0])


# ---------------------------------------------------------------------------
# Type 0 -> type 1
# ---------------------------------------------------------------------------

def test_type0_to_type1_structure(type0_file):
    mid = to_type1(type0_file)

    assert mid.type == 1
    assert mid is not type0_file
    # Conductor + channels 0, 1, 9.
    assert len(mid.tracks) == 4


def test_type0_to_type1_global_meta_in_first_track(type0_file):
    mid = type0_file.to_type1()
    conductor = mid.tracks[0]

    global_types = {msg.type for msg in conductor
                    if msg.is_meta and msg.type != 'track_name'}
    assert {'set_tempo', 'time_signature', 'key_signature'} <= global_types

    for track in mid.tracks[1:]:
        for msg in track:
            assert not (msg.is_meta and msg.type in {
                'set_tempo', 'time_signature', 'key_signature'})


def test_type0_to_type1_splits_by_channel(type0_file):
    mid = type0_file.to_type1()
    for track in mid.tracks[1:]:
        channels = {msg.channel for msg in track if hasattr(msg, 'channel')}
        assert len(channels) == 1

    all_channels = set()
    for track in mid.tracks[1:]:
        all_channels.update(
            msg.channel for msg in track if hasattr(msg, 'channel'))
    assert all_channels == {0, 1, 9}


def test_type0_to_type1_sysex_and_unknown_meta_go_to_conductor(type0_file):
    conductor = type0_file.to_type1().tracks[0]
    assert any(msg.type == 'sysex' for msg in conductor)
    assert any(isinstance(msg, UnknownMetaMessage) for msg in conductor)


def test_type0_to_type1_preserves_absolute_ticks(type0_file):
    assert absolute_ticks(type0_file) == absolute_ticks(type0_file.to_type1())


def test_type0_to_type1_preserves_order_within_channel(type1_file):
    type0 = type1_file.to_type0()
    split = type0.to_type1()

    for channel in (0, 1, 9):
        original = [
            (tick, msg.copy(time=0))
            for tick, msg in absolute_timeline(type0)
            if getattr(msg, 'channel', None) == channel
        ]
        split_track = next(
            track for track in split.tracks[1:]
            if any(getattr(m, 'channel', None) == channel for m in track))
        tick = 0
        rebuilt = []
        for msg in split_track:
            tick += msg.time
            rebuilt.append((tick, msg.copy(time=0)))
        assert rebuilt == original


def test_type0_to_type1_empty_file_has_no_tracks():
    mid = MidiFile(type=0)
    mid.tracks = [MidiTrack()]

    converted = mid.to_type1()
    assert converted.type == 1
    assert converted.tracks == []


def test_type1_to_type1_is_idempotent(type1_file, type0_file):
    once = type0_file.to_type1()
    twice = once.to_type1()

    assert absolute_ticks(once) == absolute_ticks(twice)
    assert [t.name for t in once.tracks] == [t.name for t in twice.tracks]


def test_type0_with_multiple_in_memory_tracks_merges_them():
    # Such a file can't be saved, but splitting it must still merge
    # every source track before grouping events by channel.
    mid = MidiFile(type=0)
    mid.tracks = [
        MidiTrack([Message('note_on', channel=0, note=60, time=0),
                   Message('note_off', channel=0, note=60, time=10)]),
        MidiTrack([Message('note_on', channel=1, note=48, time=5),
                   Message('note_off', channel=1, note=48, time=5)]),
    ]

    converted = mid.to_type1()
    assert converted.type == 1
    assert len(converted.tracks) == 2
    assert absolute_ticks(converted) == absolute_ticks(mid)


def test_type0_to_type1_no_shared_messages(type0_file):
    split = type0_file.to_type1()
    source_ids = {id(msg) for track in type0_file.tracks for msg in track}
    split_ids = {id(msg) for track in split.tracks for msg in track}
    assert source_ids.isdisjoint(split_ids)


def test_end_of_track_messages_normalized_on_split():
    mid = MidiFile(type=1)
    # end_of_track followed by another event in the same track: its
    # delta time must be carried over, not silently lost.
    mid.tracks = [
        MidiTrack([
            Message('note_on', channel=0, note=60, time=0),
            MetaMessage('end_of_track', time=10),
            Message('note_off', channel=0, note=60, time=5),
            MetaMessage('end_of_track', time=0),
        ]),
        MidiTrack([
            MetaMessage('set_tempo', tempo=400000, time=0),
        ]),
    ]

    split = mid.to_type1()
    assert not any(
        msg.type == 'end_of_track'
        for track in split.tracks for msg in track)

    read_back = save_and_read(split)
    for track in read_back.tracks:
        eot = [msg for msg in track if msg.type == 'end_of_track']
        assert len(eot) == 1
        assert eot[0] is track[-1]

    # The note_off still lands on absolute tick 15.
    channel_track = next(
        track for track in read_back.tracks
        if any(msg.type == 'note_off' for msg in track))
    tick = 0
    positions = {}
    for msg in channel_track:
        tick += msg.time
        positions[msg.type] = tick
    assert positions['note_off'] == 15


# ---------------------------------------------------------------------------
# Conflicting track names
# ---------------------------------------------------------------------------

def test_conflicting_track_names_are_deduplicated():
    mid = MidiFile(type=1)
    first = MidiTrack()
    first.name = 'Piano'
    first.append(Message('note_on', channel=0, note=60, time=0))

    second = MidiTrack()
    second.name = 'Piano'
    second.append(Message('note_on', channel=1, note=48, time=0))

    third = MidiTrack()
    third.name = 'Piano (2)'
    third.append(Message('note_on', channel=2, note=36, time=0))

    mid.tracks = [first, second, third]

    names = [msg.name for track in mid.to_type1().tracks
             for msg in track if msg.type == 'track_name']
    assert len(names) == len(set(names))
    # Deterministic: the first duplicate gets suffix (2); when that
    # name is already taken, another (2) is appended rather than guessing.
    assert names == ['Piano', 'Piano (2)', 'Piano (2) (2)']


def test_track_name_rename_does_not_touch_source():
    mid = MidiFile(type=1)
    first = MidiTrack()
    first.name = 'Piano'
    first.append(Message('note_on', channel=0, note=60, time=0))
    second = MidiTrack()
    second.name = 'Piano'
    second.append(Message('note_on', channel=1, note=48, time=0))
    mid.tracks = [first, second]

    mid.to_type1()

    assert first.name == 'Piano'
    assert second.name == 'Piano'


# ---------------------------------------------------------------------------
# Type 2 and illegal input
# ---------------------------------------------------------------------------

def test_type2_cannot_convert_to_type0():
    mid = MidiFile(type=2)
    mid.tracks.append(MidiTrack([Message('note_on', time=0)]))
    with pytest.raises(TypeError):
        to_type0(mid)
    with pytest.raises(TypeError):
        mid.to_type0()


def test_type2_cannot_convert_to_type1():
    mid = MidiFile(type=2)
    mid.tracks.append(MidiTrack([Message('note_on', time=0)]))
    with pytest.raises(TypeError):
        to_type1(mid)
    with pytest.raises(TypeError):
        mid.to_type1()


def test_negative_time_rejected():
    mid = MidiFile(type=1)
    mid.tracks.append(MidiTrack([Message('note_on', time=-1)]))
    with pytest.raises(ValueError):
        mid.to_type0()
    with pytest.raises(ValueError):
        mid.to_type1()


def test_non_integer_time_rejected():
    mid = MidiFile(type=1)
    mid.tracks.append(MidiTrack([Message('note_on', time=1.5)]))
    with pytest.raises(ValueError):
        mid.to_type0()
    with pytest.raises(ValueError):
        mid.to_type1()


# ---------------------------------------------------------------------------
# Source immutability
# ---------------------------------------------------------------------------

def test_source_not_modified(type1_file):
    snapshot = copy.deepcopy(type1_file)
    type1_file.to_type0()
    type1_file.to_type1()

    assert repr(type1_file) == repr(snapshot)


# ---------------------------------------------------------------------------
# Save / read-back
# ---------------------------------------------------------------------------

def test_type0_conversion_saves_and_reads_back(type1_file):
    converted = type1_file.to_type0()
    read_back = save_and_read(converted)

    assert read_back.type == 0
    assert len(read_back.tracks) == 1
    assert absolute_ticks(read_back) == absolute_ticks(type1_file)


def test_type1_conversion_saves_and_reads_back(type0_file, type1_file):
    converted = type0_file.to_type1()
    read_back = save_and_read(converted)

    assert read_back.type == 1
    assert absolute_ticks(read_back) == absolute_ticks(type1_file)


def test_round_trip_through_both_formats(type1_file):
    type0 = type1_file.to_type0()
    back = type0.to_type1()
    read_back = save_and_read(save_and_read(back).to_type0())

    assert read_back.type == 0
    assert absolute_ticks(read_back) == absolute_ticks(type1_file)


def test_unknown_meta_bytes_survive_save(type0_file):
    split = type0_file.to_type1()
    read_back = save_and_read(split)

    unknown = [msg for track in read_back.tracks for msg in track
               if isinstance(msg, UnknownMetaMessage)]
    assert len(unknown) == 1
    assert unknown[0].type_byte == 0x4d
    assert unknown[0].data == (7, 8)


def test_variable_tempo_survives_conversion(type1_file):
    tempos = [
        msg.tempo
        for tick, msg in absolute_timeline(type1_file)
        if msg.type == 'set_tempo'
    ]
    converted = type1_file.to_type0().to_type1()
    converted_tempos = [
        msg.tempo
        for tick, msg in absolute_timeline(converted)
        if msg.type == 'set_tempo'
    ]
    assert converted_tempos == tempos == [500000, 250000]
