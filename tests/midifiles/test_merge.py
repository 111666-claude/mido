# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

import io

from pytest import raises

from mido.messages import Message
from mido.midifiles.merge import merge_midifiles
from mido.midifiles.meta import MetaMessage
from mido.midifiles.midifiles import MidiFile
from mido.midifiles.tracks import MidiTrack

EOT = MetaMessage('end_of_track', time=0)


def make_file(type=1, ticks_per_beat=480, tracks=()):
    return MidiFile(type=type,
                    ticks_per_beat=ticks_per_beat,
                    tracks=[MidiTrack(track) for track in tracks])


def note(track, delta, type='note_on', channel=0, note=64, velocity=64):
    track.append(Message(type, channel=channel, note=note,
                         velocity=velocity, time=delta))


def abs_times(track):
    now = 0
    result = []
    for msg in track:
        now += msg.time
        result.append((now, msg))
    return result


def test_different_ticks_per_beat():
    # Primary: 480 ticks per beat, default tempo (0.5 s per beat).
    primary = make_file(ticks_per_beat=480, tracks=[[
        Message('note_on', channel=0, note=60, velocity=64, time=0),
        Message('note_off', channel=0, note=60, velocity=64, time=480),
        EOT,
    ]])
    # Secondary: 240 ticks per beat, tempo 1000000 (1 s per beat), so
    # the note_on at tick 240 plays at 1.0 s -> tick 960 on the
    # primary timeline.
    secondary = make_file(ticks_per_beat=240, tracks=[[
        MetaMessage('set_tempo', tempo=1000000, time=0),
        Message('note_on', channel=1, note=64, velocity=64, time=240),
        Message('note_off', channel=1, note=64, velocity=64, time=240),
        EOT,
    ]])

    merged = merge_midifiles(primary, secondary)

    assert merged.type == 1
    assert merged.ticks_per_beat == 480
    assert len(merged.tracks) == 2

    secondary_track = abs_times(merged.tracks[1])
    # The secondary set_tempo is used for conversion, not copied.
    assert [msg.type for _, msg in secondary_track] == [
        'note_on', 'note_off', 'end_of_track']
    assert secondary_track[0][0] == 960
    assert secondary_track[1][0] == 1920


def test_tempo_changes_in_both_files():
    # Primary: 480 tpb, tempo 500000, at tick 480 (0.5 s) tempo
    # changes to 250000 (0.25 s per beat -> 480 ticks = 0.25 s... so
    # 0.5 s more takes 960 ticks).
    primary = make_file(ticks_per_beat=480, tracks=[[
        MetaMessage('set_tempo', tempo=500000, time=0),
        MetaMessage('set_tempo', tempo=250000, time=480),
        EOT,
    ]])
    # Secondary: 240 tpb, tempo 500000, at tick 240 (0.5 s) tempo
    # changes to 1000000 (1 s per beat). The note lands at tick 360,
    # i.e. 0.5 s + 120 ticks * 1 s / 240 ticks = 1.0 s.
    secondary = make_file(ticks_per_beat=240, tracks=[[
        MetaMessage('set_tempo', tempo=500000, time=0),
        MetaMessage('set_tempo', tempo=1000000, time=240),
        Message('note_on', channel=1, note=64, velocity=64, time=120),
        EOT,
    ]])

    merged = merge_midifiles(primary, secondary)

    # 1.0 s on the primary timeline: 0.5 s at 500000 us/beat (480
    # ticks) + 0.5 s at 250000 us/beat (960 ticks) = tick 1440.
    events = abs_times(merged.tracks[1])
    assert events[0][0] == 1440
    assert events[0][1].type == 'note_on'


def test_overlapping_notes_and_playback_time():
    primary = make_file(ticks_per_beat=480, tracks=[[
        Message('note_on', channel=0, note=60, velocity=64, time=0),
        Message('note_off', channel=0, note=60, velocity=64, time=960),
        EOT,
    ]])
    # Secondary note starts in the middle of the primary note and
    # outlives it: 240 tpb, default tempo -> tick 240 is 0.5 s,
    # tick 240 + 480 is 1.5 s.
    secondary = make_file(ticks_per_beat=240, tracks=[[
        Message('note_on', channel=1, note=64, velocity=64, time=240),
        Message('note_off', channel=1, note=64, velocity=64, time=480),
        EOT,
    ]])

    merged = merge_midifiles(primary, secondary)

    events = abs_times(merged.tracks[1])
    assert events[0][0] == 480  # 0.5 s on primary timeline
    assert events[1][0] == 1440  # 1.5 s on primary timeline

    # Playback time of the merged file matches the sum of both files.
    playback = [msg.time for msg in merged
                if not msg.is_meta or msg.type == 'set_tempo']
    assert abs(sum(playback) - 1.5) < 1e-9


def test_same_tick_order_is_stable():
    secondary = make_file(ticks_per_beat=240, tracks=[[
        Message('note_on', channel=1, note=64, velocity=64, time=0),
        Message('note_on', channel=1, note=65, velocity=64, time=0),
        Message('note_on', channel=1, note=66, velocity=64, time=0),
        EOT,
    ]])
    primary = make_file(ticks_per_beat=480, tracks=[[EOT]])

    merged = merge_midifiles(primary, secondary)

    notes = [msg.note for msg in merged.tracks[1]
             if msg.type == 'note_on']
    assert notes == [64, 65, 66]
    assert [msg.time for msg in merged.tracks[1]
            if msg.type == 'note_on'] == [0, 0, 0]


def test_tracks_stay_identifiable():
    primary = make_file(ticks_per_beat=480, tracks=[
        [MetaMessage('track_name', name='melody', time=0),
         Message('note_on', channel=0, note=60, velocity=64, time=0),
         EOT],
        [MetaMessage('track_name', name='melody2', time=0),
         Message('note_on', channel=1, note=62, velocity=64, time=0),
         EOT],
    ])
    secondary = make_file(ticks_per_beat=240, tracks=[
        [MetaMessage('track_name', name='drums', time=0),
         Message('note_on', channel=9, note=36, velocity=64, time=0),
         EOT],
    ])

    merged = merge_midifiles(primary, secondary)

    assert [track.name for track in merged.tracks] == [
        'melody', 'melody2', 'drums']
    # Primary tracks are copied verbatim.
    assert list(merged.tracks[0]) == list(primary.tracks[0])
    assert list(merged.tracks[1]) == list(primary.tracks[1])


def test_channel_map():
    primary = make_file(ticks_per_beat=480, tracks=[[
        Message('note_on', channel=0, note=60, velocity=64, time=0),
        EOT,
    ]])
    secondary = make_file(ticks_per_beat=480, tracks=[[
        Message('note_on', channel=0, note=64, velocity=64, time=0),
        Message('control_change', channel=0, control=7, value=100,
                time=0),
        EOT,
    ]])

    merged = merge_midifiles(primary, secondary, channel_map={0: 4})

    channels = [msg.channel for msg in merged.tracks[1]
                if not msg.is_meta]
    assert channels == [4, 4]
    # The primary track keeps its channel.
    assert merged.tracks[0][0].channel == 0


def test_channel_conflict_with_primary():
    primary = make_file(tracks=[[
        Message('note_on', channel=0, note=60, velocity=64, time=0),
        EOT,
    ]])
    secondary = make_file(tracks=[[
        Message('note_on', channel=0, note=64, velocity=64, time=0),
        EOT,
    ]])

    with raises(ValueError, match='channel map conflict'):
        merge_midifiles(primary, secondary)

    # Mapping onto a channel used by the primary file is a conflict
    # too.
    with raises(ValueError, match='channel map conflict'):
        merge_midifiles(primary, secondary, channel_map={0: 0})


def test_channel_map_two_sources_one_target():
    primary = make_file(tracks=[[EOT]])
    secondary = make_file(tracks=[[
        Message('note_on', channel=0, note=64, velocity=64, time=0),
        Message('note_on', channel=1, note=65, velocity=64, time=0),
        EOT,
    ]])

    with raises(ValueError, match='channel map conflict'):
        merge_midifiles(primary, secondary, channel_map={0: 5, 1: 5})


def test_invalid_channel_map_values():
    primary = make_file(tracks=[[EOT]])
    secondary = make_file(tracks=[[
        Message('note_on', channel=0, note=64, velocity=64, time=0),
        EOT,
    ]])

    for channel_map in [{0: 16}, {0: -1}, {16: 0}, {0: 1.5}]:
        with raises(ValueError, match='invalid channel'):
            merge_midifiles(primary, secondary, channel_map=channel_map)


def test_type_2_rejected():
    primary = make_file(type=2, tracks=[[EOT]])
    secondary = make_file(tracks=[[EOT]])

    with raises(TypeError, match='type 2'):
        merge_midifiles(primary, secondary)
    with raises(TypeError, match='type 2'):
        merge_midifiles(secondary, primary)


def test_invalid_ticks_per_beat():
    for ticks_per_beat in (0, -25, 1.5):
        bad = make_file(ticks_per_beat=ticks_per_beat, tracks=[[EOT]])
        good = make_file(tracks=[[EOT]])
        with raises(ValueError, match='ticks_per_beat'):
            merge_midifiles(bad, good)
        with raises(ValueError, match='ticks_per_beat'):
            merge_midifiles(good, bad)


def test_negative_delta_time_rejected():
    bad = make_file(tracks=[[
        Message('note_on', channel=0, note=64, velocity=64, time=-10),
    ]])
    good = make_file(tracks=[[EOT]])

    with raises(ValueError, match='message time'):
        merge_midifiles(bad, good)
    with raises(ValueError, match='message time'):
        merge_midifiles(good, bad)


def test_time_out_of_range():
    # A delta of MAX_TICK + 1 ticks at default tempo lands far beyond
    # the largest encodable tick on the primary timeline.
    secondary = make_file(ticks_per_beat=480, tracks=[[
        Message('note_on', channel=1, note=64, velocity=64,
                time=0x0FFFFFFF + 1),
    ]])
    primary = make_file(ticks_per_beat=480, tracks=[[EOT]])

    with raises(ValueError, match='out of range'):
        merge_midifiles(primary, secondary)


def test_input_files_not_modified():
    primary = make_file(ticks_per_beat=480, tracks=[[
        MetaMessage('set_tempo', tempo=500000, time=0),
        Message('note_on', channel=0, note=60, velocity=64, time=240),
        EOT,
    ]])
    secondary = make_file(ticks_per_beat=240, tracks=[[
        MetaMessage('set_tempo', tempo=1000000, time=0),
        Message('note_on', channel=1, note=64, velocity=64, time=120),
        EOT,
    ]])
    primary_before = [list(track) for track in primary.tracks]
    secondary_before = [list(track) for track in secondary.tracks]

    merge_midifiles(primary, secondary)

    assert [list(track) for track in primary.tracks] == primary_before
    assert [list(track) for track in secondary.tracks] == secondary_before


def test_no_negative_deltas():
    # Rounding can move events slightly; deltas must stay >= 0.
    secondary = make_file(ticks_per_beat=1000, tracks=[[
        Message('note_on', channel=1, note=64, velocity=64, time=1),
        Message('note_on', channel=1, note=65, velocity=64, time=1),
        Message('note_on', channel=1, note=66, velocity=64, time=1),
        EOT,
    ]])
    primary = make_file(ticks_per_beat=480, tracks=[[EOT]])

    merged = merge_midifiles(primary, secondary)

    for track in merged.tracks:
        for msg in track:
            assert msg.time >= 0


def test_save_and_reread():
    primary = make_file(ticks_per_beat=480, tracks=[
        [MetaMessage('set_tempo', tempo=400000, time=0),
         MetaMessage('time_signature', numerator=3, denominator=4,
                     time=0),
         Message('note_on', channel=0, note=60, velocity=64, time=0),
         Message('note_off', channel=0, note=60, velocity=64, time=480),
         EOT],
    ])
    secondary = make_file(ticks_per_beat=960, tracks=[
        [MetaMessage('track_name', name='drums', time=0),
         MetaMessage('set_tempo', tempo=800000, time=0),
         Message('note_on', channel=9, note=36, velocity=100, time=960),
         Message('note_off', channel=9, note=36, velocity=64, time=480),
         EOT],
    ])

    merged = merge_midifiles(primary, secondary)

    outfile = io.BytesIO()
    merged.save(file=outfile)
    outfile.seek(0)
    reread = MidiFile(file=outfile)

    assert reread.type == 1
    assert reread.ticks_per_beat == 480
    assert len(reread.tracks) == 2
    for merged_track, reread_track in zip(merged.tracks, reread.tracks):
        assert list(merged_track) == list(reread_track)

    # The secondary note_on (960 ticks at 960 tpb = 1 beat = 0.8 s at
    # 800000 us/beat) lands at 0.8 s on the primary timeline
    # (400000 us/beat, 480 tpb): 0.8 / (400000e-6 / 480) = 960 ticks,
    # within one tick of the ideal position.
    events = abs_times(reread.tracks[1])
    note_on = next(t for t, msg in events if msg.type == 'note_on')
    assert abs(note_on - 960) <= 1


def test_type_0_input():
    primary = make_file(type=0, ticks_per_beat=480, tracks=[[
        Message('note_on', channel=0, note=60, velocity=64, time=0),
        EOT,
    ]])
    secondary = make_file(type=0, ticks_per_beat=120, tracks=[[
        Message('note_on', channel=1, note=64, velocity=64, time=120),
        EOT,
    ]])

    merged = merge_midifiles(primary, secondary)

    assert merged.type == 1
    assert len(merged.tracks) == 2
    # 120 ticks at 120 tpb = 1 beat = 0.5 s = 480 primary ticks.
    events = abs_times(merged.tracks[1])
    assert events[0][0] == 480
