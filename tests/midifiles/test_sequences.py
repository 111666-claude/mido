# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

import io

from pytest import approx, raises

import mido
from mido.messages import Message
from mido.midifiles import (
    MAX_TICK,
    MidiFile,
    MidiTrack,
    SequenceEvent,
    SequenceTimeOverflow,
    TrackSequence,
    get_track_sequence,
)
from mido.midifiles.meta import MetaMessage
from mido.midifiles.sequences import DEFAULT_TEMPO

# A type 2 file with two independent patterns and two different tempos.
#
# Track 0 (480 ticks per beat, tempo 500000 then 1000000):
#   tick   0: track_name 'slow', set_tempo 500000, note_on 60 and a
#             sysex that share the tick with them (order must be kept)
#   tick 480: note_off 60 (0.5 s)
#   tick 960: set_tempo 1000000 (1.0 s)
#   tick 1440: end_of_track (2.0 s)
#
# Track 1 (tempo 250000):
#   tick   0: set_tempo 250000, note_on 67
#   tick 240: text 'fast', note_off 67 (0.125 s)
#   tick 480: end_of_track (0.25 s)
TYPE2_HEXDUMP = """
4d 54 68 64 00 00 00 06  # MThd
00 02                    # type 2
00 02                    # 2 tracks
01 e0                    # 480 ticks per beat
4d 54 72 6b 00 00 00 2b  # MTrk, 43 bytes
00 ff 03 04 73 6c 6f 77  # track_name 'slow'
00 ff 51 03 07 a1 20     # set_tempo 500000
00 90 3c 40              # note_on 60 velocity 64
00 f0 03 01 02 f7        # sysex [1, 2]
83 60 80 3c 40           # note_off 60 after 480 ticks
83 60 ff 51 03 0f 42 40  # set_tempo 1000000 after 480 ticks
83 60 ff 2f 00           # end_of_track after 480 ticks
4d 54 72 6b 00 00 00 1d  # MTrk, 29 bytes
00 ff 51 03 03 d0 90     # set_tempo 250000
00 90 43 64              # note_on 67 velocity 100
81 70 ff 01 04 66 61 73 74  # text 'fast' after 240 ticks
00 80 43 40              # note_off 67
81 70 ff 2f 00           # end_of_track after 240 ticks
"""


def parse_hexdump(hexdump):
    data = bytearray()
    for line in hexdump.splitlines():
        data += bytearray.fromhex(line.split('#')[0])
    return data


TYPE2_BYTES = bytes(parse_hexdump(TYPE2_HEXDUMP))


def read_fixture():
    return MidiFile(file=io.BytesIO(TYPE2_BYTES))


def make_type2(*tracks, ticks_per_beat=480):
    return MidiFile(type=2, ticks_per_beat=ticks_per_beat,
                    tracks=list(tracks))


def test_two_patterns_have_independent_lengths():
    mid = read_fixture()

    slow = mid.track_sequence(0)
    fast = mid.track_sequence(1)

    assert slow.track == 0
    assert fast.track == 1
    assert isinstance(slow, TrackSequence)

    # Track 0: 480 ticks at 0.5 s/beat, then 960 ticks at 1.0 s/beat.
    assert slow.length_ticks == 1440
    assert slow.length == approx(2.0)
    # Track 1 uses its own 250000 tempo: 480 ticks at 0.25 s/beat.
    assert fast.length_ticks == 480
    assert fast.length == approx(0.25)

    # The file-level length is still unavailable for type 2.
    with raises(ValueError):
        _ = mid.length


def test_absolute_ticks_and_seconds_per_pattern():
    slow = read_fixture().track_sequence(0)
    assert [event.tick for event in slow] == [0, 0, 0, 0, 480, 960, 1440]
    assert [event.seconds for event in slow] == [
        approx(0.0), approx(0.0), approx(0.0), approx(0.0),
        approx(0.5), approx(1.0), approx(2.0),
    ]

    fast = read_fixture().track_sequence(1)
    assert [event.tick for event in fast] == [0, 0, 240, 240, 480]
    assert [event.seconds for event in fast] == [
        approx(0.0), approx(0.0), approx(0.125), approx(0.125),
        approx(0.25),
    ]


def test_event_sequence_keeps_same_tick_order_meta_and_sysex():
    slow = read_fixture().track_sequence(0)
    assert [event.type for event in slow] == [
        'track_name',
        'set_tempo',
        'note_on',
        'sysex',
        'note_off',
        'set_tempo',
        'end_of_track',
    ]
    assert [event.index for event in slow] == list(range(len(slow)))

    sysex = slow[3]
    assert sysex.is_sysex
    assert sysex.message.data == (1, 2)
    assert sysex.tick == 0
    assert slow[0].is_meta
    assert slow[2].message == Message('note_on', note=60, velocity=64,
                                      time=0)
    # The message copy still carries the original delta time in ticks.
    assert slow[4].message.time == 480

    fast = read_fixture().track_sequence(1)
    assert [event.type for event in fast] == [
        'set_tempo', 'note_on', 'text', 'note_off', 'end_of_track',
    ]


def test_tempo_change_only_applies_after_its_own_tick():
    # The set_tempo at tick 960 is timestamped with the previous tempo
    # (1.0 s) and only the following ticks use the new one.
    slow = read_fixture().track_sequence(0)
    assert slow[5].type == 'set_tempo'
    assert slow[5].message.tempo == 1000000
    assert slow[5].seconds == approx(1.0)
    assert slow.length == approx(2.0)


def test_tracks_do_not_share_tempo_maps():
    # If the tempos leaked between tracks, track 0 would pick up the
    # 250000 tempo (or track 1 the 1000000 one) and the durations
    # would differ. Both are computed in isolation.
    mid = read_fixture()
    assert mid.track_sequence(0).length == approx(2.0)
    assert mid.track_sequence(1).length == approx(0.25)

    # Reordering the tracks must not change the per-pattern durations.
    reversed_mid = MidiFile(type=2, ticks_per_beat=480,
                            tracks=list(reversed(mid.tracks)))
    assert reversed_mid.track_sequence(0).length == approx(0.25)
    assert reversed_mid.track_sequence(1).length == approx(2.0)


def test_default_tempo_before_first_set_tempo():
    track = MidiTrack([
        Message('note_on', note=60, time=480),
        MetaMessage('set_tempo', tempo=250000, time=480),
        Message('note_off', note=60, time=480),
    ])
    sequence = make_type2(track).track_sequence(0)
    # 480 ticks at the default tempo is 0.5 s.
    assert sequence[0].seconds == approx(0.5)
    # 480 more ticks at the default tempo, then the change takes effect.
    assert sequence[1].seconds == approx(1.0)
    assert sequence[2].seconds == approx(1.25)
    assert sequence.length == approx(1.25)
    assert DEFAULT_TEMPO == 500000


def test_module_function_and_method_agree():
    mid = read_fixture()
    assert get_track_sequence(mid, 1) == mid.track_sequence(1)


def test_empty_track():
    mid = make_type2(MidiTrack())
    sequence = mid.track_sequence(0)
    assert isinstance(sequence, TrackSequence)
    assert sequence.events == []
    assert len(sequence) == 0
    assert sequence.length_ticks == 0
    assert sequence.length == 0.0


def test_invalid_track_index():
    mid = read_fixture()
    with raises(IndexError):
        mid.track_sequence(2)
    with raises(IndexError):
        mid.track_sequence(-1)
    with raises(TypeError):
        mid.track_sequence(0.0)
    with raises(TypeError):
        mid.track_sequence('0')
    with raises(TypeError):
        mid.track_sequence(True)


def test_type_0_and_type_1_are_rejected():
    for file_type in (0, 1):
        mid = MidiFile(type=file_type)
        mid.tracks.append(MidiTrack())
        with raises(TypeError, match='type 2'):
            mid.track_sequence(0)
        with raises(TypeError, match='type 2'):
            get_track_sequence(mid, 0)


def test_type_check_happens_before_index_check():
    # Even an invalid index gives the type error for a type 1 file.
    mid = MidiFile(type=1, tracks=[MidiTrack()])
    with raises(TypeError, match='type 2'):
        mid.track_sequence(99)


def test_invalid_tempo():
    track = MidiTrack([MetaMessage('set_tempo', tempo=500000)])
    mid = make_type2(track)
    vars(track[0])['tempo'] = 0
    with raises(ValueError, match='invalid tempo'):
        mid.track_sequence(0)

    track[0] = MetaMessage('set_tempo', tempo=500000)
    vars(track[0])['tempo'] = -1
    with raises(ValueError, match='invalid tempo'):
        mid.track_sequence(0)

    track[0] = MetaMessage('set_tempo', tempo=500000)
    vars(track[0])['tempo'] = 500000.5
    with raises(TypeError, match='tempo'):
        mid.track_sequence(0)


def test_invalid_delta_times():
    mid = make_type2(MidiTrack([Message('note_on', note=60, time=1.5)]))
    with raises(ValueError, match='non-integer delta'):
        mid.track_sequence(0)

    mid = make_type2(MidiTrack([Message('note_on', note=60, time=-1)]))
    with raises(ValueError, match='negative delta'):
        mid.track_sequence(0)


def test_tick_overflow():
    # A single delta time that does not fit in a MIDI variable-length
    # quantity overflows.
    mid = make_type2(MidiTrack([
        Message('note_on', note=60, time=MAX_TICK + 1),
    ]))
    with raises(SequenceTimeOverflow):
        mid.track_sequence(0)

    # The accumulated absolute tick overflows even though every delta
    # on its own fits.
    mid = make_type2(MidiTrack([
        Message('note_on', note=60, time=MAX_TICK),
        Message('note_off', note=60, time=1),
    ]))
    with raises(SequenceTimeOverflow):
        mid.track_sequence(0)

    # SequenceTimeOverflow can be caught as either exception type.
    with raises(OverflowError):
        mid.track_sequence(0)
    with raises(ValueError):
        mid.track_sequence(0)


def test_maximum_tick_is_allowed():
    mid = make_type2(MidiTrack([
        Message('note_on', note=60, time=MAX_TICK),
    ]))
    sequence = mid.track_sequence(0)
    assert sequence.length_ticks == MAX_TICK


def test_invalid_time_base():
    mid = make_type2(MidiTrack(), ticks_per_beat=0)
    with raises(ValueError, match='time division'):
        mid.track_sequence(0)

    mid = make_type2(MidiTrack(), ticks_per_beat=-480)
    with raises(ValueError, match='time division'):
        mid.track_sequence(0)

    mid = make_type2(MidiTrack(), ticks_per_beat=480.0)
    with raises(TypeError, match='ticks_per_beat'):
        mid.track_sequence(0)


def test_file_and_original_midi_are_not_modified():
    mid = read_fixture()
    before = [msg.dict() for track in mid.tracks for msg in track]

    slow = mid.track_sequence(0)
    fast = mid.track_sequence(1)

    # Event messages are copies: changing them does not touch the file.
    slow[0].message.name = 'changed'
    slow[2].message.note = 1
    assert mid.tracks[0][0].name == 'slow'
    assert mid.tracks[0][2].note == 60

    after = [msg.dict() for track in mid.tracks for msg in track]
    assert before == after

    # Saving after building the sequences produces the original bytes.
    outfile = io.BytesIO()
    mid.save(file=outfile)
    assert outfile.getvalue() == TYPE2_BYTES

    # The sequences themselves are unaffected by the original tracks.
    assert fast[1].message.note == 67


def test_sequence_events_are_copies_of_unknown_meta_and_sysex():
    mid = read_fixture()
    sequence = mid.track_sequence(0)
    assert isinstance(sequence[3], SequenceEvent)
    assert sequence.messages[3] is not mid.tracks[0][3]
    assert sequence[3].message == mid.tracks[0][3]


def test_read_fixture_still_parses_to_the_original_tracks():
    mid = read_fixture()
    assert mid.type == 2
    assert mid.ticks_per_beat == 480
    assert mid.tracks[0].name == 'slow'
    assert mid.tracks[0][2] == Message('note_on', note=60, velocity=64,
                                       time=0)
    assert mid.tracks[1][2] == MetaMessage('text', text='fast', time=240)


def test_exported_from_package():
    assert mido.TrackSequence is TrackSequence
    assert mido.SequenceEvent is SequenceEvent
    assert mido.SequenceTimeOverflow is SequenceTimeOverflow
    assert mido.get_track_sequence is get_track_sequence
    assert mido.MAX_TICK == MAX_TICK
