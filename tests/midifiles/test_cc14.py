# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

import io

from pytest import raises

from mido import Message, MetaMessage, MidiFile, read_cc14, write_cc14

# Fixed test file: type 1, two tracks, two channels, 480 ticks per beat.
#
# Track 0 (channel 0):
#   tick 0:   CC 1 = 63 (duplicate, superseded by the next message)
#   tick 0:   CC 1 = 64 (MSB)
#   tick 0:   CC 33 = 2 (LSB) -> (0, 1) = 64 * 128 + 2 = 8194
#   tick 96:  note_on 60
#   tick 96:  CC 7 = 100 (not part of any rewritten pair)
#   tick 192: CC 1 = 65 (MSB)
#   tick 192: note_off 60
#
# Track 1 (channel 1, plus a cross-track conflict on channel 0):
#   tick 0:   CC 33 = 3 (LSB only, MSB defaults to 0) -> (1, 1) = 3
#   tick 0:   CC 1 = 10 (MSB) -> (1, 1) = 10 * 128 + 3 = 1283
#   tick 48:  CC 33 = 5 -> (1, 1) = 1285
#   tick 48:  CC 0 = 1, CC 32 = 1 (bank select: an ordinary pair)
#             -> (1, 0) = 129
#   tick 192: CC 33 = 9 on channel 0. Same tick as the MSB in track
#             0, and the later track wins: (0, 1) = 65 * 128 + 9 =
#             8329.
TWO_TRACK_FILE = """
4d 54 68 64  # MThd
00 00 00 06  # chunk size
00 01        # type 1
00 02        # 2 tracks
01 e0        # 480 ticks per beat

4d 54 72 6b  # MTrk (track 0)
00 00 00 20  # chunk size (32 bytes)
00 b0 01 3f  # CC ch0 control 1 = 63
00 b0 01 40  # CC ch0 control 1 = 64
00 b0 21 02  # CC ch0 control 33 = 2
60 90 3c 64  # note_on ch0 note 60 vel 100
00 b0 07 64  # CC ch0 control 7 = 100
60 b0 01 41  # CC ch0 control 1 = 65
00 80 3c 00  # note_off ch0 note 60
00 ff 2f 00  # end_of_track

4d 54 72 6b  # MTrk (track 1)
00 00 00 1d  # chunk size (29 bytes)
00 b1 21 03  # CC ch1 control 33 = 3
00 b1 01 0a  # CC ch1 control 1 = 10
30 b1 21 05  # CC ch1 control 33 = 5
00 b1 00 01  # CC ch1 control 0 = 1 (bank select MSB)
00 b1 20 01  # CC ch1 control 32 = 1 (bank select LSB)
81 10 b0 21 09  # CC ch0 control 33 = 9 (delta 144 as variable int)
00 ff 2f 00  # end_of_track
"""

EXPECTED_CURVES = {
    (0, 1): [(0, 8194), (192, 8329)],
    (0, 7): [(96, 12800)],
    (1, 0): [(48, 129)],
    (1, 1): [(0, 1283), (48, 1285)],
}


def parse_hexdump(hexdump):
    data = bytearray()
    for line in hexdump.splitlines():
        data += bytearray.fromhex(line.split('#')[0])
    return data


def read_test_file():
    return MidiFile(file=io.BytesIO(parse_hexdump(TWO_TRACK_FILE)))


def save_and_reread(mid):
    outfile = io.BytesIO()
    mid.save(file=outfile)
    outfile.seek(0)
    return MidiFile(file=outfile)


def test_read_curves():
    assert read_cc14(read_test_file()) == EXPECTED_CURVES


def test_read_does_not_modify_file():
    mid = read_test_file()
    read_cc14(mid)
    assert read_cc14(save_and_reread(mid)) == EXPECTED_CURVES


def test_write_and_reread():
    mid = read_test_file()
    curves = read_cc14(mid)

    # Modify only the (channel 0, control 1) curve: change one value
    # and add a point at the end of the file.
    new_curve = [(0, 8194), (192, 8329), (480, 16383)]
    curves[(0, 1)] = new_curve
    write_cc14(mid, {(0, 1): curves[(0, 1)]})

    # The in-memory file shows the new curve; everything else is
    # unchanged.
    assert read_cc14(mid) == {**EXPECTED_CURVES, (0, 1): new_curve}

    # Save and reload: the curve survives the round trip through an
    # actual MIDI file.
    reloaded = save_and_reread(mid)
    assert read_cc14(reloaded) == {**EXPECTED_CURVES, (0, 1): new_curve}

    # Track 0 keeps its notes and the unrelated CC 7 with their
    # original ticks and same-tick order. The rewritten pair comes
    # back as MSB immediately followed by LSB, inserted after the
    # pre-existing messages at the same tick.
    assert list(reloaded.tracks[0]) == [
        Message('control_change', channel=0, control=1, value=64, time=0),
        Message('control_change', channel=0, control=33, value=2, time=0),
        Message('note_on', channel=0, note=60, velocity=100, time=96),
        Message('control_change', channel=0, control=7, value=100, time=0),
        Message('note_off', channel=0, note=60, velocity=0, time=96),
        Message('control_change', channel=0, control=1, value=65, time=0),
        Message('control_change', channel=0, control=33, value=9, time=0),
        Message('control_change', channel=0, control=1, value=127, time=288),
        Message('control_change', channel=0, control=33, value=127, time=0),
        MetaMessage('end_of_track', time=0),
    ]

    # Track 1 is untouched except that the conflicting channel 0
    # message (CC 33 at tick 192) moved to track 0.
    assert list(reloaded.tracks[1]) == [
        Message('control_change', channel=1, control=33, value=3, time=0),
        Message('control_change', channel=1, control=1, value=10, time=0),
        Message('control_change', channel=1, control=33, value=5, time=48),
        Message('control_change', channel=1, control=0, value=1, time=0),
        Message('control_change', channel=1, control=32, value=1, time=0),
        MetaMessage('end_of_track', time=144),
    ]


def test_write_empty_curve_removes_pair():
    mid = read_test_file()
    write_cc14(mid, {(0, 1): []})

    curves = read_cc14(mid)
    assert (0, 1) not in curves
    assert curves == {key: curve for key, curve in EXPECTED_CURVES.items()
                      if key != (0, 1)}

    # The pair is gone from both tracks, everything else stays.
    for track in mid.tracks:
        for msg in track:
            assert not (msg.type == 'control_change'
                        and msg.channel == 0
                        and msg.control in (1, 33))
    assert list(mid.tracks[0]) == [
        Message('note_on', channel=0, note=60, velocity=100, time=96),
        Message('control_change', channel=0, control=7, value=100, time=0),
        Message('note_off', channel=0, note=60, velocity=0, time=96),
        MetaMessage('end_of_track', time=0),
    ]


def test_write_new_pair_goes_to_first_track():
    mid = read_test_file()
    write_cc14(mid, {(2, 5): [(24, 1000)]})

    assert read_cc14(mid)[(2, 5)] == [(24, 1000)]
    # The new pair lands in track 0 at tick 24, after the three
    # pre-existing tick 0 messages.
    assert list(mid.tracks[0])[3:5] == [
        Message('control_change', channel=2, control=5, value=7, time=24),
        Message('control_change', channel=2, control=37, value=104, time=0),
    ]
    # Nothing was added to track 1.
    assert len(mid.tracks[1]) == 7


def test_write_duplicate_ticks_last_wins():
    mid = read_test_file()
    write_cc14(mid, {(0, 1): [(0, 100), (0, 200), (48, 300)]})
    assert read_cc14(mid)[(0, 1)] == [(0, 200), (48, 300)]


def test_write_validates_before_modifying():
    mid = read_test_file()
    before = [list(track) for track in mid.tracks]

    bad_curves = [
        {(0, 1): [(0, 16384)]},          # value too large
        {(0, 1): [(0, -1)]},             # negative value
        {(0, 1): [(-1, 0)]},             # negative tick
        {(0, 32): [(0, 0)]},             # control must be the MSB (0..31)
        {(16, 1): [(0, 0)]},             # channel out of range
        {(0, 1): [(0, 0), (0, 16385)]},  # invalid even as a duplicate
    ]
    for curves in bad_curves:
        with raises(ValueError):
            write_cc14(mid, curves)

    # A failed write leaves the file untouched.
    assert [list(track) for track in mid.tracks] == before
    assert read_cc14(mid) == EXPECTED_CURVES
