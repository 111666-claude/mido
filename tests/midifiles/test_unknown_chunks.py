# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

import io
import struct

from pytest import raises

from mido.messages import Message
from mido.midifiles.meta import MetaMessage
from mido.midifiles.midifiles import (
    MAX_UNKNOWN_CHUNK_LENGTH,
    MidiFile,
    MidiTrack,
    UnknownChunk,
)


def chunk(name, data):
    """Build a raw IFF-style chunk: 4-byte name, length, payload."""
    return name + struct.pack('>L', len(data)) + data


def header(num_tracks, type=1, ticks_per_beat=120):
    return (b'MThd'
            + struct.pack('>L', 6)
            + struct.pack('>hhh', type, num_tracks, ticks_per_beat))


# Two canonical tracks whose encoding mido reproduces byte for byte.
TRACK_0 = bytes.fromhex(
    '00 90 40 40'          # note_on note=64 vel=64, delta 0
    '00 80 40 40'          # note_off note=64 vel=64, delta 0
    '00 ff 2f 00'          # end_of_track
)
TRACK_1 = bytes.fromhex(
    '00 90 47 10'          # note_on note=71 vel=16, delta 0
    '40 80 47 10'          # note_off note=71 vel=16, delta 64
    '00 ff 2f 00'          # end_of_track
)

PRE_DATA = b'project notes before the first track'
MID_DATA = b'private block between two tracks'
TAIL_DATA = b'trailing block after every track'


def build_interspersed_file():
    """File with unknown chunks before, between and after the tracks."""
    return (
        header(num_tracks=2)
        + chunk(b'PRE ', PRE_DATA)
        + chunk(b'MTrk', TRACK_0)
        + chunk(b'MID!', MID_DATA)
        + chunk(b'MTrk', TRACK_1)
        + chunk(b'TAIL', TAIL_DATA)
    )


def parse(data, **kwargs):
    return MidiFile(file=io.BytesIO(data), **kwargs)


def save(mid, keep_unknown_chunks=None):
    out = io.BytesIO()
    mid.save(file=out, keep_unknown_chunks=keep_unknown_chunks)
    return out.getvalue()


def test_unknown_chunks_are_recorded_with_position():
    mid = parse(build_interspersed_file())

    # The track count still comes from the MThd header.
    assert len(mid.tracks) == 2

    assert mid.unknown_chunks == [
        UnknownChunk(b'PRE ', PRE_DATA, track=0),
        UnknownChunk(b'MID!', MID_DATA, track=1),
        UnknownChunk(b'TAIL', TAIL_DATA, track=2),
    ]


def test_tracks_are_still_parsed_as_events():
    mid = parse(build_interspersed_file())

    assert mid.tracks[0] == [
        Message('note_on', note=64, velocity=64, time=0),
        Message('note_off', note=64, velocity=64, time=0),
        MetaMessage('end_of_track', time=0),
    ]
    assert mid.tracks[1] == [
        Message('note_on', note=71, velocity=16, time=0),
        Message('note_off', note=71, velocity=16, time=64),
        MetaMessage('end_of_track', time=0),
    ]


def test_byte_exact_round_trip():
    original = build_interspersed_file()
    mid = parse(original)

    # The whole file, chunks and tracks alike, is written back verbatim.
    assert save(mid) == original


def test_round_trip_survives_a_second_parse():
    mid = parse(save(parse(build_interspersed_file())))

    assert len(mid.tracks) == 2
    assert mid.unknown_chunks == [
        UnknownChunk(b'PRE ', PRE_DATA, track=0),
        UnknownChunk(b'MID!', MID_DATA, track=1),
        UnknownChunk(b'TAIL', TAIL_DATA, track=2),
    ]
    assert mid.tracks[1][1] == Message(
        'note_off', note=71, velocity=16, time=64)


def test_file_without_unknown_chunks_keeps_legacy_output():
    original = header(num_tracks=2) + chunk(b'MTrk', TRACK_0) \
        + chunk(b'MTrk', TRACK_1)

    mid = parse(original)
    assert mid.unknown_chunks == []
    assert save(mid) == original


def test_unknown_chunks_can_be_dropped_on_save():
    mid = parse(build_interspersed_file())

    without_chunks = (
        header(num_tracks=2)
        + chunk(b'MTrk', TRACK_0)
        + chunk(b'MTrk', TRACK_1)
    )
    assert save(mid, keep_unknown_chunks=False) == without_chunks


def test_constructor_flag_controls_default_save_behavior():
    mid = parse(build_interspersed_file(), keep_unknown_chunks=False)

    # Chunks are still read...
    assert len(mid.unknown_chunks) == 3

    # ...but dropped by default when the flag was set at load time.
    assert save(mid) == (
        header(num_tracks=2)
        + chunk(b'MTrk', TRACK_0)
        + chunk(b'MTrk', TRACK_1)
    )


def test_unknown_chunks_can_be_built_programmatically():
    mid = MidiFile(
        type=1,
        ticks_per_beat=120,
        tracks=[MidiTrack([Message('note_on', note=64, velocity=64)])],
        unknown_chunks=[
            UnknownChunk(b'NOTE', b'before', track=0),
            UnknownChunk(b'END!', b'after', track=1),
        ],
    )

    round_tripped = parse(save(mid))
    assert round_tripped.unknown_chunks == [
        UnknownChunk(b'NOTE', b'before', track=0),
        UnknownChunk(b'END!', b'after', track=1),
    ]
    assert round_tripped.tracks[0][0] == Message(
        'note_on', note=64, velocity=64, time=0)


def test_duplicate_identifier_is_rejected():
    corrupt = (
        header(num_tracks=1)
        + chunk(b'NOTE', b'first')
        + chunk(b'NOTE', b'second')
        + chunk(b'MTrk', TRACK_0)
    )

    with raises(OSError, match='duplicate chunk identifier'):
        parse(corrupt)


def test_reserved_chunk_name_in_private_position_is_rejected():
    corrupt = (
        header(num_tracks=1)
        + chunk(b'MThd', b'\x00\x01\x00\x01\x00\x78')
        + chunk(b'MTrk', TRACK_0)
    )

    with raises(OSError, match='unexpected'):
        parse(corrupt)


def test_extra_track_beyond_header_count_is_rejected():
    corrupt = (
        header(num_tracks=1)
        + chunk(b'MTrk', TRACK_0)
        + chunk(b'MTrk', TRACK_1)
    )

    with raises(OSError, match='unexpected MTrk'):
        parse(corrupt)


def test_truncated_chunk_header_has_context():
    corrupt = header(num_tracks=1) + chunk(b'MTrk', TRACK_0) + b'\x01\x02\x03'

    with raises(EOFError, match='truncated chunk header'):
        parse(corrupt)


def test_truncated_chunk_payload_has_context():
    # The short chunk must be at the end of the file so the payload
    # read actually runs into the end of stream instead of swallowing
    # the following chunk.
    corrupt = (
        header(num_tracks=1)
        + chunk(b'MTrk', TRACK_0)
        + b'NOTE' + struct.pack('>L', 10) + b'abc'
    )

    with raises(EOFError, match='truncated unknown chunk'):
        parse(corrupt)


def test_forged_chunk_length_is_rejected():
    corrupt = (
        header(num_tracks=1)
        + b'NOTE' + struct.pack('>L', 0xFFFFFFFF)
        + chunk(b'MTrk', TRACK_0)
    )

    with raises(OSError, match='maximum'):
        parse(corrupt)


def test_max_length_bound_is_enforced_before_reading():
    # A payload exactly at the limit header, with no data attached,
    # must fail on truncation rather than allocation; one byte over the
    # limit fails on the bound itself.
    over = (
        header(num_tracks=1)
        + b'NOTE' + struct.pack('>L', MAX_UNKNOWN_CHUNK_LENGTH + 1)
    )

    with raises(OSError, match=str(MAX_UNKNOWN_CHUNK_LENGTH)):
        parse(over)


def test_truncated_file_header_has_context():
    corrupt = b'MThd' + struct.pack('>L', 6) + b'\x00\x01'

    with raises(EOFError, match='truncated MThd'):
        parse(corrupt)


def test_missing_track_when_file_ends_has_context():
    with raises(EOFError, match='declares 1 MTrk'):
        parse(header(num_tracks=1))


def test_unknown_chunk_validation():
    with raises(ValueError):
        UnknownChunk(b'abc', b'')          # not four bytes
    with raises(ValueError):
        UnknownChunk(b'MTrk', b'')         # reserved name
    with raises(TypeError):
        UnknownChunk(b'NOPE', 'not bytes')
    with raises(ValueError):
        UnknownChunk(b'NOPE', b'', -1)     # negative position


def test_save_rejects_duplicate_identifiers():
    mid = MidiFile(
        type=1,
        tracks=[MidiTrack([Message('note_on', note=64, velocity=64)])],
        unknown_chunks=[
            UnknownChunk(b'NOTE', b'a', track=0),
            UnknownChunk(b'NOTE', b'b', track=1),
        ],
    )

    with raises(ValueError, match='duplicate chunk identifier'):
        save(mid)


def test_save_rejects_position_beyond_last_track():
    mid = MidiFile(
        type=1,
        tracks=[MidiTrack([Message('note_on', note=64, velocity=64)])],
        unknown_chunks=[UnknownChunk(b'NOTE', b'a', track=5)],
    )

    with raises(ValueError, match='only has 1 tracks'):
        save(mid)
