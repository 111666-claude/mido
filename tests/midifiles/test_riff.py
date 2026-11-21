# SPDX-FileCopyrightText: 2026
#
# SPDX-License-Identifier: MIT

import io
import struct

from pytest import raises

from mido.messages import Message
from mido.midifiles import RiffError
from mido.midifiles.meta import MetaMessage
from mido.midifiles.midifiles import MidiFile, MidiTrack
from mido.midifiles.riff import build_rmid, read_rmid_smf

# A fixed RMID sample with a LIST/INFO chunk and odd-byte padding.
#
# Layout (all RIFF sizes are little endian):
#
#   RIFF size=92 form=RMID
#     LIST size=18 INFO
#       INAM size=5 b'Hello' + 1 pad byte (odd-sized chunk)
#     JUNK size=3 + 1 pad byte           (unknown chunk, ignored)
#     data size=42                       (the embedded SMF, even size)
#
# The SMF is a type 1 file with one track: a track name, a note on,
# a note off and the end of track marker.
RMID_SAMPLE = """
52 49 46 46  # RIFF
5c 00 00 00  # RIFF size = 92 (little endian)
52 4d 49 44  # RMID form type
4c 49 53 54  # LIST
12 00 00 00  # LIST size = 18
49 4e 46 4f  # INFO list type
49 4e 41 4d  # INAM
05 00 00 00  # INAM size = 5
48 65 6c 6c 6f 00  # b'Hello' + odd-byte pad
4a 55 4e 4b  # JUNK (unknown chunk, must be skipped)
03 00 00 00  # JUNK size = 3
ab cd ef 00  # payload + odd-byte pad
64 61 74 61  # data
2a 00 00 00  # data size = 42
4d 54 68 64  # MThd
00 00 00 06  # SMF header size (big endian)
00 01        # type 1
00 01        # 1 track
00 78        # 120 ticks per beat
4d 54 72 6b  # MTrk
00 00 00 14  # track size = 20
00 ff 03 04 52 4d 49 44  # track_name name='RMID'
00 90 3c 40              # note_on note=60 velocity=64
78 80 3c 40              # note_off delta=120, note=60, velocity=64
00 ff 2f 00              # end_of_track
"""

EXPECTED_MESSAGES = [
    MetaMessage('track_name', name='RMID', time=0),
    Message('note_on', note=60, velocity=64, time=0),
    Message('note_off', note=60, velocity=64, time=120),
    MetaMessage('end_of_track', time=0),
]


def parse_hexdump(hexdump):
    data = bytearray()
    for line in hexdump.splitlines():
        data += bytearray.fromhex(line.split('#')[0])
    return bytes(data)


def riff_chunk(chunk_id, payload):
    """Build a RIFF chunk with little endian size and odd-byte pad."""
    out = chunk_id + struct.pack('<L', len(payload)) + payload
    if len(payload) % 2:
        out += b'\x00'
    return out


def make_rmid(chunks, form_type=b'RMID'):
    """Wrap already-built chunks in a RIFF container."""
    body = form_type + b''.join(chunks)
    return b'RIFF' + struct.pack('<L', len(body)) + body


def make_smf(track_name='RMID'):
    """Build an SMF in memory; longer names make its byte length odd."""
    mid = MidiFile(type=1, ticks_per_beat=120)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage('track_name', name=track_name, time=0))
    track.append(Message('note_on', note=60, velocity=64, time=0))
    track.append(Message('note_off', note=60, velocity=64, time=120))
    buffer = io.BytesIO()
    mid.save(file=buffer)
    return buffer.getvalue()


def open_sample():
    return MidiFile(file=io.BytesIO(parse_hexdump(RMID_SAMPLE)))


def test_read_rmid_extracts_smf():
    mid = open_sample()

    assert mid.type == 1
    assert mid.ticks_per_beat == 120
    assert len(mid.tracks) == 1
    assert list(mid.tracks[0]) == EXPECTED_MESSAGES


def test_plain_smf_is_not_treated_as_riff():
    # A normal SMF starts with MThd, not RIFF, and reads exactly as before.
    smf = make_smf()
    stream = read_rmid_smf(io.BytesIO(smf))
    # The marker is replayed so the whole SMF is still readable.
    assert stream is not None
    assert stream.read() == smf

    mid = MidiFile(file=io.BytesIO(smf))
    assert list(mid.tracks[0]) == EXPECTED_MESSAGES


def test_non_seekable_fallback():
    # A stream without seek() with plain SMF data still works.
    smf = make_smf()

    class NoSeek:
        def __init__(self, data):
            self._buffer = io.BytesIO(data)

        def read(self, size=-1):
            return self._buffer.read(size)

        def tell(self):
            return self._buffer.tell()

    mid = MidiFile(file=NoSeek(smf))
    assert list(mid.tracks[0]) == EXPECTED_MESSAGES


def test_save_and_read_back_roundtrip():
    smf = make_smf()
    original = MidiFile(file=io.BytesIO(smf))

    buffer = io.BytesIO()
    original.save(file=buffer, rmid=True)
    raw = buffer.getvalue()

    # Correct RIFF header, little endian sizes and data wrapper.
    assert raw[:4] == b'RIFF'
    riff_size = struct.unpack('<L', raw[4:8])[0]
    assert riff_size == len(raw) - 8
    assert raw[8:12] == b'RMID'
    assert raw[12:16] == b'data'
    data_size = struct.unpack('<L', raw[16:20])[0]
    assert data_size == len(smf)
    assert raw[20:20 + len(smf)] == smf

    reread = MidiFile(file=io.BytesIO(raw))
    assert reread.type == original.type
    assert reread.ticks_per_beat == original.ticks_per_beat
    assert reread.tracks == original.tracks


def test_odd_sized_smf_gets_pad_byte():
    # A five-character track name makes the SMF an odd number of bytes.
    smf = make_smf(track_name='RMIDx')
    assert len(smf) % 2 == 1

    rmid = build_rmid(smf)
    assert len(rmid) % 2 == 0  # the pad byte keeps the RIFF file even
    riff_size = struct.unpack('<L', rmid[4:8])[0]
    assert riff_size == len(rmid) - 8
    data_size = struct.unpack('<L', rmid[16:20])[0]
    assert data_size == len(smf)
    assert rmid[-1] == 0  # pad byte after the odd-sized data chunk

    reread = MidiFile(file=io.BytesIO(rmid))
    assert list(reread.tracks[0]) == [
        MetaMessage('track_name', name='RMIDx', time=0),
        Message('note_on', note=60, velocity=64, time=0),
        Message('note_off', note=60, velocity=64, time=120),
        MetaMessage('end_of_track', time=0),
    ]


def test_default_save_is_plain_smf():
    mid = MidiFile(file=io.BytesIO(make_smf()))
    buffer = io.BytesIO()
    mid.save(file=buffer)
    assert buffer.getvalue()[:4] == b'MThd'


def test_save_to_filename(tmp_path):
    mid = MidiFile(file=io.BytesIO(make_smf()))
    path = tmp_path / 'song.rmi'
    mid.save(str(path), rmid=True)

    reread = MidiFile(str(path))
    assert reread.tracks == mid.tracks


def test_bad_form_type():
    raw = make_rmid([riff_chunk(b'data', make_smf())], form_type=b'WAVE')
    with raises(RiffError) as info:
        MidiFile(file=io.BytesIO(raw))
    assert info.value.offset == 8


def test_truncated_riff_size_field():
    raw = b'RIFF\x5c'  # only one byte of the 4-byte size field
    with raises(RiffError) as info:
        MidiFile(file=io.BytesIO(raw))
    assert info.value.offset == 5


def test_truncated_form_type():
    raw = b'RIFF' + struct.pack('<L', 4) + b'RM'
    with raises(RiffError) as info:
        MidiFile(file=io.BytesIO(raw))
    assert info.value.offset == 10


def test_missing_data_chunk():
    # Only a LIST chunk, no data chunk.
    list_chunk = riff_chunk(b'LIST', b'INFO' + riff_chunk(b'INAM', b'Hello'))
    raw = make_rmid([list_chunk])
    with raises(RiffError) as info:
        MidiFile(file=io.BytesIO(raw))
    assert 'no data chunk' in str(info.value)
    assert info.value.offset == len(raw)


def test_multiple_data_chunks():
    smf = make_smf()
    raw = make_rmid([
        riff_chunk(b'data', smf),
        riff_chunk(b'data', smf),
    ])
    with raises(RiffError) as info:
        MidiFile(file=io.BytesIO(raw))
    assert 'multiple data chunks' in str(info.value)
    # The second data chunk header starts right after the first chunk.
    assert info.value.offset == 12 + 8 + len(smf)


def test_data_chunk_size_past_riff_end():
    smf = make_smf()
    # Claim a size larger than the actual container.
    raw = (b'RIFF' + struct.pack('<L', 12 + 8 + len(smf))
           + b'RMID' + b'data'
           + struct.pack('<L', len(smf) + 100) + smf)
    with raises(RiffError) as info:
        MidiFile(file=io.BytesIO(raw))
    assert 'extends past end' in str(info.value)
    assert info.value.offset == 12  # position of the offending chunk header


def test_truncated_data_chunk_payload():
    smf = make_smf()
    raw = (b'RIFF' + struct.pack('<L', 12 + 8 + len(smf))
           + b'RMID' + b'data'
           + struct.pack('<L', len(smf)) + smf[:-4])
    with raises(RiffError) as info:
        MidiFile(file=io.BytesIO(raw))
    assert 'truncated' in str(info.value)
    assert info.value.offset == len(raw)


def test_nested_chunk_size_past_list_end():
    # LIST claims 18 bytes but its INAM subchunk claims 30.
    nested = b'INAM' + struct.pack('<L', 30) + b'Hello'
    list_chunk = b'LIST' + struct.pack('<L', 18) + b'INFO' + nested
    raw = make_rmid([list_chunk, riff_chunk(b'data', make_smf())])
    with raises(RiffError) as info:
        MidiFile(file=io.BytesIO(raw))
    assert 'extends past end' in str(info.value)
    # LIST header at 12, its type takes 4, nested header starts at 24.
    assert info.value.offset == 24


def test_missing_odd_byte_pad():
    # An odd JUNK chunk with no pad byte before the next chunk.
    junk = b'JUNK' + struct.pack('<L', 3) + b'\xab\xcd\xef'
    data = riff_chunk(b'data', make_smf())
    raw = make_rmid([junk, data])
    # The RIFF size itself is now inconsistent with its contents, which
    # must be reported at a definite position rather than misparsed.
    with raises(RiffError):
        MidiFile(file=io.BytesIO(raw))


def test_data_inside_list_is_not_the_smf():
    # A data chunk nested in LIST must not be used as the MIDI stream.
    smf = make_smf()
    inner = riff_chunk(b'data', smf)
    list_chunk = riff_chunk(b'LIST', b'INFO' + inner)
    raw = make_rmid([list_chunk])
    with raises(RiffError) as info:
        MidiFile(file=io.BytesIO(raw))
    assert 'no data chunk' in str(info.value)


def test_embedded_data_is_still_validated_as_smf():
    # RIFF structure is fine but the payload is not an SMF.
    raw = make_rmid([riff_chunk(b'data', b'not a midi file at all')])
    with raises(OSError, match='MThd not found'):
        MidiFile(file=io.BytesIO(raw))


def test_riff_size_bounds_the_parse():
    # The declared RIFF size ends after LIST, so the trailing data chunk
    # must not be considered; a missing-data error is raised instead.
    list_chunk = riff_chunk(b'LIST', b'INFO' + riff_chunk(b'INAM', b'Hi'))
    data = riff_chunk(b'data', make_smf())
    declared_body = b'RMID' + list_chunk
    raw = (b'RIFF' + struct.pack('<L', len(declared_body))
           + declared_body + data)
    with raises(RiffError, match='no data chunk'):
        MidiFile(file=io.BytesIO(raw))
