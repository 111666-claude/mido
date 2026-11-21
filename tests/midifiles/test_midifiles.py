# SPDX-FileCopyrightText: 2017 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

import io

from pytest import raises

from mido.messages import Message
from mido.midifiles.meta import KeySignatureError, MetaMessage
from mido.midifiles.midifiles import MidiFile, MidiTrack

HEADER_ONE_TRACK = """
4d 54 68 64  # MThd
00 00 00 06  # Chunk size
      00 01  # Type 1
      00 01  # 1 track
      00 78  # 120 ticks per beat
"""


def parse_hexdump(hexdump):
    data = bytearray()
    for line in hexdump.splitlines():
        data += bytearray.fromhex(line.split('#')[0])
    return data


def read_file(hexdump, clip=False):
    return MidiFile(file=io.BytesIO(parse_hexdump(hexdump)), clip=clip)


def test_no_tracks():
    assert read_file("""
    4d 54 68 64  # MThd
    00 00 00 06  # Chunk size
    00 01  # Type 1
    00 00  # 0 tracks
    00 78  # 120 ticks per beat
    """).tracks == []


def test_single_message():
    assert read_file(HEADER_ONE_TRACK + """
    4d 54 72 6b  # MTrk
    00 00 00 04
    20 90 40 40  # note_on
    """).tracks[0] == [Message('note_on', note=64, velocity=64, time=32)]


def test_too_long_message():
    with raises(IOError):
        read_file(HEADER_ONE_TRACK + """
        4d 54 72 6b  # MTrk
        00 00 00 04
        00 ff 03 ff ff 7f # extremely long track name message
        """)


def test_two_tracks():
    mid = read_file("""
    4d54 6864 0000 0006 0001 0002 0040        # Header
    4d54 726b 0000 0008 00 90 40 10  40 80 40 10   # Track 0
    4d54 726b 0000 0008 00 90 47 10  40 80 47 10   # Track 1
    """)
    assert len(mid.tracks) == 2
    # TODO: add some more tests here.


def test_empty_file():
    with raises(EOFError):
        read_file("")


def test_eof_in_track():
    with raises(EOFError):
        read_file(HEADER_ONE_TRACK + """
        4d 54 72 6b  # MTrk
        00 00 00 01  # Chunk size
        # Oops, no data here.
        """)


def test_invalid_data_byte_no_clipping():
    # TODO: should this raise IOError?
    with raises(IOError):
        read_file(HEADER_ONE_TRACK + """
        4d 54 72 6b  # MTrk
        00 00 00 04  # Chunk size
        00 90 ff 40  # note_on note=255 velocity=64
        """)


def test_invalid_data_byte_with_clipping_high():
    midi_file = read_file(HEADER_ONE_TRACK + """
                          4d 54 72 6b  # MTrk
                          00 00 00 04  # Chunk size
                          00 90 ff 40  # note_on note=255 velocity=64
                          """, clip=True)
    assert midi_file.tracks[0][0].note == 127


def test_meta_messages():
    # TODO: should this raise IOError?
    mid = read_file(HEADER_ONE_TRACK + """
    4d 54 72 6b  # MTrk
    00 00 00 0c  # Chunk size
    00 ff 03 04 54 65 73 74  # track_name name='Test'
    00 ff 2f 00  # end_of_track
    """)

    track = mid.tracks[0]

    assert track[0] == MetaMessage('track_name', name='Test')
    assert track[1] == MetaMessage('end_of_track')


def test_meta_message_bad_key_sig_throws_key_signature_error_sharps():
    with raises(KeySignatureError):
        read_file(HEADER_ONE_TRACK + """
            4d 54 72 6b  # MTrk
            00 00 00 09  # Chunk size
            00 ff 59 02 08 # Key Signature with 8 sharps
            00 ff 2f 00  # end_of_track
            """)


def test_meta_message_bad_key_sig_throws_key_signature_error_flats():
    with raises(KeySignatureError):
        read_file(HEADER_ONE_TRACK + """
            4d 54 72 6b  # MTrk
            00 00 00 09  # Chunk size
            00 ff 59 02 f8 # Key Signature with 8 flats
            00 ff 2f 00  # end_of_track
            """)


def test_meta_messages_with_length_0():
    """sequence_number and midi_port with no data bytes should be accepted.

    In rare cases these messages have length 0 and thus no data
    bytes. (See issues 42 and 93.) It is unclear why these messages
    are missing their data. It could be cased by a bug in the software
    that created the files.

    So far this has been fixed by adding a test to each of these two
    meta message classes. If the problem appears with other message
    types it may be worth finding a more general solution.
    """
    mid = read_file(HEADER_ONE_TRACK + """
    4d 54 72 6b  # MTrk
    00 00 00 17

    00 ff 00 00  # sequence_number with no data bytes (should be 2).
    00 ff 21 00  # midi_port with no data bytes (should be 1).

    00 ff 00 02 00 01  # sequence_number with correct number of data bytes (2).
    00 ff 21 01 01     # midi_port with correct number of data bytes (1).

    00 ff 2f 00
    """)

    assert mid.tracks[0] == [
        MetaMessage('sequence_number', number=0),
        MetaMessage('midi_port', port=0),

        MetaMessage('sequence_number', number=1),
        MetaMessage('midi_port', port=1),

        MetaMessage('end_of_track'),
    ]


def test_midifile_repr():
    midifile = MidiFile(type=1, ticks_per_beat=123, tracks=[
        MidiTrack([
            Message('note_on', channel=1, note=2, time=3),
            Message('note_off', channel=1, note=2, time=3)]),
        MidiTrack([
            MetaMessage('sequence_number', number=5),
            Message('note_on', channel=2, note=6, time=9),
            Message('note_off', channel=2, note=6, time=9)]),
    ])
    midifile_eval = eval(repr(midifile))  # noqa: S307
    for track, track_eval in zip(midifile.tracks, midifile_eval.tracks):
        for m1, m2 in zip(track, track_eval):
            assert m1 == m2


def _reference_track_bytes(track):
    """Encode a track with the original in-memory algorithm.

    Used to verify that the streaming writer produces byte-identical
    output, including running status and end_of_track handling.
    """
    from mido.midifiles.meta import encode_variable_int
    from mido.midifiles.tracks import fix_end_of_track

    data = bytearray()
    running_status_byte = None
    for msg in fix_end_of_track(track):
        data.extend(encode_variable_int(msg.time))
        if msg.is_meta:
            data.extend(msg.bytes())
            running_status_byte = None
        elif msg.type == 'sysex':
            data.append(0xf0)
            data.extend(encode_variable_int(len(msg.data) + 1))
            data.extend(msg.data)
            data.append(0xf7)
            running_status_byte = None
        else:
            msg_bytes = msg.bytes()
            status_byte = msg_bytes[0]
            if status_byte == running_status_byte:
                data.extend(msg_bytes[1:])
            else:
                data.extend(msg_bytes)
            if status_byte < 0xf0:
                running_status_byte = status_byte
            else:
                running_status_byte = None

    return b'MTrk' + bytes([len(data) >> 24 & 0xff,
                            len(data) >> 16 & 0xff,
                            len(data) >> 8 & 0xff,
                            len(data) & 0xff]) + bytes(data)


class _NonSeekable:
    """Minimal non-seekable binary file object."""

    def __init__(self, max_write=None):
        self.data = bytearray()
        self.max_write = max_write

    def write(self, data):
        data = bytes(data)
        if self.max_write is not None:
            data = data[:self.max_write]
        self.data.extend(data)
        return len(data)

    def seekable(self):
        return False


def test_write_track_matches_reference_encoding():
    from mido.midifiles.midifiles import write_track

    track = MidiTrack([
        Message('note_on', note=60, velocity=64, time=0),
        Message('note_on', note=62, velocity=64, time=1),
        Message('sysex', data=[1, 2, 3], time=2),
        Message('note_on', note=64, velocity=64, time=3),
        MetaMessage('text', text='hello', time=4),
        MetaMessage('end_of_track', time=5),
    ])
    outfile = io.BytesIO()
    write_track(outfile, track)
    assert outfile.getvalue() == _reference_track_bytes(track)


def test_write_track_flushes_across_buffer_boundaries():
    """Chunked writes must not change the bytes or running status."""
    import mido.midifiles.midifiles as midifiles

    track = MidiTrack()
    for i in range(500):
        track.append(Message('note_on', note=i % 128, velocity=64,
                             time=i % 200))
        track.append(Message('note_off', note=i % 128, velocity=0,
                             time=0))
    track.append(MetaMessage('text', text='x' * 300))

    original_size = midifiles.WRITE_BUFFER_SIZE
    try:
        expected = None
        for size in (1, 2, 3, 7, 16, 64, 1024):
            midifiles.WRITE_BUFFER_SIZE = size
            outfile = io.BytesIO()
            midifiles.write_track(outfile, track)
            if expected is None:
                expected = outfile.getvalue()
            else:
                assert outfile.getvalue() == expected
    finally:
        midifiles.WRITE_BUFFER_SIZE = original_size

    assert expected == _reference_track_bytes(track)


def test_write_track_non_seekable_matches_seekable():
    track = MidiTrack()
    for i in range(1000):
        track.append(Message('note_on', note=i % 128, velocity=64, time=1))
    track.append(Message('sysex', data=list(range(128)), time=0))

    midifile = MidiFile(type=0, tracks=[track])
    seekable = io.BytesIO()
    midifile.save(file=seekable)

    non_seekable = _NonSeekable()
    midifile.save(file=non_seekable)

    assert bytes(non_seekable.data) == seekable.getvalue()


def test_write_track_non_seekable_short_writes():
    track = MidiTrack([
        Message('note_on', note=1, velocity=1, time=0),
        Message('sysex', data=list(range(64)), time=1),
    ])
    outfile = _NonSeekable(max_write=3)
    from mido.midifiles.midifiles import write_track
    write_track(outfile, track)
    assert bytes(outfile.data) == _reference_track_bytes(track)


def test_save_round_trip_non_seekable_big_track():
    track = MidiTrack()
    for i in range(2000):
        track.append(Message('note_on', channel=i % 16, note=i % 128,
                             velocity=i % 128, time=i % 100))
    midifile = MidiFile(type=0, tracks=[track])

    outfile = _NonSeekable()
    midifile.save(file=outfile)

    loaded = MidiFile(file=io.BytesIO(bytes(outfile.data)))
    assert list(loaded.tracks[0]) == list(track) + [MetaMessage('end_of_track')]


def test_write_track_length_overflow():
    import mido.midifiles.midifiles as midifiles

    track = MidiTrack([Message('note_on', note=1, velocity=1, time=0)])
    original = midifiles.MAX_TRACK_LENGTH
    midifiles.MAX_TRACK_LENGTH = 3
    try:
        with raises(ValueError, match='32 bit'):
            midifiles.write_track(io.BytesIO(), track)
    finally:
        midifiles.MAX_TRACK_LENGTH = original


def test_write_track_bad_time_reports_event():
    from mido.midifiles.midifiles import write_track

    good = Message('note_on', note=1, velocity=1, time=0)
    bad_float = Message('note_on', skip_checks=True,
                        note=1, velocity=1, time=0.5)
    bad_negative = Message('note_on', skip_checks=True,
                           note=1, velocity=1, time=-1)

    with raises(ValueError, match=r'event 2.*time=0\.5'):
        write_track(io.BytesIO(), [good, good, bad_float])
    with raises(ValueError, match=r'event 0.*time=-1'):
        write_track(io.BytesIO(), [bad_negative])


def test_write_track_rolls_back_seekable_on_error():
    from mido.midifiles.midifiles import write_track

    outfile = io.BytesIO()
    outfile.write(b'header')
    outfile.seek(0, io.SEEK_END)

    track = [Message('note_on', note=1, velocity=1, time=0)] * 1000
    track.append(Message('note_on', skip_checks=True,
                         note=1, velocity=1, time=0.5))

    with raises(ValueError):
        write_track(outfile, track)

    assert outfile.getvalue() == b'header'


def test_save_partial_track_does_not_look_complete():
    """A failing later track must not leave a valid-looking MTrk."""
    track_ok = MidiTrack([Message('note_on', note=1, velocity=1, time=0)])
    track_bad = MidiTrack([
        Message('note_on', note=1, velocity=1, time=0),
        Message('note_on', skip_checks=True, note=1, velocity=1, time=0.5),
    ])
    midifile = MidiFile(type=1, tracks=[track_ok, track_bad])
    outfile = io.BytesIO()

    with raises(ValueError, match='track 1'):
        midifile.save(file=outfile)

    data = outfile.getvalue()
    # The header claims two tracks but the second was rolled back, so
    # the file ends where the first track ends.
    assert data == b'MThd' + (6).to_bytes(4, 'big') + \
        (1).to_bytes(2, 'big') + (2).to_bytes(2, 'big') + \
        (480).to_bytes(2, 'big') + _reference_track_bytes(track_ok)
