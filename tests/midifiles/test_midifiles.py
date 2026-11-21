# SPDX-FileCopyrightText: 2017 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

import io
from pytest import raises
from mido.messages import Message
from mido.midifiles.midifiles import MidiFile, MidiTrack
from mido.midifiles.meta import MetaMessage, KeySignatureError

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
    midifile_eval = eval(repr(midifile))
    for track, track_eval in zip(midifile.tracks, midifile_eval.tracks):
        for m1, m2 in zip(track, track_eval):
            assert m1 == m2


def _make_file():
    return MidiFile(type=1, ticks_per_beat=120, tracks=[
        MidiTrack([
            Message('note_on', note=60, velocity=64, time=0),
            Message('note_off', note=60, velocity=64, time=10),
        ]),
    ])


def _merged_msg_count(midifile):
    # Exclude the end_of_track message appended by merge_tracks().
    return sum(1 for msg in midifile.merged_track
               if msg.type != 'end_of_track')


def test_merged_track_is_cached_for_read_only_access():
    midifile = _make_file()
    assert midifile.merged_track is midifile.merged_track
    # Playback iteration reads the merged track but must not drop cache.
    list(midifile)
    assert midifile.merged_track is midifile.merged_track


def test_merged_track_invalidated_by_track_mutations():
    midifile = _make_file()
    track = midifile.tracks[0]
    assert _merged_msg_count(midifile) == 2

    # Prime the cache before every mutation, like an editor that
    # previews the file between edits.
    midifile.merged_track
    track.append(Message('note_off', note=60, velocity=0, time=5))
    assert _merged_msg_count(midifile) == 3

    midifile.merged_track
    track.insert(0, Message('control_change', control=1, value=1, time=0))
    assert _merged_msg_count(midifile) == 4

    midifile.merged_track
    track[0] = Message('control_change', control=2, value=2, time=0)
    assert _merged_msg_count(midifile) == 4

    midifile.merged_track
    track[0:2] = [Message('pitchwheel', pitch=1, time=0)]
    assert _merged_msg_count(midifile) == 3

    midifile.merged_track
    track.extend([Message('note_off', note=60, time=1)])
    assert _merged_msg_count(midifile) == 4

    midifile.merged_track
    track += [Message('note_off', note=60, time=1)]
    assert _merged_msg_count(midifile) == 5

    midifile.merged_track
    track.remove(track[0])
    assert _merged_msg_count(midifile) == 4

    midifile.merged_track
    del track[0]
    assert _merged_msg_count(midifile) == 3

    midifile.merged_track
    track.pop()
    assert _merged_msg_count(midifile) == 2

    midifile.merged_track
    track.clear()
    assert _merged_msg_count(midifile) == 0


def test_merged_track_invalidated_by_track_list_mutations():
    midifile = _make_file()
    assert _merged_msg_count(midifile) == 2

    midifile.merged_track
    midifile.add_track()
    midifile.tracks[-1].append(Message('note_on', note=72, time=0))
    assert _merged_msg_count(midifile) == 3

    midifile.merged_track
    midifile.tracks.insert(0, MidiTrack([
        Message('note_on', note=80, time=0)]))
    assert _merged_msg_count(midifile) == 4

    midifile.merged_track
    midifile.tracks[0] = MidiTrack([
        Message('note_on', note=81, time=0)])
    notes = {getattr(msg, 'note', None)
             for msg in midifile.merged_track}
    assert 81 in notes and 80 not in notes

    midifile.merged_track
    midifile.tracks[0:1] = [MidiTrack([Message('note_on', note=82, time=0)])]
    assert any(getattr(msg, 'note', None) == 82
               for msg in midifile.merged_track)

    midifile.merged_track
    midifile.tracks.extend([MidiTrack([Message('note_on', note=83, time=0)])])
    assert _merged_msg_count(midifile) == 5

    midifile.merged_track
    midifile.tracks += [MidiTrack([Message('note_on', note=84, time=0)])]
    assert _merged_msg_count(midifile) == 6

    midifile.merged_track
    midifile.tracks.remove(midifile.tracks[-1])
    assert _merged_msg_count(midifile) == 5

    midifile.merged_track
    del midifile.tracks[-1]
    assert _merged_msg_count(midifile) == 4

    midifile.merged_track
    midifile.tracks.pop()
    assert _merged_msg_count(midifile) == 3

    midifile.merged_track
    midifile.tracks.clear()
    assert _merged_msg_count(midifile) == 0


def test_merged_track_invalidated_via_direct_track_reference():
    midifile = _make_file()
    # The editor holds its own reference to the track object.
    track = midifile.tracks[0]
    midifile.merged_track
    track.append(Message('note_off', note=60, velocity=0, time=5))
    assert _merged_msg_count(midifile) == 3


def test_merged_track_deleter_forces_recompute():
    midifile = _make_file()
    cached = midifile.merged_track
    del midifile.merged_track
    assert midifile.merged_track is not cached


def test_type2_merged_track_still_forbidden_after_mutation():
    midifile = MidiFile(type=2, tracks=[
        MidiTrack([Message('note_on', time=0)])])
    with raises(TypeError):
        midifile.merged_track
    midifile.tracks[0].append(Message('note_off', time=1))
    with raises(TypeError):
        midifile.merged_track


def test_merged_track_keeps_end_of_track_and_tie_order():
    midifile = MidiFile(type=1, tracks=[
        MidiTrack([Message('control_change', control=i, time=10)])
        for i in range(3)
    ])
    merged = midifile.merged_track
    assert merged[-1].type == 'end_of_track'
    assert [msg.control for msg in merged
            if msg.type == 'control_change'] == [0, 1, 2]
