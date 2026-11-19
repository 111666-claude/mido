# SPDX-FileCopyrightText: 2024
#
# SPDX-License-Identifier: MIT

import io
import struct

from pytest import raises

from mido import (
    MAX_SYSEX_LENGTH,
    Message,
    MidiFile,
    MidiTrack,
    NestedSysexError,
    OrphanSysexError,
    SysExFragment,
    SysexTooLongError,
    UnterminatedSysexError,
    get_sysex_fragments,
    iter_sysex_messages,
    iter_track_events,
)
from mido.midifiles.midifiles import (
    encode_variable_int,
    write_chunk,
)

# A three-part tone preset: manufacturer + bulk data + checksum.
TONE_PART_1 = [0x41, 0x10, 0x42, 0x12, 0x01]
TONE_PART_2 = [0x00] * 32
TONE_PART_3 = [0x7f, 0x00, 0x40]
TONE_DATA = TONE_PART_1 + TONE_PART_2 + TONE_PART_3


def escape_event(delta, status, raw):
    """Build raw bytes for an F0/F7 escape event."""
    return bytes(encode_variable_int(delta) + [status]
                 + encode_variable_int(len(raw)) + list(raw))


def note_event(delta, type_, note=64, velocity=99):
    msg = Message(type_, note=note, velocity=velocity, time=delta)
    return bytes(encode_variable_int(msg.time) + msg.bytes())


def build_file_bytes(events, num_tracks=1, type_=1):
    track = bytearray()
    for event in events:
        track += event
    track += bytes([0x00, 0xff, 0x2f, 0x00])  # end of track

    buf = io.BytesIO()
    buf.write(b'MThd' + struct.pack('>L', 6)
              + struct.pack('>hhh', type_, num_tracks, 480))
    write_chunk(buf, b'MTrk', bytes(track))
    return buf.getvalue()


def load(events, **kwargs):
    return MidiFile(file=io.BytesIO(build_file_bytes(events)), **kwargs)


def three_part_tone_events():
    """F0 fragment, note, F7 fragment, note, F7 terminator fragment."""
    return [
        escape_event(0, 0xf0, [0xf0] + TONE_PART_1),
        note_event(10, 'note_on'),
        escape_event(20, 0xf7, TONE_PART_2),
        note_event(30, 'note_off'),
        escape_event(40, 0xf7, TONE_PART_3 + [0xf7]),
    ]


# ---------------------------------------------------------------------
# Requirement 1: per-track iteration over complete SysEx messages
# ---------------------------------------------------------------------

def test_three_fragments_merge_into_one_message():
    mid = load(three_part_tone_events())
    track = mid.tracks[0]

    messages = list(iter_sysex_messages(track, track_index=0))

    assert len(messages) == 1
    merged = messages[0]
    assert merged.data == TONE_DATA
    assert len(merged.fragments) == 3
    assert merged.terminated is True
    assert merged.contains_f0 is False
    assert merged.track_index == 0


def test_fragment_positions_and_ticks_are_preserved():
    mid = load(three_part_tone_events())
    track = mid.tracks[0]
    merged = list(iter_sysex_messages(track))[0]

    first, second, third = merged.fragments
    # Absolute ticks: 0, 0+10+20, +30+40.
    assert (first.tick, second.tick, third.tick) == (0, 30, 100)
    assert merged.start_tick == 0
    assert merged.end_tick == 100
    # Original fragment locations inside the track.
    assert [f.index for f in merged.fragments] == [0, 2, 4]
    assert [f.delta for f in merged.fragments] == [0, 20, 40]
    assert first.status == 0xf0
    assert second.status == 0xf7
    assert third.status == 0xf7
    # The fragment points at the message still living in the track.
    assert track[first.index] is first.message


def test_raw_fragment_bytes_are_kept():
    mid = load(three_part_tone_events())
    fragments = get_sysex_fragments(mid.tracks[0])

    assert fragments[0].raw == [0xf0] + TONE_PART_1
    assert fragments[1].raw == TONE_PART_2
    assert fragments[2].raw == TONE_PART_3 + [0xf7]
    assert fragments[0].terminated is False
    assert fragments[2].terminated is True
    # Payload strips the framing bytes.
    assert fragments[0].payload == TONE_PART_1
    assert fragments[2].payload == TONE_PART_3


def test_midifile_iter_sysex_entry_point():
    mid = load(three_part_tone_events())
    messages = list(mid.iter_sysex())

    assert len(messages) == 1
    assert messages[0].data == TONE_DATA
    assert messages[0].track_index == 0


def test_merged_as_message():
    mid = load(three_part_tone_events())
    merged = list(mid.iter_sysex())[0]
    msg = merged.as_message()

    assert msg.type == 'sysex'
    assert list(msg.data) == TONE_DATA
    assert msg.time == merged.start_tick


def test_several_complete_messages_in_one_track():
    events = [
        escape_event(0, 0xf0, [0xf0, 0x01, 0xf7]),
        escape_event(5, 0xf0, [0xf0, 0x02, 0x03, 0xf7]),
    ]
    mid = load(events)

    messages = list(iter_sysex_messages(mid.tracks[0]))
    assert [m.data for m in messages] == [[1], [2, 3]]
    assert all(len(m.fragments) == 1 for m in messages)


# ---------------------------------------------------------------------
# Requirement 2: interleaved events keep their order; lossless write-back
# ---------------------------------------------------------------------

def test_interleaved_events_keep_order_in_event_view():
    mid = load(three_part_tone_events())
    events = list(iter_track_events(mid.tracks[0]))

    # The merged SysEx is delivered at its last fragment's position,
    # notes and end-of-track keep their places and delta times.
    assert events[0] == Message('note_on', note=64, velocity=99, time=10)
    assert events[1] == Message('note_off', note=64, velocity=99, time=30)
    merged = events[2]
    assert merged.data == TONE_DATA
    assert merged.start_tick == 0
    assert events[3].type == 'end_of_track'


def test_track_itself_is_not_mutated_by_merging():
    mid = load(three_part_tone_events())
    track = mid.tracks[0]
    before = list(track)

    list(iter_sysex_messages(track))
    list(iter_track_events(track))

    assert list(track) == before
    # The three stripped sysex messages remain in the track.
    assert sum(1 for m in track if m.type == 'sysex') == 3


def test_roundtrip_is_byte_identical():
    original = build_file_bytes(three_part_tone_events())
    mid = MidiFile(file=io.BytesIO(original))

    out = io.BytesIO()
    mid.save(file=out)

    assert out.getvalue() == original


def test_roundtrip_after_modifying_fragment_raw_bytes():
    original = build_file_bytes(three_part_tone_events())
    mid = MidiFile(file=io.BytesIO(original))
    fragment = mid.tracks[0].sysex_fragments[1]
    fragment.raw = [0x55, 0x66]  # writable segmentation info

    out = io.BytesIO()
    mid.save(file=out)
    reloaded = MidiFile(file=io.BytesIO(out.getvalue()))

    assert get_sysex_fragments(reloaded.tracks[0])[1].raw == [0x55, 0x66]
    # Everything else survived the round-trip unchanged.
    merged = list(reloaded.iter_sysex())[0]
    assert merged.data == TONE_PART_1 + [0x55, 0x66] + TONE_PART_3
    assert [m.type for m in reloaded.tracks[0]
            if m.type != 'sysex'] == ['note_on', 'note_off',
                                      'end_of_track']


def test_standard_single_sysex_still_roundtrips():
    events = [escape_event(0, 0xf0, [0xf0, 0x41, 0x10, 0xf7])]
    original = build_file_bytes(events)
    mid = MidiFile(file=io.BytesIO(original))

    out = io.BytesIO()
    mid.save(file=out)
    assert out.getvalue() == original


def test_sysex_added_in_memory_uses_standard_wrapping():
    track = MidiTrack([
        Message('sysex', data=[1, 2, 3], time=0),
    ])
    mid = MidiFile(tracks=[track], type=0)
    out = io.BytesIO()
    mid.save(file=out)

    reloaded = MidiFile(file=io.BytesIO(out.getvalue()))
    assert reloaded.tracks[0][0] == Message(
        'sysex', data=[1, 2, 3], time=0)


# ---------------------------------------------------------------------
# Requirement 3: malformed input has well-defined results
# ---------------------------------------------------------------------

def test_orphan_f7_raises_by_default():
    events = [escape_event(0, 0xf7, [0x42, 0xf7])]
    mid = load(events)

    with raises(OrphanSysexError):
        list(iter_sysex_messages(mid.tracks[0]))


def test_orphan_f7_can_be_returned():
    events = [escape_event(0, 0xf7, [0x42, 0xf7])]
    mid = load(events)

    messages = list(iter_sysex_messages(
        mid.tracks[0], allow_orphan=True))
    assert len(messages) == 1
    assert messages[0].data == [0x42]
    assert messages[0].fragments[0].status == 0xf7


def test_nested_f0_raises_by_default():
    events = [
        escape_event(0, 0xf0, [0xf0, 0x01]),          # open, no F7
        escape_event(5, 0xf0, [0xf0, 0x02, 0xf7]),    # nested start
    ]
    mid = load(events)

    with raises(NestedSysexError):
        list(iter_sysex_messages(mid.tracks[0]))


def test_embedded_f0_byte_raises():
    # The embedded F0 sits inside a continuation fragment, so the file
    # itself loads and the grouping layer flags the nested start.
    events = [
        escape_event(0, 0xf0, [0xf0, 0x01]),
        escape_event(5, 0xf7, [0xf0, 0x02, 0xf7]),
    ]
    mid = load(events)

    with raises(NestedSysexError):
        list(iter_sysex_messages(mid.tracks[0]))


def test_invalid_status_byte_in_single_sysex_still_raises():
    events = [escape_event(0, 0xf0, [0xf0, 0x01, 0xf0, 0x02, 0xf7])]
    with raises(OSError):
        load(events)


def test_nested_f0_can_be_split_into_two_messages():
    events = [
        escape_event(0, 0xf0, [0xf0, 0x01]),
        escape_event(5, 0xf0, [0xf0, 0x02, 0xf7]),
    ]
    mid = load(events)

    messages = list(iter_sysex_messages(
        mid.tracks[0], allow_nested=True))
    assert [m.data for m in messages] == [[1], [2]]
    assert messages[0].terminated is False
    assert messages[1].terminated is True


def test_missing_terminator_raises_at_end_of_track():
    events = [escape_event(0, 0xf0, [0xf0, 0x01])]
    mid = load(events)

    with raises(UnterminatedSysexError):
        list(iter_sysex_messages(mid.tracks[0]))


def test_missing_terminator_can_be_returned():
    events = [escape_event(0, 0xf0, [0xf0, 0x01])]
    mid = load(events)

    messages = list(iter_sysex_messages(
        mid.tracks[0], allow_unterminated=True))
    assert len(messages) == 1
    assert messages[0].data == [1]
    assert messages[0].terminated is False


def test_oversized_payload_raises():
    # A payload over MAX_SYSEX_LENGTH split across two fragments: each
    # fragment is readable on its own, only the merge exceeds the limit.
    half = MAX_SYSEX_LENGTH // 2 + 1
    events = [
        escape_event(0, 0xf0, [0xf0] + [0x42] * half),
        escape_event(0, 0xf7, [0x43] * half + [0xf7]),
    ]
    mid = load(events)

    with raises(SysexTooLongError):
        list(iter_sysex_messages(mid.tracks[0]))


def test_file_level_size_limit_still_enforced():
    big = [0x42] * (MAX_SYSEX_LENGTH + 1)
    events = [escape_event(0, 0xf0, [0xf0] + big + [0xf7])]
    with raises(OSError):
        load(events)


def test_custom_max_length():
    events = [
        escape_event(0, 0xf0, [0xf0, 1, 2, 3, 4, 0xf7]),
    ]
    mid = load(events)

    with raises(SysexTooLongError):
        list(iter_sysex_messages(mid.tracks[0], max_length=3))

    messages = list(iter_sysex_messages(
        mid.tracks[0], max_length=4))
    assert messages[0].data == [1, 2, 3, 4]


def test_continuation_does_not_cross_tracks():
    buf = io.BytesIO()
    buf.write(b'MThd' + struct.pack('>L', 6)
              + struct.pack('>hhh', 1, 2, 480))
    # Track 0: open SysEx, never terminated.
    write_chunk(buf, b'MTrk', bytes(
        escape_event(0, 0xf0, [0xf0, 0x01])
        + bytes([0x00, 0xff, 0x2f, 0x00])))
    # Track 1: F7 continuation, orphaned because groups stay per-track.
    write_chunk(buf, b'MTrk', bytes(
        escape_event(0, 0xf7, [0x02, 0xf7])
        + bytes([0x00, 0xff, 0x2f, 0x00])))
    mid = MidiFile(file=io.BytesIO(buf.getvalue()))

    with raises(UnterminatedSysexError):
        list(mid.iter_sysex())
    with raises(OrphanSysexError):
        list(iter_sysex_messages(mid.tracks[1]))

    # Lenient view keeps both problems visible, separately.
    messages = list(mid.iter_sysex(
        allow_unterminated=True, allow_orphan=True))
    assert [(m.track_index, m.data) for m in messages] == [
        (0, [1]),
        (1, [2]),
    ]


# ---------------------------------------------------------------------
# Requirement 4: full integration check with the three-part tone data
# ---------------------------------------------------------------------

def test_tone_preset_end_to_end():
    original = build_file_bytes(three_part_tone_events())

    # 1. Import and see one whole tone preset instead of three pieces.
    mid = MidiFile(file=io.BytesIO(original))
    presets = list(mid.iter_sysex())
    assert len(presets) == 1
    preset = presets[0]
    assert preset.data == TONE_DATA
    assert len(preset.fragments) == 3
    assert preset.start_tick == 0
    assert preset.end_tick == 100

    # 2. Interleaved notes are untouched and still in playback order.
    notes = [m for m in mid.tracks[0] if m.type in
             ('note_on', 'note_off')]
    assert [(m.type, m.time) for m in notes] == [
        ('note_on', 10), ('note_off', 30)]

    # 3. Segmentation metadata is writable.
    for fragment in preset.fragments:
        assert isinstance(fragment, SysExFragment)
    preset.fragments[0].delta = 7

    # 4. Saving reproduces the original file except the edited delta.
    out = io.BytesIO()
    mid.save(file=out)
    assert out.getvalue() != original

    reloaded = MidiFile(file=io.BytesIO(out.getvalue()))
    reloaded_preset = list(reloaded.iter_sysex())[0]
    assert reloaded_preset.data == TONE_DATA
    assert reloaded_preset.fragments[0].delta == 7
    assert [(m.type, m.time) for m in reloaded.tracks[0]
            if m.type in ('note_on', 'note_off')] == [
        ('note_on', 10), ('note_off', 30)]


def test_multiple_three_part_presets_roundtrip():
    events = three_part_tone_events() + [
        escape_event(5, 0xf0, [0xf0] + TONE_PART_1),
        note_event(10, 'note_on', note=48),
        escape_event(20, 0xf7, TONE_PART_2),
        note_event(30, 'note_off', note=48),
        escape_event(40, 0xf7, TONE_PART_3 + [0xf7]),
    ]
    original = build_file_bytes(events)
    mid = MidiFile(file=io.BytesIO(original))

    presets = list(mid.iter_sysex())
    assert len(presets) == 2
    assert all(p.data == TONE_DATA for p in presets)

    out = io.BytesIO()
    mid.save(file=out)
    assert out.getvalue() == original


# ---------------------------------------------------------------------
# Metadata lifecycle: in-memory tracks, copies, clipping
# ---------------------------------------------------------------------

def test_in_memory_track_synthesises_fragments():
    track = MidiTrack([
        Message('sysex', data=[1, 2], time=0),
        Message('note_on', note=60, time=10),
    ])
    # No recorded metadata yet.
    assert track.sysex_fragments == []

    messages = list(iter_sysex_messages(track))
    assert len(messages) == 1
    assert messages[0].data == [1, 2]
    assert messages[0].start_tick == 0

    mid = MidiFile(tracks=[track], type=0)
    out = io.BytesIO()
    mid.save(file=out)
    reloaded = MidiFile(file=io.BytesIO(out.getvalue()))
    assert reloaded.tracks[0][0].data == (1, 2)


def test_track_copy_keeps_fragment_metadata():
    mid = load(three_part_tone_events())
    original = mid.tracks[0]
    clone = original.copy()

    assert len(clone.sysex_fragments) == 3
    assert clone.sysex_fragments is not original.sysex_fragments
    assert clone.sysex_fragments[0].raw == [0xf0] + TONE_PART_1

    out = io.BytesIO()
    MidiFile(tracks=[clone], type=0).save(file=out)
    reloaded = MidiFile(file=io.BytesIO(out.getvalue()))
    assert list(reloaded.iter_sysex())[0].data == TONE_DATA


def test_clip_mode_roundtrip_keeps_raw_fragment():
    events = [escape_event(0, 0xf0, [0xf0, 0x01, 0xff, 0xf7])]
    original = build_file_bytes(events)
    mid = MidiFile(file=io.BytesIO(original), clip=True)

    # The message payload is clipped, but the raw fragment still
    # contains the original byte for a lossless save.
    fragment = get_sysex_fragments(mid.tracks[0])[0]
    assert fragment.raw == [0xf0, 0x01, 0xff, 0xf7]
    assert mid.tracks[0][0].data == (1, 127)

    out = io.BytesIO()
    mid.save(file=out)
    assert out.getvalue() == original
