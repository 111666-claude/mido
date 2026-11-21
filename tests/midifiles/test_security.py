# SPDX-FileCopyrightText: 2024 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for the optional safe-read budget (ReadBudget / SecurityError)."""

import io

import pytest

from mido import Message, MetaMessage, MidiFile, ReadBudget, SecurityError
from mido.midifiles.meta import encode_variable_int

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_file(num_tracks=1, events_per_track=3):
    """Build a small, valid MIDI file and return its bytes."""
    mid = MidiFile(type=1 if num_tracks > 1 else 0)
    for _ in range(num_tracks):
        track = mid.add_track()
        for j in range(events_per_track):
            track.append(Message('note_on', note=60, velocity=64, time=j))
            track.append(Message('note_off', note=60, velocity=64, time=1))
        track.append(MetaMessage('end_of_track', time=0))
    if num_tracks == 1:
        mid.type = 0
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def make_header(num_tracks):
    return (b'MThd' + (6).to_bytes(4, 'big')
            + (1).to_bytes(2, 'big')
            + num_tracks.to_bytes(2, 'big')
            + (120).to_bytes(2, 'big'))


def make_track(payload):
    return b'MTrk' + len(payload).to_bytes(4, 'big') + payload


class NonSeekableReader:
    """A binary stream that only implements read() (no tell/seek)."""

    def __init__(self, data):
        self._buf = io.BytesIO(data)

    def read(self, size=-1):
        return self._buf.read(size)


def load(data, **budget_kwargs):
    budget = ReadBudget(**budget_kwargs)
    return MidiFile(file=io.BytesIO(data), budget=budget), budget


# ---------------------------------------------------------------------------
# Compatibility: without a budget nothing changes
# ---------------------------------------------------------------------------

def test_no_budget_default_behaviour_unchanged():
    data = make_file()
    mid = MidiFile(file=io.BytesIO(data))
    assert len(mid.tracks[0]) == 7  # 3 on/off pairs + end_of_track


def test_budget_with_all_limits_none_accepts_valid_file():
    data = make_file(num_tracks=2, events_per_track=2)
    mid, budget = load(data, max_vlq_length=None)
    assert len(mid.tracks) == 2
    assert budget.bytes_read == len(data)
    assert budget.events_read == 5 * 2


def test_budget_with_non_seekable_stream():
    data = make_file()
    budget = ReadBudget(max_bytes=10 ** 6)
    mid = MidiFile(file=NonSeekableReader(data), budget=budget)
    assert len(mid.tracks[0]) == 7
    assert budget.bytes_read == len(data)


def test_budget_with_filename(tmp_path):
    data = make_file()
    path = tmp_path / 'song.mid'
    path.write_bytes(data)
    mid = MidiFile(str(path), budget=ReadBudget(max_bytes=10 ** 6))
    assert len(mid.tracks[0]) == 7


def test_security_error_is_os_error():
    # Existing code that catches OSError/IOError keeps working.
    assert issubclass(SecurityError, OSError)


def test_budget_argument_type_checked():
    with pytest.raises(TypeError):
        MidiFile(file=io.BytesIO(make_file()), budget=42)


@pytest.mark.parametrize('kwargs', [
    {'max_bytes': -1},
    {'max_tracks': -2},
    {'max_track_length': -3},
    {'max_event_length': -4},
    {'max_events': -5},
    {'max_vlq_length': 0},
])
def test_budget_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        ReadBudget(**kwargs)


def test_budget_rejects_non_integer_values():
    with pytest.raises(ValueError):
        ReadBudget(max_bytes='lots')


# ---------------------------------------------------------------------------
# max_bytes
# ---------------------------------------------------------------------------

def test_max_bytes_exactly_at_limit_passes():
    data = make_file()
    mid, budget = load(data, max_bytes=len(data))
    assert budget.bytes_read == len(data)
    assert len(mid.tracks[0]) == 7


def test_max_bytes_one_too_small_rejected():
    data = make_file()
    with pytest.raises(SecurityError) as info:
        # The 14-byte file header alone exceeds this budget.
        load(data, max_bytes=10)
    err = info.value
    assert err.offset is not None
    # The header is read before any track starts.
    assert err.track is None


def test_non_seekable_stream_enforces_max_bytes():
    data = make_file()
    with pytest.raises(SecurityError):
        MidiFile(file=NonSeekableReader(data),
                 budget=ReadBudget(max_bytes=5))


# ---------------------------------------------------------------------------
# max_tracks
# ---------------------------------------------------------------------------

def test_max_tracks_rejects_declared_count():
    data = make_header(5)
    with pytest.raises(SecurityError) as info:
        load(data, max_tracks=2)
    err = info.value
    assert err.track is None
    assert err.offset == 14


def test_max_tracks_at_boundary_passes():
    data = make_file(num_tracks=2)
    mid, _ = load(data, max_tracks=2)
    assert len(mid.tracks) == 2


# ---------------------------------------------------------------------------
# max_track_length
# ---------------------------------------------------------------------------

def test_max_track_length_rejects_oversized_declaration():
    data = make_header(1) + (b'MTrk' + (100000).to_bytes(4, 'big'))
    with pytest.raises(SecurityError) as info:
        load(data, max_track_length=1000)
    err = info.value
    assert err.track == 0
    assert err.offset == 22  # just after the 8-byte chunk header


def test_track_declaration_longer_than_stream_rejected():
    # Track claims 100 bytes but the stream ends right after the header.
    data = make_header(1) + (b'MTrk' + (100).to_bytes(4, 'big'))
    with pytest.raises(SecurityError) as info:
        load(data)
    err = info.value
    assert err.track == 0
    assert 'end of input' in str(err)


def test_track_declaration_longer_than_stream_without_budget_raises_eof():
    # Backward compatibility: no budget -> plain EOFError, as before.
    data = make_header(1) + (b'MTrk' + (100).to_bytes(4, 'big'))
    with pytest.raises(EOFError):
        MidiFile(file=io.BytesIO(data))


# ---------------------------------------------------------------------------
# max_event_length and SysEx
# ---------------------------------------------------------------------------

def test_sysex_length_beyond_remaining_input_rejected():
    # SysEx VLQ claims 492 bytes but only two follow inside the chunk.
    payload = bytes([0x00, 0xf0]) + bytes(encode_variable_int(492)) \
        + b'\x01\x02'
    data = make_header(1) + make_track(payload)
    with pytest.raises(SecurityError) as info:
        load(data)
    err = info.value
    assert err.track == 0
    assert 'SysEx' in str(err)
    # Offset points at the start of the declared payload.
    assert err.offset == 14 + 8 + 1 + 1 + len(encode_variable_int(492))


def test_sysex_exceeding_max_event_length_rejected():
    payload = bytes([0x00, 0xf0]) + bytes(encode_variable_int(10)) \
        + b'\x00' * 10
    data = make_header(1) + make_track(payload)
    with pytest.raises(SecurityError) as info:
        load(data, max_event_length=5)
    assert 'SysEx' in str(info.value)
    assert info.value.track == 0


def test_meta_event_exceeding_max_event_length_rejected():
    # track_name meta message with 20 bytes of text.
    payload = (bytes([0x00, 0xff, 0x03])
               + bytes(encode_variable_int(20)) + b'A' * 20
               + bytes([0x00, 0xff, 0x2f, 0x00]))
    data = make_header(1) + make_track(payload)
    with pytest.raises(SecurityError) as info:
        load(data, max_event_length=10)
    assert 'meta message' in str(info.value)


def test_event_at_exactly_max_event_length_passes():
    payload = (bytes([0x00, 0xff, 0x03])
               + bytes(encode_variable_int(10)) + b'A' * 10
               + bytes([0x00, 0xff, 0x2f, 0x00]))
    data = make_header(1) + make_track(payload)
    mid, _ = load(data, max_event_length=10)
    assert mid.tracks[0][0].name == 'A' * 10


# ---------------------------------------------------------------------------
# max_events
# ---------------------------------------------------------------------------

def test_max_events_enforced_before_reading():
    data = make_file(events_per_track=3)  # 7 events in the single track
    with pytest.raises(SecurityError) as info:
        load(data, max_events=2)
    err = info.value
    assert err.track == 0
    # Fails before the 3rd event is consumed: at most two events worth of
    # bytes have been read.
    assert err.offset < len(data)


def test_max_events_at_boundary_passes():
    data = make_file(events_per_track=2)  # 5 events
    mid, budget = load(data, max_events=5)
    assert budget.events_read == 5
    assert len(mid.tracks[0]) == 5


# ---------------------------------------------------------------------------
# VLQ
# ---------------------------------------------------------------------------

def test_non_terminating_vlq_rejected_quickly():
    # Eight continuation bytes, never terminating.
    payload = b'\x80' * 8
    data = make_header(1) + make_track(payload)
    with pytest.raises(SecurityError) as info:
        load(data)
    err = info.value
    assert err.track == 0
    # The default cap is 4 bytes, so the 5th byte must never be read.
    assert err.offset == 14 + 8 + 4


def test_five_byte_vlq_rejected_by_default():
    payload = b'\xff\xff\xff\xff\x7f' + b'\x90\x40\x40'
    data = make_header(1) + make_track(payload + b'\x00' * 4)
    with pytest.raises(SecurityError) as info:
        load(data)
    assert 'variable-length quantity' in str(info.value)


def test_four_byte_vlq_is_accepted():
    # 4-byte delta time (spec maximum): 0x8f 0xff 0xff 0x7f = 33554431.
    payload = (b'\x8f\xff\xff\x7f\x90\x40\x40'
               + b'\x00\xff\x2f\x00')
    data = make_header(1) + make_track(payload)
    mid, _ = load(data)
    assert mid.tracks[0][0].time == 33554431


def test_vlq_running_past_track_end_rejected():
    # The chunk only contains 3 continuation bytes, so the VLQ cannot
    # terminate inside the track.
    payload = b'\x80\x80\x80'
    data = make_header(1) + make_track(payload)
    with pytest.raises(SecurityError) as info:
        MidiFile(file=io.BytesIO(data),
                 budget=ReadBudget(max_vlq_length=None))
    assert info.value.track == 0


def test_max_vlq_length_can_be_disabled():
    # A 5-byte VLQ is non-standard but must parse when the cap is off.
    payload = b'\xff\xff\xff\xff\x7f\x90\x40\x40\x00\xff\x2f\x00'
    data = make_header(1) + make_track(payload)
    mid, _ = load(data, max_vlq_length=None)
    # ff ff ff ff 7f decodes to 2**35 - 1.
    assert mid.tracks[0][0].time == 2 ** 35 - 1


# ---------------------------------------------------------------------------
# Rejection speed / resource use
# ---------------------------------------------------------------------------

def test_huge_declared_sysex_does_not_allocate():
    # Claim a ~1 GiB SysEx in a tiny stream. The check must fire before
    # read_bytes allocates, so the test stays fast and memory-bound.
    payload = (bytes([0x00, 0xf0])
               + bytes(encode_variable_int(1 << 30)) + b'\x00\x00')
    data = make_header(1) + make_track(payload)
    with pytest.raises(SecurityError) as info:
        load(data, max_event_length=1 << 20)
    assert info.value.track == 0


def test_huge_declared_track_does_not_read_stream():
    # Largest size a 32-bit chunk header can hold, with no payload.
    data = make_header(1) + (b'MTrk' + (0xffffffff).to_bytes(4, 'big'))
    with pytest.raises(SecurityError) as info:
        load(data, max_track_length=1 << 20)
    # Fails at the chunk header without reading any payload.
    assert info.value.offset == 22


def test_non_terminating_vlq_reads_at_most_four_bytes():
    payload = b'\x80' * 1000
    data = make_header(1) + make_track(payload)
    budget = ReadBudget()
    with pytest.raises(SecurityError):
        MidiFile(file=io.BytesIO(data), budget=budget)
    # 14 header + 8 chunk header + 4 VLQ bytes, nothing more.
    assert budget.bytes_read == 26


# ---------------------------------------------------------------------------
# Error reporting and debug mode
# ---------------------------------------------------------------------------

def test_error_carries_track_and_offset():
    data = make_file(num_tracks=3)
    with pytest.raises(SecurityError) as info:
        load(data, max_events=1)
    err = info.value
    assert isinstance(err.track, int)
    assert isinstance(err.offset, int)
    assert err.offset >= 0


def test_debug_mode_still_enforces_budget():
    data = make_file()
    with pytest.raises(SecurityError):
        MidiFile(file=io.BytesIO(data), debug=True,
                 budget=ReadBudget(max_bytes=5))


def test_multiple_tracks_second_track_reported():
    # First track valid, second track claims an impossible SysEx length.
    first = make_track(b'\x00\xff\x2f\x00')
    bad_sysex = (bytes([0x00, 0xf0])
                 + bytes(encode_variable_int(500)) + b'\x00')
    second = make_track(bad_sysex)
    data = make_header(2) + first + second
    with pytest.raises(SecurityError) as info:
        load(data)
    assert info.value.track == 1


def test_budget_counters_are_updated_on_success():
    data = make_file()
    _, budget = load(data, max_bytes=10 ** 6, max_events=10 ** 6)
    assert budget.bytes_read == len(data)
    assert budget.events_read == 7
