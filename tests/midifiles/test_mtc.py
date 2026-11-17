# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

import pytest

from mido.messages import Message
from mido.midifiles import MidiFile, MidiTrack
from mido.midifiles.meta import MetaMessage
from mido.midifiles.mtc import (
    FRAME_RATES,
    MtcDecoder,
    TimeCode,
    extract_timecode_events,
)

RATE_CODES = {24: 0, 25: 1, 29.97: 2, 30: 3}


def quarter_frame(frame_type, frame_value, time=0):
    return Message('quarter_frame', frame_type=frame_type,
                   frame_value=frame_value, time=time)


def encode_timecode(hours, minutes, seconds, frames, frame_rate=25,
                    delta=0):
    """Encode a time code as eight quarter_frame messages."""
    rate_code = RATE_CODES[frame_rate]
    nibbles = [
        frames & 0x0f,
        (frames >> 4) & 1,
        seconds & 0x0f,
        (seconds >> 4) & 7,
        minutes & 0x0f,
        (minutes >> 4) & 7,
        hours & 0x0f,
        ((hours >> 4) & 1) | (rate_code << 1),
    ]
    return [quarter_frame(t, v, time=delta) for t, v in enumerate(nibbles)]


def make_frame_by_frame_stream(start, count, frame_rate=25, delta=10):
    """Fixture: count consecutive frames of time code starting at
    start=(hours, minutes, seconds, frames)."""
    hours, minutes, seconds, frames = start
    nominal = 30 if frame_rate == 29.97 else frame_rate
    messages = []
    for _ in range(count):
        messages.extend(
            encode_timecode(hours, minutes, seconds, frames,
                            frame_rate=frame_rate, delta=delta))
        frames += 1
        if frames == nominal:
            frames = 0
            seconds += 1
            if seconds == 60:
                seconds = 0
                minutes += 1
                if minutes == 60:
                    minutes = 0
                    hours = (hours + 1) % 24
    return messages


def test_decode_complete_timecode_all_frame_rates():
    for frame_rate in FRAME_RATES.values():
        decoder = MtcDecoder()
        event = None
        for msg in encode_timecode(1, 2, 3, 4, frame_rate=frame_rate):
            event = decoder.feed(msg) or event
        assert event is not None
        assert event.time_code == TimeCode(hours=1, minutes=2, seconds=3,
                                           frames=4, frame_rate=frame_rate)


def test_event_keeps_absolute_tick_and_original_messages():
    decoder = MtcDecoder()
    messages = encode_timecode(10, 20, 30, 5, frame_rate=24)
    event = None
    tick = 0
    for msg in messages:
        tick += 7
        event = decoder.feed(msg, tick) or event
    # The event is stamped with the tick of the completing message.
    assert event.tick == tick
    assert event.messages == tuple(messages)


def test_ignores_interleaved_non_quarter_frame_messages():
    decoder = MtcDecoder()
    messages = encode_timecode(0, 0, 1, 0)
    event = None
    for msg in messages:
        assert decoder.feed(Message('note_on', note=60)) is None
        event = decoder.feed(msg) or event
    assert event is not None
    assert event.time_code == TimeCode(seconds=1)


def test_frame_by_frame_fixture():
    messages = make_frame_by_frame_stream((1, 0, 0, 23), 4,
                                          frame_rate=25, delta=10)
    events = extract_timecode_events(messages)
    assert [str(e.time_code) for e in events] == [
        '01:00:00:23', '01:00:00:24', '01:00:01:00', '01:00:01:01']
    # delta=10 per message: group n completes at message 8*n+7.
    assert [e.tick for e in events] == [80, 160, 240, 320]


def test_missing_message_drops_group_and_resyncs():
    good = encode_timecode(0, 0, 0, 1)
    broken = encode_timecode(0, 0, 0, 0)
    del broken[3]  # lose the seconds-high nibble
    decoder = MtcDecoder()
    events = []
    for msg in broken + good:
        event = decoder.feed(msg)
        if event is not None:
            events.append(event)
    # The broken group yields nothing; the frame_type 0 message of the
    # following group (arriving while types 4..7 were expected) starts
    # a fresh group which then completes.
    assert len(events) == 1
    assert events[0].time_code == TimeCode(frames=1)


def test_duplicate_message_resyncs():
    messages = encode_timecode(0, 0, 0, 0)
    # Duplicate the frame_type 2 message.
    messages = messages[:3] + [messages[2]] + messages[3:]
    messages += encode_timecode(0, 0, 0, 1)
    events = extract_timecode_events(messages)
    assert len(events) == 1
    assert events[0].time_code == TimeCode(frames=1)


def test_out_of_order_message_resyncs():
    messages = encode_timecode(0, 0, 0, 0)
    # Swap nibbles 4 and 5.
    messages[4], messages[5] = messages[5], messages[4]
    messages += encode_timecode(0, 0, 0, 1)
    events = extract_timecode_events(messages)
    assert len(events) == 1
    assert events[0].time_code == TimeCode(frames=1)


def test_stray_nibble_does_not_start_group():
    decoder = MtcDecoder()
    # A frame_type 5 message with no group in progress is ignored.
    assert decoder.feed(quarter_frame(5, 3)) is None
    messages = encode_timecode(2, 3, 4, 5)
    event = None
    for msg in messages:
        event = decoder.feed(msg) or event
    assert event.time_code == TimeCode(hours=2, minutes=3, seconds=4,
                                       frames=5)


@pytest.mark.parametrize('hours,minutes,seconds,frames,rate', [
    (24, 0, 0, 0, 25),    # hours out of range
    (0, 60, 0, 0, 25),    # minutes out of range
    (0, 0, 60, 0, 25),    # seconds out of range
    (0, 0, 0, 25, 25),    # frame number >= frame rate
    (0, 0, 0, 24, 24),    # frame number >= frame rate
    (0, 0, 0, 30, 30),    # frame number >= frame rate
    (0, 0, 0, 30, 29.97),  # frame number >= nominal rate
])
def test_out_of_range_values_are_dropped(hours, minutes, seconds,
                                         frames, rate):
    messages = encode_timecode(hours, minutes, seconds, frames,
                               frame_rate=rate)
    assert extract_timecode_events(messages) == []


def test_midnight_wraparound_fixture():
    # 23:59:59:24 at 25 fps, then wrapping to 00:00:00:00.
    messages = make_frame_by_frame_stream((23, 59, 59, 24), 3,
                                          frame_rate=25)
    events = extract_timecode_events(messages)
    assert [str(e.time_code) for e in events] == [
        '23:59:59:24', '00:00:00:00', '00:00:00:01']
    assert [e.days for e in events] == [0, 1, 1]
    frames_per_day = 24 * 60 * 60 * 25
    assert events[1].absolute_frames == frames_per_day
    assert events[2].absolute_frames == frames_per_day + 1


def test_backwards_jump_is_not_a_day_wrap():
    messages = (encode_timecode(12, 0, 0, 0)
                + encode_timecode(1, 0, 0, 0))
    events = extract_timecode_events(messages)
    assert [e.days for e in events] == [0, 0]


def test_midifile_extraction_merges_tracks_in_stable_order():
    # The eight nibbles of one time code are split across two tracks
    # with interleaved absolute ticks.
    track1 = MidiTrack()
    track2 = MidiTrack()
    nibbles = encode_timecode(5, 6, 7, 8, frame_rate=30)
    for i, msg in enumerate(nibbles):
        msg.time = 10
        (track1 if i % 2 == 0 else track2).append(msg)
    track1.append(MetaMessage('end_of_track', time=0))
    track2.append(MetaMessage('end_of_track', time=0))
    mid = MidiFile(type=1, tracks=[track1, track2])

    events = extract_timecode_events(mid)
    assert len(events) == 1
    assert events[0].time_code == TimeCode(hours=5, minutes=6, seconds=7,
                                           frames=8, frame_rate=30)
    assert events[0].tick == 40
    # Original messages are preserved in order.
    assert [m.frame_type for m in events[0].messages] == list(range(8))


def test_timecode_total_frames_and_str():
    tc = TimeCode(hours=1, minutes=2, seconds=3, frames=4, frame_rate=25)
    assert tc.total_frames == ((3600 + 120 + 3) * 25) + 4
    assert str(tc) == '01:02:03:04'
    assert tc.nominal_frame_rate == 25
    drop = TimeCode(frames=29, frame_rate=29.97)
    assert drop.nominal_frame_rate == 30


def test_timecode_validation():
    with pytest.raises(ValueError, match='frame rate'):
        TimeCode(frame_rate=48)
    with pytest.raises(ValueError, match='hours'):
        TimeCode(hours=24)
    with pytest.raises(ValueError, match='frames'):
        TimeCode(frames=25, frame_rate=25)


def test_decoder_reset():
    decoder = MtcDecoder()
    for msg in encode_timecode(0, 0, 0, 0)[:5]:
        decoder.feed(msg)
    decoder.reset()
    # The partial group is gone; a lone completing nibble is ignored.
    assert decoder.feed(quarter_frame(7, 0)) is None
