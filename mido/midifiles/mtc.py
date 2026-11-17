# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

"""
MIDI Time Code (MTC) timeline extraction.

External recorders and sequencers often drive a show by streaming MTC
quarter_frame messages. Eight consecutive quarter_frame messages carry
one complete time code (hours, minutes, seconds, frames and frame
rate), two frames' worth of messages per time code frame.

This module reassembles those groups from a MidiFile or any message
stream and reports each complete time code with its absolute tick and
the original messages it was built from.

Resynchronization rules (so a corrupt stream never yields a fabricated
time code):

* A group must be eight quarter_frame messages with frame_type 0..7 in
  strictly increasing order. Other message types may be interleaved
  and are ignored.
* A lost, duplicated or out-of-order quarter_frame message discards
  the partially assembled group. If the offending message has
  frame_type 0 it starts a fresh group, otherwise the decoder waits
  for the next frame_type 0.
* A complete group whose values are out of range (hours > 23,
  minutes/seconds > 59, frames >= the nominal frame rate) is dropped
  and never reported.
"""

from ..messages import Message
from .midifiles import MidiFile

# Frame rate code (bits 1-2 of the frame_type 7 nibble) to actual rate.
FRAME_RATES = {0: 24, 1: 25, 2: 29.97, 3: 30}

# Nominal rate used for frame number validation and frame counting.
# 29.97 fps (drop frame) numbers its frames 0..29 like 30 fps.
NOMINAL_FRAME_RATES = {0: 24, 1: 25, 2: 30, 3: 30}

_MESSAGES_PER_TIME_CODE = 8


class TimeCode:
    """A single MIDI time code: hours, minutes, seconds, frames and rate.

    frame_rate is the actual rate (24, 25, 29.97 or 30). The nominal
    rate (30 for 29.97) is used for frame counting and validation.
    """

    __slots__ = ['frame_rate', 'frames', 'hours', 'minutes', 'seconds']

    def __init__(self, hours=0, minutes=0, seconds=0, frames=0,
                 frame_rate=25):
        if frame_rate not in FRAME_RATES.values():
            raise ValueError(
                f'invalid frame rate {frame_rate!r} '
                '(must be 24, 25, 29.97 or 30)')
        nominal = 30 if frame_rate == 29.97 else frame_rate
        if not 0 <= hours <= 23:
            raise ValueError(f'hours out of range: {hours}')
        if not 0 <= minutes <= 59:
            raise ValueError(f'minutes out of range: {minutes}')
        if not 0 <= seconds <= 59:
            raise ValueError(f'seconds out of range: {seconds}')
        if not 0 <= frames < nominal:
            raise ValueError(f'frames out of range: {frames}')

        self.hours = hours
        self.minutes = minutes
        self.seconds = seconds
        self.frames = frames
        self.frame_rate = frame_rate

    @property
    def nominal_frame_rate(self):
        """Frame numbering rate (30 for 29.97, else same as frame_rate)."""
        return 30 if self.frame_rate == 29.97 else self.frame_rate

    @property
    def total_frames(self):
        """Frame count since midnight (using the nominal frame rate)."""
        rate = self.nominal_frame_rate
        return ((self.hours * 60 + self.minutes) * 60
                + self.seconds) * rate + self.frames

    def __eq__(self, other):
        if not isinstance(other, TimeCode):
            return NotImplemented
        return (self.hours, self.minutes, self.seconds,
                self.frames, self.frame_rate) == (
                    other.hours, other.minutes, other.seconds,
                    other.frames, other.frame_rate)

    def __repr__(self):
        return (f'TimeCode(hours={self.hours}, minutes={self.minutes}, '
                f'seconds={self.seconds}, frames={self.frames}, '
                f'frame_rate={self.frame_rate})')

    def __str__(self):
        return (f'{self.hours:02d}:{self.minutes:02d}:'
                f'{self.seconds:02d}:{self.frames:02d}')


class TimeCodeEvent:
    """A complete time code assembled from eight quarter_frame messages.

    Attributes:

    * tick -- absolute tick of the completing (frame_type 7) message.
    * time_code -- the decoded TimeCode.
    * messages -- tuple of the eight original quarter_frame messages.
    * days -- midnight wraparounds seen before this event (only set by
      extract_timecode_events()).
    """

    __slots__ = ['days', 'messages', 'tick', 'time_code']

    def __init__(self, tick, time_code, messages, days=0):
        self.tick = tick
        self.time_code = time_code
        self.messages = messages
        self.days = days

    @property
    def absolute_frames(self):
        """Frame count including midnight wraparounds."""
        frames_per_day = 24 * 60 * 60 * self.time_code.nominal_frame_rate
        return self.days * frames_per_day + self.time_code.total_frames

    def __repr__(self):
        return (f'TimeCodeEvent(tick={self.tick}, '
                f'time_code={self.time_code!r}, days={self.days})')


class MtcDecoder:
    """Reassemble MTC time codes from a stream of messages.

    Feed messages with feed(msg, tick=0). It returns a TimeCodeEvent
    every time a complete, valid group of eight quarter_frame messages
    has been received, and None otherwise. Non-quarter_frame messages
    are ignored. See the module docstring for the resynchronization
    rules.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Discard any partially assembled time code."""
        self._nibbles = [0] * _MESSAGES_PER_TIME_CODE
        self._messages = [None] * _MESSAGES_PER_TIME_CODE
        self._ticks = [0] * _MESSAGES_PER_TIME_CODE
        # Next expected frame_type. 0 means idle (waiting for a group
        # to start) as well as mid-group after a frame_type 0 message.
        self._next = 0

    def feed(self, msg, tick=0):
        """Feed one message, returning a TimeCodeEvent or None."""
        if not isinstance(msg, Message) or msg.type != 'quarter_frame':
            return None

        frame_type = msg.frame_type
        if frame_type != self._next:
            # Lost, duplicated or out-of-order message: drop the
            # partial group. A frame_type 0 message starts over,
            # anything else leaves the decoder idle until one arrives.
            self._next = 0
            if frame_type != 0:
                return None

        self._nibbles[frame_type] = msg.frame_value
        self._messages[frame_type] = msg
        self._ticks[frame_type] = tick
        self._next = frame_type + 1

        if frame_type != _MESSAGES_PER_TIME_CODE - 1:
            return None

        time_code = self._build_time_code()
        messages = tuple(self._messages)
        last_tick = self._ticks[frame_type]
        self._next = 0

        if time_code is None:
            # Out-of-range values: drop the group rather than report
            # a fabricated time code.
            return None
        return TimeCodeEvent(tick=last_tick, time_code=time_code,
                             messages=messages)

    def _build_time_code(self):
        n = self._nibbles
        rate_code = n[7] >> 1
        hours = ((n[7] & 1) << 4) | n[6]
        minutes = (n[5] << 4) | n[4]
        seconds = (n[3] << 4) | n[2]
        frames = ((n[1] & 1) << 4) | n[0]

        if (hours > 23 or minutes > 59 or seconds > 59
                or frames >= NOMINAL_FRAME_RATES[rate_code]):
            return None

        return TimeCode(hours=hours, minutes=minutes, seconds=seconds,
                        frames=frames, frame_rate=FRAME_RATES[rate_code])


def _iter_with_absolute_ticks(source):
    """Yield (message, absolute_tick) pairs in stable order.

    A MidiFile is merged across tracks: messages are ordered by
    absolute tick, ties broken by track order and position within the
    track. Any other iterable is treated as a single message stream
    whose message.time attributes are delta ticks.
    """
    if isinstance(source, MidiFile):
        timed = []
        for track in source.tracks:
            tick = 0
            for msg in track:
                tick += msg.time
                timed.append((tick, msg))
        # sorted() is stable, so equal ticks keep track order and
        # in-track order.
        timed.sort(key=lambda pair: pair[0])
        for tick, msg in timed:
            yield msg, tick
    else:
        tick = 0
        for msg in source:
            tick += msg.time
            yield msg, tick


def extract_timecode_events(source):
    """Extract the MTC timeline from a MidiFile or message stream.

    Returns a list of TimeCodeEvent, one per complete and valid time
    code, in stable order. Each event carries the absolute tick of its
    completing message, the decoded TimeCode and the eight original
    quarter_frame messages.

    Midnight wraparound (23:59:59 -> 00:00:00) is tracked in the
    events' days attribute and absolute_frames property. Other
    backwards jumps (such as a rewound tape) leave days unchanged.
    """
    decoder = MtcDecoder()
    events = []
    days = 0
    previous = None

    for msg, tick in _iter_with_absolute_ticks(source):
        event = decoder.feed(msg, tick)
        if event is None:
            continue

        if (previous is not None
                and previous.hours == 23 and event.time_code.hours == 0
                and event.time_code.total_frames < previous.total_frames):
            days += 1
        event.days = days
        previous = event.time_code
        events.append(event)

    return events
