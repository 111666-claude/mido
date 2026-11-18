# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Per-track sequence views for type 2 (asynchronous) MIDI files.

In a type 2 file each track is an independent pattern. The tracks do
not share a timeline, so ``MidiFile`` refuses to merge them (and
``MidiFile.length`` is not available either). A previewer that wants to
play one pattern at a time needs the events of a single track with
their absolute positions in ticks and seconds.

A :class:`TrackSequence` provides exactly that. Every track is
converted with its own tempo map built from the ``set_tempo`` messages
in that track only, so pattern durations and event times are
independent of the other tracks. The time division
(``ticks_per_beat``) belongs to the file header and is shared by all
tracks.

The file is only read. Messages in a sequence are independent copies,
so neither the tracks nor the data written by a later ``save()`` can be
changed through a sequence view.
"""

from dataclasses import dataclass
from numbers import Integral

from .midifiles import DEFAULT_TEMPO
from .units import tick2second

# The largest value a MIDI variable-length quantity can hold. Both the
# delta time of a single message and the absolute tick of a message
# must fit, so that the sequence can be represented in a MIDI file.
MAX_TICK = 0x0FFFFFFF


class SequenceTimeOverflow(OverflowError, ValueError):
    """A time value does not fit in a MIDI variable-length quantity.

    Raised when a delta time or an absolute tick exceeds
    :data:`MAX_TICK`. It is both an :class:`OverflowError` (the value
    is out of range for MIDI time) and a :class:`ValueError` (the
    track holds an invalid value), so either can be caught.
    """


@dataclass(frozen=True)
class SequenceEvent:
    """One message of a track sequence with absolute positions.

    Attributes:
        track: Index of the track the message came from.
        index: Position of the message inside the track (0-based).
        tick: Absolute time in ticks from the start of the track.
        seconds: Absolute time in seconds from the start of the track,
            computed with the track's own tempo map.
        message: An independent copy of the message. Its ``time``
            attribute is still the original delta time in ticks, so the
            sequence can be used to rebuild the track byte for byte.
    """

    track: int
    index: int
    tick: int
    seconds: float
    message: object

    @property
    def type(self):
        """Message type (for example ``'note_on'`` or ``'set_tempo'``)."""
        return self.message.type

    @property
    def is_meta(self):
        """True if the message is a meta message."""
        return self.message.is_meta

    @property
    def is_sysex(self):
        """True if the message is a system exclusive message."""
        return self.message.type == 'sysex'


class TrackSequence:
    """An independent pattern from a type 2 MIDI file.

    Iterating over the sequence yields :class:`SequenceEvent` objects
    in track order, so messages that share a tick keep their original
    order. Meta messages (including ``end_of_track``) and sysex
    messages are included.

    Attributes:
        track: Index of the track.
        events: List of :class:`SequenceEvent` objects.
        length_ticks: Playback length in ticks, i.e. the absolute tick
            of the last message (or 0 for an empty track).
        length: Playback length in seconds, using the track's own tempo
            map (or 0.0 for an empty track).
    """

    def __init__(self, track, events, length_ticks, length):
        self.track = track
        self.events = list(events)
        self.length_ticks = length_ticks
        self.length = length

    @property
    def index(self):
        """Index of the track (alias for :attr:`track`)."""
        return self.track

    @property
    def messages(self):
        """The copied messages of the sequence, without time positions."""
        return [event.message for event in self.events]

    def __iter__(self):
        return iter(self.events)

    def __len__(self):
        return len(self.events)

    def __getitem__(self, index):
        return self.events[index]

    def __eq__(self, other):
        if not isinstance(other, TrackSequence):
            return NotImplemented
        return (self.track == other.track
                and self.events == other.events
                and self.length_ticks == other.length_ticks
                and self.length == other.length)

    def __repr__(self):
        return (f'{self.__class__.__name__}(track={self.track!r}, '
                f'events={self.events!r}, '
                f'length_ticks={self.length_ticks!r}, '
                f'length={self.length!r})')


def _check_type(midifile):
    if midifile.type != 2:
        raise TypeError(
            f'track sequences are only available for type 2 '
            f'(asynchronous) files, not type {midifile.type}; type 0 and '
            f'type 1 files share one synchronized timeline')


def _check_track_index(midifile, track_index):
    if isinstance(track_index, bool) or not isinstance(track_index,
                                                       Integral):
        raise TypeError(
            f'track index must be an integer, got {track_index!r}')
    num_tracks = len(midifile.tracks)
    if not 0 <= track_index < num_tracks:
        raise IndexError(
            f'track index {track_index} out of range for a file with '
            f'{num_tracks} track(s)')
    return int(track_index)


def _check_ticks_per_beat(midifile):
    ticks_per_beat = midifile.ticks_per_beat
    if isinstance(ticks_per_beat, bool) or not isinstance(
            ticks_per_beat, Integral):
        raise TypeError(
            f'ticks_per_beat must be an integer, got {ticks_per_beat!r}')
    if ticks_per_beat <= 0:
        raise ValueError(
            f'invalid time division {ticks_per_beat!r}: SMPTE time bases '
            f'and non-positive ticks_per_beat are not supported')
    return int(ticks_per_beat)


def _check_tempo(tempo, track_index, message_index):
    if isinstance(tempo, bool) or not isinstance(tempo, Integral):
        raise TypeError(
            f'tempo must be an integer, got {tempo!r} at message '
            f'{message_index} in track {track_index}')
    if tempo <= 0:
        raise ValueError(
            f'invalid tempo {tempo} at message {message_index} in track '
            f'{track_index} (must be positive microseconds per beat)')
    return int(tempo)


def get_track_sequence(midifile, track_index):
    """Return the independent sequence of one track of a type 2 file.

    ``midifile`` must be a type 2 (asynchronous) file and
    ``track_index`` a valid index into its ``tracks`` list.

    Each event gets the absolute tick and absolute time in seconds
    from the start of the track. Seconds are computed from that track's
    ``set_tempo`` messages only; the default tempo of 500000
    microseconds per beat applies before the first one. A ``set_tempo``
    message is timestamped with the tempo that was in effect before it,
    and the new tempo applies to the ticks that follow it.

    The file is not modified: the returned messages are copies. An
    empty track gives a sequence with no events and a length of zero.

    Raises:
        TypeError: for type 0 or type 1 files, for a non-integer track
            index or an invalid ``ticks_per_beat``.
        IndexError: if ``track_index`` is out of range.
        ValueError: for a negative or non-integer delta time, an invalid
            tempo, or an invalid time division (including SMPTE).
        SequenceTimeOverflow: if a delta time or absolute tick exceeds
            the maximum MIDI tick value (both an ``OverflowError`` and
            a ``ValueError``).
    """
    _check_type(midifile)
    track_index = _check_track_index(midifile, track_index)
    ticks_per_beat = _check_ticks_per_beat(midifile)

    track = midifile.tracks[track_index]
    tempo = DEFAULT_TEMPO
    tick = 0
    seconds = 0.0
    events = []

    for message_index, msg in enumerate(track):
        delta = msg.time
        if isinstance(delta, bool) or not isinstance(delta, Integral):
            raise ValueError(
                f'message {message_index} in track {track_index} has a '
                f'non-integer delta time {delta!r} (MIDI file times must '
                f'be integer ticks)')
        if delta < 0:
            raise ValueError(
                f'message {message_index} in track {track_index} has a '
                f'negative delta time {delta!r}')
        if delta > MAX_TICK:
            raise SequenceTimeOverflow(
                f'delta time {delta} at message {message_index} in track '
                f'{track_index} exceeds the maximum MIDI tick value '
                f'{MAX_TICK}')
        if tick + delta > MAX_TICK:
            raise SequenceTimeOverflow(
                f'absolute tick {tick + delta} at message {message_index} '
                f'in track {track_index} exceeds the maximum MIDI tick '
                f'value {MAX_TICK}')

        tick += delta
        seconds += tick2second(delta, ticks_per_beat, tempo)
        events.append(SequenceEvent(
            track=track_index,
            index=message_index,
            tick=tick,
            seconds=seconds,
            message=msg.copy(),
        ))

        if msg.type == 'set_tempo':
            tempo = _check_tempo(msg.tempo, track_index, message_index)

    return TrackSequence(track_index, events, tick, seconds)
