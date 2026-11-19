# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

"""
Split MIDI tracks into instrument lanes.

A single MIDI track often contains several instruments. Every time a
``program_change`` message is sent on a channel the instrument changes,
so notes played before and after the change belong to different
instruments. This module separates the channel messages of one or more
tracks into :class:`InstrumentLane` objects, one for every
``(channel, program)`` interval, so each instrument can be edited
independently.

Only channel messages (messages carrying a ``channel`` attribute) are
assigned to a lane. Meta messages and system exclusive messages are not
instrument specific and are left out.

Ownership rules
---------------

* Note events are owned by the instrument active at their *note_on*.
  A note is always closed on the same lane, even when a
  ``program_change`` happens while the note is sounding or when the
  ``note_off`` is missing.
* ``control_change`` (including the sustain pedal, controller 64),
  ``pitchwheel`` and channel pressure / polyphonic key pressure events
  are owned by the instrument active at the tick of the event.
* ``program_change`` is owned by the lane it switches *to*.
* The drum channel (channel 10, index 9) is never split: percussion is
  selected by note number instead of program, so all its messages land
  in a single lane whose ``program`` is ``None``, regardless of any
  program changes it contains.
"""

from dataclasses import dataclass, field

from ..messages import Message
from .meta import MetaMessage
from .midifiles import DEFAULT_TEMPO
from .tracks import MidiTrack, _to_abstime, _to_reltime
from .units import tick2second

#: The General MIDI drum channel, as a zero-based channel index.
DRUM_CHANNEL = 9


@dataclass
class LaneEvent:
    """A channel message placed in an :class:`InstrumentLane`.

    Attributes
    ----------
    message:
        The MIDI message. Its ``time`` attribute is its absolute
        position in ticks (it is no longer a delta time).
    tick:
        Absolute position in ticks, identical to ``message.time``.
    second:
        Absolute position in seconds, computed from the file tempo map.
    track:
        Index of the source track the message came from.
    synthesized:
        ``True`` when a closing ``note_off`` had to be generated because
        the source track ended while the note was still sounding.
    """

    message: object
    tick: int
    second: float
    track: int
    synthesized: bool = False


@dataclass
class InstrumentLane:
    """All channel messages belonging to one instrument interval.

    A lane is identified by a channel and a program number. The drum
    channel is represented by a single lane whose ``program`` is
    ``None``.
    """

    channel: int
    program: object  # int for normal channels, None for the drum channel
    events: list = field(default_factory=list)

    @property
    def is_drum(self):
        """``True`` for the single lane of the drum channel."""
        return self.channel == DRUM_CHANNEL

    @property
    def source_tracks(self):
        """Sorted, de-duplicated indices of the source tracks in use."""
        return sorted({event.track for event in self.events})

    def to_midi_track(self):
        """Return the lane as a :class:`MidiTrack`.

        Messages keep their original absolute tick positions, converted
        back to the delta times used in MIDI files. A synthesized
        closing ``note_off`` is emitted as a real message. An
        ``end_of_track`` meta message is appended.
        """
        messages = [event.message.copy(time=event.tick)
                    for event in self.events]
        messages.sort(key=lambda message: message.time)
        track = MidiTrack(
            _to_reltime(messages, skip_checks=True),
        )
        track.append(MetaMessage('end_of_track', time=0))
        return track


def _is_channel_message(message):
    """Return True for channel voice/channel mode messages."""
    return not message.is_meta and hasattr(message, 'channel')


def build_tempo_map(tracks):
    """Build an absolute-tick tempo map from track meta messages.

    Returns a sorted list of ``(tick, tempo)`` pairs containing every
    ``set_tempo`` meta message found in the tracks.
    """
    tempo_map = []
    for track in tracks:
        tick = 0
        for message in track:
            tick += message.time
            if message.type == 'set_tempo':
                tempo_map.append((tick, message.tempo))
    tempo_map.sort(key=lambda item: item[0])
    return tempo_map


def tick_at(tick, tempo_map, ticks_per_beat):
    """Convert an absolute tick to absolute seconds with a tempo map.

    A tempo change at tick *t* affects the time of messages at ticks
    greater than *t*, matching the playback semantics of
    :meth:`MidiFile.__iter__`.
    """
    second = 0.0
    last_tick = 0
    tempo = DEFAULT_TEMPO
    for change_tick, new_tempo in tempo_map:
        if change_tick >= tick:
            break
        second += tick2second(change_tick - last_tick,
                              ticks_per_beat, tempo)
        last_tick = change_tick
        tempo = new_tempo
    second += tick2second(tick - last_tick, ticks_per_beat, tempo)
    return second


def split_track_lanes(track, track_index=0, tempo_map=None,
                      ticks_per_beat=480):
    """Split one track into instrument lanes.

    Parameters
    ----------
    track:
        The :class:`MidiTrack` to split. Message times are delta times
        in ticks, as stored in a MIDI file.
    track_index:
        Index stored on every :class:`LaneEvent` as its source track.
    tempo_map:
        An optional tempo map as produced by :func:`build_tempo_map`.
        When omitted, the default 120 BPM tempo is used.
    ticks_per_beat:
        Tick resolution used when computing event seconds.

    Returns
    -------
    list of InstrumentLane
        Lanes ordered by the tick at which they first appear, then by
        channel and program for deterministic ordering.
    """
    if tempo_map is None:
        tempo_map = []

    # State is kept per channel and reset per source track. Tracks are
    # independent chunks of a file and may reuse channels for different
    # instruments.
    active_program = {channel: 0 for channel in range(16)}
    lanes = {}            # (channel, program) -> InstrumentLane
    first_seen = {}       # lane key -> tick of its first event
    # Open notes per (channel, note), each entry holding the lane key
    # the note_on was assigned to. Notes are paired FIFO.
    open_notes = {}

    def get_lane(key, tick):
        if key not in lanes:
            lanes[key] = InstrumentLane(key[0], key[1])
            first_seen[key] = tick
        return lanes[key]

    def add_event(message, tick, key, synthesized=False):
        second = tick_at(tick, tempo_map, ticks_per_beat)
        get_lane(key, tick).events.append(
            LaneEvent(message=message, tick=tick, second=second,
                      track=track_index, synthesized=synthesized))

    def current_key(channel):
        if channel == DRUM_CHANNEL:
            return channel, None
        return channel, active_program[channel]

    absolute_tick = 0
    for message in _to_abstime(track):
        absolute_tick = message.time

        if not _is_channel_message(message):
            continue

        channel = message.channel
        mtype = message.type

        if mtype == 'program_change':
            if not 0 <= message.program <= 127:
                raise ValueError(
                    f'program_change program must be in range 0..127,'
                    f' got {message.program}')
            if channel == DRUM_CHANNEL:
                # Percussion is selected by note number, not program.
                # Keep the message on the drum lane and do not split.
                add_event(message, absolute_tick, current_key(channel))
            else:
                active_program[channel] = message.program
                # The change belongs to the lane it switches to.
                add_event(message, absolute_tick, current_key(channel))
            continue

        key = current_key(channel)

        if mtype == 'note_on' and message.velocity > 0:
            add_event(message, absolute_tick, key)
            open_notes.setdefault((channel, message.note), []).append(key)
        elif mtype == 'note_off' or (mtype == 'note_on'
                                     and message.velocity == 0):
            pending = open_notes.get((channel, message.note))
            if pending:
                # Close on the lane the note was started on, even if
                # the instrument changed in the meantime.
                note_key = pending.pop(0)
                if not pending:
                    del open_notes[(channel, message.note)]
                add_event(message, absolute_tick, note_key)
            else:
                # A note_off without a matching note_on follows the
                # current instrument instead of inventing a new lane.
                add_event(message, absolute_tick, key)
        else:
            # control_change (incl. sustain), pitchwheel,
            # polytouch and aftertouch follow the active instrument.
            add_event(message, absolute_tick, key)

    # Notes with no note_off are closed at the end of the source track
    # on the lane that owns the sounding note.
    for (channel, note), pending in open_notes.items():
        for key in pending:
            closing = Message('note_off', channel=channel, note=note,
                              velocity=0, time=0)
            add_event(closing, absolute_tick, key, synthesized=True)

    order = sorted(lanes, key=lambda key: (first_seen[key], key[0],
                                           key[1] if key[1] is not None
                                           else -1))
    return [lanes[key] for key in order]


def split_instrument_lanes(midifile):
    """Split every track of a MIDI file into instrument lanes.

    Messages from different source tracks that share the same channel
    and program are merged into one lane; each event keeps a reference
    to its source track.

    Parameters
    ----------
    midifile:
        A :class:`MidiFile` of type 0 or 1. Type 2 (asynchronous) files
        have no common tempo reference and cannot be split as a whole;
        call :func:`split_track_lanes` on each track instead.

    Returns
    -------
    list of InstrumentLane
        Lanes ordered by first appearance, then channel and program.

    Raises
    ------
    TypeError:
        For type 2 files.
    ValueError:
        For an out-of-range ``program_change`` program.
    """
    if midifile.type == 2:
        raise TypeError(
            "can't split instrument lanes in type 2 (asynchronous) file;"
            " call split_track_lanes() on each track instead")

    tempo_map = build_tempo_map(midifile.tracks)

    combined = {}
    first_seen = {}
    for track_index, track in enumerate(midifile.tracks):
        lanes = split_track_lanes(
            track,
            track_index=track_index,
            tempo_map=tempo_map,
            ticks_per_beat=midifile.ticks_per_beat,
        )
        for lane in lanes:
            key = (lane.channel, lane.program)
            if key not in combined:
                combined[key] = InstrumentLane(lane.channel, lane.program)
                first_seen[key] = lane.events[0].tick
            combined[key].events.extend(lane.events)

    order = sorted(first_seen, key=lambda key: (first_seen[key], key[0],
                                                key[1] if key[1] is not None
                                                else -1))
    result = []
    for key in order:
        lane = combined[key]
        lane.events.sort(key=lambda event: (event.tick, event.track))
        result.append(lane)
    return result
