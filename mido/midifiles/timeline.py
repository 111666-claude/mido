# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

"""
Note sounding timeline extraction.

This module turns the messages of a :class:`~mido.MidiFile` into a
timeline of *sounding* notes. Unlike the raw ``note_on`` / ``note_off``
events stored in the tracks, a note is considered to sound until its
key has been released *and* the sustain pedal (control change 64) is
up, which is what a piano roll should display.

Time conversion follows the file's tempo map: every message carries a
delta time in ticks, and the absolute tick of each event is converted
to seconds using the ``set_tempo`` meta messages in effect at that
point of the song.
"""

from collections import deque

from .units import tick2second

# The default tempo is 120 BPM.
# (500000 microseconds per beat (quarter note).)
DEFAULT_TEMPO = 500000

# Sustain pedal (damper pedal) control change number.
SUSTAIN_CONTROL = 64

# Values at or above this are considered "pedal down".
# (See the MIDI specification, Control Change Messages.)
PEDAL_DOWN_THRESHOLD = 64

# Possible values of SoundingNote.end_reason.
END_REASON_NOTE_OFF = 'note_off'
END_REASON_END_OF_TRACK = 'end_of_track'


class SoundingNote:
    """A note as it actually sounds, including sustain pedal time.

    Attributes:
        track: index of the track the note was played in.
        channel: MIDI channel (0..15).
        note: note pitch (MIDI note number, 0..127).
        velocity: note-on velocity (1..127).
        start_time: absolute start time in seconds.
        end_time: absolute end time in seconds. Equals ``start_time``
            when the note could not be given any duration.
        end_reason: why the note ended. One of
            :data:`END_REASON_NOTE_OFF` (ended by a note-off, possibly
            delayed until the sustain pedal was released) or
            :data:`END_REASON_END_OF_TRACK` (still sounding when the
            file ended, so it was closed at the end of the file).
    """

    __slots__ = [
        'track',
        'channel',
        'note',
        'velocity',
        'start_time',
        'end_time',
        'end_reason',
    ]

    def __init__(self, track, channel, note, velocity, start_time,
                 end_time, end_reason):
        self.track = track
        self.channel = channel
        self.note = note
        self.velocity = velocity
        self.start_time = start_time
        self.end_time = end_time
        self.end_reason = end_reason

    def __eq__(self, other):
        if not isinstance(other, SoundingNote):
            return NotImplemented
        return (
            self.track == other.track
            and self.channel == other.channel
            and self.note == other.note
            and self.velocity == other.velocity
            and self.start_time == other.start_time
            and self.end_time == other.end_time
            and self.end_reason == other.end_reason
        )

    def __repr__(self):
        return (
            'SoundingNote(track={}, channel={}, note={}, velocity={}, '
            'start_time={}, end_time={}, end_reason={!r})'.format(
                self.track,
                self.channel,
                self.note,
                self.velocity,
                self.start_time,
                self.end_time,
                self.end_reason,
            )
        )


class NoteTimeline(list):
    """A list of :class:`SoundingNote` objects.

    The notes are ordered by start time, then end time, then track,
    channel and note.

    ``unmatched_note_offs`` contains one
    ``(track, channel, note, time)`` tuple for every note-off (or
    zero-velocity note-on) that had no matching note-on. ``time`` is
    given in absolute seconds. These events do not produce notes.
    """

    def __init__(self, notes=(), unmatched_note_offs=()):
        super().__init__(notes)
        self.unmatched_note_offs = list(unmatched_note_offs)


def _validate_ticks_per_beat(ticks_per_beat):
    if not isinstance(ticks_per_beat, int) or ticks_per_beat <= 0:
        raise ValueError(
            'invalid ticks_per_beat {!r}: must be a positive integer '
            '(pulses per quarter note)'.format(ticks_per_beat))


def get_note_timeline(midifile):
    """Extract the note sounding timeline from a MIDI file.

    Returns a :class:`NoteTimeline` containing one
    :class:`SoundingNote` per note-on, with absolute start and end
    times in seconds.

    Pairing rules (applied per channel and note pitch):

    * A ``note_on`` with a non-zero velocity starts a note.
    * A ``note_off`` and a ``note_on`` with velocity 0 both act as
      note-offs. If several identical notes overlap, note-offs are
      paired with the oldest note first (first in, first out).
    * Control change 64 (sustain pedal) values of 64..127 mean pedal
      down; values of 0..63 mean pedal up. While the pedal is down,
      released notes keep sounding. They all end when the pedal is
      released; notes whose keys are still held are not affected.
    * Notes still sounding at the end of the file end at the end of
      the file with ``end_reason='end_of_track'``.
    * Note-offs without a matching note-on are collected in
      ``NoteTimeline.unmatched_note_offs`` and otherwise ignored.

    Only type 0 and type 1 files are supported, since type 2 files
    have tracks that are not synchronized. The input ``MidiFile`` is
    never modified.

    Raises:
        ValueError: for type 2 files, invalid
            ``ticks_per_beat`` / PPQ values, or negative absolute
            event times.
    """
    if midifile.type == 2:
        raise ValueError(
            "can't extract a note timeline from a type 2 "
            "(asynchronous) file")

    ticks_per_beat = midifile.ticks_per_beat
    _validate_ticks_per_beat(ticks_per_beat)

    # Collect every event as (absolute tick, track index, message) and
    # remember the end of the longest track. Messages are not copied
    # or mutated, so the input file is left untouched.
    timed_events = []
    end_tick = 0
    for track_index, track in enumerate(midifile.tracks):
        absolute_tick = 0
        for message in track:
            absolute_tick += message.time
            if absolute_tick < 0:
                raise ValueError(
                    'negative absolute time in track {} at message {!r}'
                    .format(track_index, message))
            timed_events.append((absolute_tick, track_index, message))
        if absolute_tick > end_tick:
            end_tick = absolute_tick

    # Sorting by tick keeps the original (track) order for events at
    # the same tick, which makes pairing deterministic.
    timed_events.sort(key=lambda event: (event[0], event[1]))

    # Build the tempo map as (tick, seconds_at_tick, tempo_after)
    # points. Tempo changes at the same tick are applied in playback
    # order, so the last one wins, matching MidiFile iteration.
    tempo_points = [(0, 0.0, DEFAULT_TEMPO)]
    tempo = DEFAULT_TEMPO
    last_tick = 0
    seconds = 0.0
    for tick, _track_index, message in timed_events:
        if tick > last_tick:
            seconds += tick2second(tick - last_tick, ticks_per_beat, tempo)
            last_tick = tick
        if message.type == 'set_tempo':
            tempo = message.tempo
            if tick == tempo_points[-1][0]:
                point_tick, point_seconds, _old_tempo = tempo_points[-1]
                tempo_points[-1] = (point_tick, point_seconds, tempo)
            else:
                tempo_points.append((tick, seconds, tempo))

    point_index = 0

    def to_seconds(tick):
        # Events are visited in chronological order, so the segment
        # cursor only ever moves forward.
        nonlocal point_index
        while (point_index + 1 < len(tempo_points)
               and tempo_points[point_index + 1][0] <= tick):
            point_index += 1
        segment_tick, segment_seconds, segment_tempo = \
            tempo_points[point_index]
        return (segment_seconds
                + tick2second(tick - segment_tick,
                              ticks_per_beat,
                              segment_tempo))

    # Per (channel, note) key:
    #   active    = deque of notes whose keys are still held
    #   sustained = deque of notes whose keys were released while the
    #               pedal was down
    active = {}
    sustained = {}
    # Per channel: whether the sustain pedal is currently down.
    pedal_down = {}

    notes = []
    unmatched_note_offs = []

    def release(key, track_index, tick):
        """Pair a note-off with the oldest matching note."""
        channel, note = key
        queue = active.get(key)
        if queue:
            sounding_note = queue.popleft()
            if not queue:
                del active[key]
            if pedal_down.get(channel, False):
                sustained.setdefault(key, deque()).append(sounding_note)
            else:
                sounding_note.end_time = to_seconds(tick)
                sounding_note.end_reason = END_REASON_NOTE_OFF
                notes.append(sounding_note)
        else:
            unmatched_note_offs.append(
                (track_index, channel, note, to_seconds(tick)))

    for tick, track_index, message in timed_events:
        if message.type == 'note_on' and message.velocity > 0:
            key = (message.channel, message.note)
            active.setdefault(key, deque()).append(
                SoundingNote(
                    track=track_index,
                    channel=message.channel,
                    note=message.note,
                    velocity=message.velocity,
                    start_time=to_seconds(tick),
                    end_time=None,
                    end_reason=None,
                ))

        elif (message.type == 'note_off'
              or (message.type == 'note_on' and message.velocity == 0)):
            release((message.channel, message.note), track_index, tick)

        elif (message.type == 'control_change'
              and message.control == SUSTAIN_CONTROL):
            channel = message.channel
            is_down = message.value >= PEDAL_DOWN_THRESHOLD
            if is_down:
                pedal_down[channel] = True
            elif pedal_down.get(channel, False):
                pedal_down[channel] = False
                # Pedal released: every note whose key was released
                # while the pedal was down stops sounding now. Notes
                # whose keys are still held (in `active`) are untouched.
                for key, queue in list(sustained.items()):
                    if key[0] != channel:
                        continue
                    while queue:
                        sounding_note = queue.popleft()
                        sounding_note.end_time = to_seconds(tick)
                        sounding_note.end_reason = END_REASON_NOTE_OFF
                        notes.append(sounding_note)
                    del sustained[key]

    # Close everything still sounding at the end of the file.
    end_seconds = to_seconds(end_tick)
    for queue in list(active.values()) + list(sustained.values()):
        while queue:
            sounding_note = queue.popleft()
            sounding_note.end_time = end_seconds
            sounding_note.end_reason = END_REASON_END_OF_TRACK
            notes.append(sounding_note)

    notes.sort(key=lambda n: (n.start_time, n.end_time, n.track,
                              n.channel, n.note))

    # Diagnostics should also be in a stable order.
    unmatched_note_offs.sort(key=lambda event: (event[3], event[0],
                                                event[1], event[2]))

    return NoteTimeline(notes, unmatched_note_offs)
