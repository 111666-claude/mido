# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Extract audible note intervals from a MidiFile.

This is intended for things like piano roll displays, where what
matters is how long each note actually sounds. A note that is released
while the sustain pedal (CC 64) is held down keeps sounding until the
pedal comes up, so its interval ends at the pedal release instead of
at the key release.

Intervals are returned in absolute ticks, so no tempo information is
needed. (Convert to seconds with tick2second() if required.)

Rules
-----
* Tracks are merged by absolute tick. Messages at the same tick are
  processed in merged stream order (stable sort: earlier tracks and
  earlier positions within a track come first). This also resolves
  same-tick pedal/key conflicts deterministically.
* A note-on with velocity 0 is treated as a note-off.
* A note-on for a (channel, note) that is already sounding ends the
  previous interval at that tick and starts a new one (retriggering).
* A note-off while the sustain pedal is down moves the note to the
  sustained state; it ends when the pedal is released (CC 64 < 64).
* CC 123 (all notes off) and CC 120 (all sound off) end all sounding
  notes on the channel immediately, even sustained ones.
* A note-off with no matching note-on (orphan) is ignored.
* Notes still sounding at the end of the file end at the tick of the
  last event in the file.
* The returned intervals are sorted by (start, end, channel, note,
  velocity), so the order is always stable.
"""

from typing import NamedTuple

from .tracks import merge_tracks

SUSTAIN_PEDAL = 64
ALL_SOUND_OFF = 120
ALL_NOTES_OFF = 123

# CC 64 values at or above this mean the pedal is down.
PEDAL_THRESHOLD = 64


class Note(NamedTuple):
    """An audible note interval in absolute ticks.

    channel, note and velocity are from the note-on message. start is
    the tick of the note-on and end is the tick where the note stops
    sounding (note-off, pedal release, all-notes-off or end of file,
    whichever applies).
    """

    channel: int
    note: int
    velocity: int
    start: int
    end: int

    @property
    def duration(self):
        """Length of the interval in ticks."""
        return self.end - self.start


def get_note_intervals(midifile):
    """Extract audible note intervals from a MidiFile.

    Returns a list of Note namedtuples sorted by (start, end, channel,
    note, velocity). See the module docstring for the exact rules.
    """
    # Merge all tracks into one absolute-tick event stream. The sort
    # in merge_tracks() is stable, so same-tick events keep their
    # original (track, position) order.
    events = []
    now = 0
    for msg in merge_tracks(midifile.tracks, skip_checks=True):
        now += msg.time
        events.append((now, msg))
    end_of_file = now

    notes = []
    # (channel, note) -> [start_tick, velocity]. A key is in exactly
    # one of these at any time: held (key still down) or sustained
    # (key released, pedal still down).
    held = {}
    sustained = {}
    pedal_down = set()

    def close(state, key, tick):
        start, velocity = state.pop(key)
        channel, note = key
        notes.append(Note(channel, note, velocity, start, tick))

    for tick, msg in events:
        if msg.type == 'note_on' and msg.velocity > 0:
            key = (msg.channel, msg.note)
            # Retriggering a sounding key ends the previous interval.
            if key in held:
                close(held, key, tick)
            elif key in sustained:
                close(sustained, key, tick)
            held[key] = (tick, msg.velocity)
        elif msg.type == 'note_off' or msg.type == 'note_on':
            # note_on here means velocity 0, which is a note-off.
            key = (msg.channel, msg.note)
            if key in held:
                if msg.channel in pedal_down:
                    sustained[key] = held.pop(key)
                else:
                    close(held, key, tick)
            # Orphaned note-offs are ignored.
        elif msg.type == 'control_change':
            if msg.control == SUSTAIN_PEDAL:
                if msg.value >= PEDAL_THRESHOLD:
                    pedal_down.add(msg.channel)
                else:
                    pedal_down.discard(msg.channel)
                    # Pedal release ends everything sustained on
                    # this channel.
                    for key in [k for k in sustained
                                if k[0] == msg.channel]:
                        close(sustained, key, tick)
            elif msg.control in (ALL_SOUND_OFF, ALL_NOTES_OFF):
                # Ends all sounding notes on the channel at once,
                # overriding the sustain pedal.
                for state in (held, sustained):
                    for key in [k for k in state if k[0] == msg.channel]:
                        close(state, key, tick)

    # Notes still sounding at end of file end at the last tick.
    for state in (held, sustained):
        for (channel, note), (start, velocity) in state.items():
            notes.append(Note(channel, note, velocity, start, end_of_file))

    notes.sort(key=lambda n: (n.start, n.end, n.channel, n.note, n.velocity))
    return notes
