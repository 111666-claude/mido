# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Random-access playback positioning (seeking) for MIDI files.

This module lets a caller jump to an arbitrary position (in seconds)
in a MIDI file without playing through everything before it, for
example when a user drags the playhead of a score editor.

``seek()`` returns a ``PlaybackPosition`` holding:

* the messages at and after the position, in playback order, with
  times in seconds relative to the position, and
* a ``ChannelState`` snapshot of every channel at the position, so
  the caller can restore already sounding notes, the sustain pedal
  and controller values before resuming playback.

Seeking never modifies the file, its tracks or its messages, so it
can be repeated any number of times.
"""

from .midifiles import DEFAULT_TEMPO
from .units import tick2second

# Control change number of the sustain (hold) pedal.
SUSTAIN_PEDAL = 64


class ChannelState:
    """Snapshot of the state of all 16 channels at a playback position.

    Attributes:

    ``notes``
        Currently sounding notes as ``{channel: {note: velocity}}``.
        A note is sounding if it was turned on (``note_on`` with
        velocity > 0) and not yet turned off (``note_off`` or
        ``note_on`` with velocity 0) before the position.

    ``controllers``
        Latest value of each controller as
        ``{channel: {control: value}}``. The sustain pedal is
        controller 64 (``SUSTAIN_PEDAL``).
    """

    def __init__(self, notes=None, controllers=None):
        # Inner dicts are copied so the snapshot is independent of
        # whatever produced it.
        self.notes = {channel: dict(channel_notes)
                      for channel, channel_notes in (notes or {}).items()}
        self.controllers = {channel: dict(channel_controls)
                            for channel, channel_controls
                            in (controllers or {}).items()}

    def update(self, msg):
        """Update the state with one message.

        Only channel voice messages that change the audible state
        (``note_on``, ``note_off`` and ``control_change``) have an
        effect; every other message is ignored.
        """
        if msg.is_meta:
            return

        if msg.type == 'note_on':
            if msg.velocity > 0:
                self.notes.setdefault(msg.channel, {})[msg.note] = msg.velocity
            else:
                # A note_on with velocity 0 is a note off.
                self._note_off(msg.channel, msg.note)
        elif msg.type == 'note_off':
            self._note_off(msg.channel, msg.note)
        elif msg.type == 'control_change':
            self.controllers.setdefault(msg.channel, {})[msg.control] = msg.value

    def _note_off(self, channel, note):
        channel_notes = self.notes.get(channel)
        if channel_notes is not None:
            channel_notes.pop(note, None)
            if not channel_notes:
                del self.notes[channel]

    def active_notes(self, channel=None):
        """Return the currently sounding notes.

        Returns a sorted list of ``(channel, note, velocity)`` tuples.
        If ``channel`` is passed only notes on that channel are
        returned.
        """
        notes = []
        for channel_number, channel_notes in self.notes.items():
            if channel is None or channel_number == channel:
                notes.extend(
                    (channel_number, note, velocity)
                    for note, velocity in channel_notes.items())
        return sorted(notes)

    def control(self, channel, control, default=0):
        """Return the latest value of a controller on a channel.

        Returns ``default`` if the controller was never set.
        """
        return self.controllers.get(channel, {}).get(control, default)

    def is_sustained(self, channel):
        """Return True if the sustain pedal is down on the channel."""
        return self.control(channel, SUSTAIN_PEDAL) >= 64

    def copy(self):
        """Return an independent copy of the state."""
        return ChannelState(self.notes, self.controllers)

    def __eq__(self, other):
        if not isinstance(other, ChannelState):
            return NotImplemented
        return (self.notes == other.notes
                and self.controllers == other.controllers)

    def __repr__(self):
        return (f'{self.__class__.__name__}(notes={self.notes!r}, '
                f'controllers={self.controllers!r})')


class PlaybackPosition:
    """The result of seeking to a position in a MIDI file.

    Attributes:

    ``time``
        The position in seconds.

    ``events``
        The messages at and after the position, in playback order.
        Each message's ``time`` attribute is the number of seconds
        from the position to the message, so the first message has
        ``time >= 0`` and times are non-decreasing. The messages are
        copies and can be modified freely.

    ``state``
        A ``ChannelState`` snapshot of all channels at the position,
        reflecting every message before it. Use it to restore already
        sounding notes, the sustain pedal and controller values before
        resuming playback.
    """

    def __init__(self, time, events, state):
        self.time = time
        self.events = events
        self.state = state

    def __iter__(self):
        return iter(self.events)

    def __len__(self):
        return len(self.events)

    def __repr__(self):
        return (f'{self.__class__.__name__}(time={self.time!r}, '
                f'events={self.events!r}, state={self.state!r})')


def seek(midifile, seconds):
    """Seek to a position in a MIDI file.

    ``seconds`` is a non-negative position in seconds from the start
    of the song. Returns a ``PlaybackPosition``.

    Messages are considered in absolute tick order after merging all
    tracks (tempo changes on any track take effect in that order, and
    the order of messages on the same tick is stable). A message
    exactly at the position is included in ``events`` with ``time ==
    0`` and is not yet reflected in ``state``; seeking to the same
    position always gives the same result.

    Seeking past the end of the song returns no events and the final
    channel state. Seeking a file with no (or only empty) tracks
    returns an empty state and, at most, the end of track message.

    Raises ``ValueError`` if ``seconds`` is negative, and ``TypeError``
    for type 2 (asynchronous) files, whose tracks can not be merged.
    """
    if seconds < 0:
        raise ValueError(f'seek position must be non-negative, got {seconds!r}')
    seconds = float(seconds)

    tempo = DEFAULT_TEMPO
    ticks_per_beat = midifile.ticks_per_beat
    now = 0.0
    state = ChannelState()
    events = []

    # merged_track raises TypeError for type 2 files. It is cached by
    # the MidiFile, so repeated seeks only pay for one pass over it.
    for msg in midifile.merged_track:
        now += tick2second(msg.time, ticks_per_beat, tempo)

        if now < seconds:
            state.update(msg)
        else:
            events.append(msg.copy(skip_checks=True, time=now - seconds))

        if msg.type == 'set_tempo':
            tempo = msg.tempo

    return PlaybackPosition(seconds, events, state)
