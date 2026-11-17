# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Controller state chasing for MIDI files.

When playback starts in the middle of a song the receiving instrument
still has the sounds and controller values from the previous section
(or from whatever was played last).  To make the new position sound
right, the controller state that is in effect there has to be sent
again before playback resumes.  This is traditionally called *chasing*
events.

The chase is expressed in absolute ticks: every message in every track
is stamped with its absolute tick and the state is collected from the
messages strictly before the target tick.  Messages exactly at the
target tick are not included, just like notes at the target are left
for normal playback.  No note message is ever chased.

The state that is reconstructed is:

* bank select (control change 0 and 32) and program change,
* ordinary control changes (the latest value of each controller),
* the current RPN/NRPN selection and its last value,
* pitch bend, and
* channel pressure (``aftertouch``).

The output is a :class:`~mido.midifiles.tracks.MidiTrack` of messages
with ``time=0``, in a deterministic order.  Channels are emitted in
ascending order.  Inside a channel the order is bank select MSB, bank
select LSB, program change, RPN/NRPN selection and value, the
remaining controllers in ascending control-number order, pitch bend
and finally channel pressure.

Messages on different tracks that share an absolute tick are ordered
by track index and then by position inside the track, so the result
does not depend on dictionary or set iteration order.  When events
conflict, the last one in that order wins.

Malformed and edge-case input is handled deterministically:

* A file without ``set_tempo`` messages works normally.  Chasing is
  tick based and does not need a tempo.
* Empty files and empty tracks produce no messages.
* A target beyond the end of the file returns the final state.
  Negative and non-integer targets are rejected, as are tracks whose
  messages add up to a negative absolute tick.
* RPN/NRPN selectors only take effect after both the MSB and LSB have
  been seen on the same channel.  Selection, data entry and the RPN
  null reset are kept per channel and never leak between channels.
  Changing the selection clears the pending data entry registers.  Only
  the currently selected parameter's value is re-sent, since that is
  the register the receiver has selected.
* Data entry (control change 6/38/96/97) without a complete selection
  is ignored.  A data entry LSB without a preceding MSB is ignored as
  well.  An increment/decrement with no previously recorded absolute
  value is ignored, since the resulting value depends on an
  instrument-specific default.
* Channel mode messages (control change 120..127) are commands, not
  state, and are not chased.  ``Reset All Controllers`` (121) sets
  modulation to 0, expression to 127, sustain to 0, centers pitch bend
  and zeroes channel pressure; volume, pan, bank select, program and
  the RPN/NRPN selection keep their values (as specified by RP-015).
* Type 2 (asynchronous) files cannot be chased because their tracks
  have no common timeline.
"""

from numbers import Integral

from ..messages import Message
from .tracks import MidiTrack

# Controller numbers used by the chase.
BANK_SELECT_MSB = 0
BANK_SELECT_LSB = 32
DATA_ENTRY_MSB = 6
DATA_ENTRY_LSB = 38
DATA_INCREMENT = 96
DATA_DECREMENT = 97
NRPN_LSB = 98
NRPN_MSB = 99
RPN_LSB = 100
RPN_MSB = 101
RESET_ALL_CONTROLLERS = 121

#: The RPN null selector (MSB and LSB both 127) deselects the current
#: Registered Parameter Number.
RPN_NULL = 0x3FFF

MAX_7BIT = 0x7F
MAX_14BIT = 0x3FFF

# Channel mode messages are commands (all notes off, omni/poly mode,
# ...).  They are not restored as controller values.
_CHANNEL_MODE_CONTROLS = frozenset(range(120, 128))

# Controllers whose value is defined after Reset All Controllers
# (CC 121, MIDI 1.0 RP-015).  The specification deliberately leaves
# volume, pan, bank select, program and the RPN/NRPN selection
# untouched, so they keep their values.
RESET_CONTROLLER_VALUES = {
    1: 0,     # modulation wheel
    11: 127,  # expression
    64: 0,    # sustain / hold pedal
}


def _check_tick(tick):
    if isinstance(tick, bool) or not isinstance(tick, Integral):
        raise TypeError(f'chase tick must be an int, got {tick!r}')
    if tick < 0:
        raise ValueError(f'chase tick must be non-negative, got {tick!r}')


def _cc(channel, control, value):
    return Message('control_change', channel=channel,
                   control=control, value=value, time=0)


class ChannelChaseState:
    """The chased state of one MIDI channel.

    Attributes:

    ``channel``
        MIDI channel number (0..15).

    ``bank_msb`` / ``bank_lsb``
        Latest bank select values (control change 0 and 32), or
        ``None`` if the file never set that half on this channel.

    ``program``
        Latest program number, or ``None`` if the file never sent a
        program change on this channel.

    ``controllers``
        Latest value of every ordinary controller as
        ``{control: value}``.  Bank selects, RPN/NRPN controls and
        channel mode messages are not included.

    ``pitchwheel``
        Latest pitch bend value (-8192..8191), or ``None``.

    ``channel_pressure``
        Latest channel pressure (``aftertouch``) value (0..127), or
        ``None``.

    ``selection``
        The currently selected parameter as ``(kind, number)``, where
        ``kind`` is ``'rpn'`` or ``'nrpn'``.  ``None`` means no
        complete selection was seen.  The RPN null selection is
        represented as ``('rpn', 16383)``.

    ``parameters``
        Last value of each parameter that received a data entry, as
        ``{(kind, number): (value, width)}``.  ``width`` is 7 or 14
        and tells how many bits the message sequence used.
    """

    def __init__(self, channel):
        _check_channel_number(channel)
        self.channel = channel
        self.bank_msb = None
        self.bank_lsb = None
        self.program = None
        self.controllers = {}
        self.pitchwheel = None
        self.channel_pressure = None
        self.selection = None
        self.parameters = {}

        # RPN and NRPN selectors are independent register pairs.  Both
        # halves must have been seen before they select a parameter.
        self._rpn_msb = None
        self._rpn_lsb = None
        self._nrpn_msb = None
        self._nrpn_lsb = None

        # Data entry registers.  They are reset on a selection change so
        # a stale MSB can never be combined with a new parameter's LSB.
        self._data_msb = None
        self._data_lsb = None

    # -- State updates -------------------------------------------------

    def update(self, msg):
        """Update the state with one message.

        Meta, note, sysex and system messages are ignored.
        """
        if msg.is_meta:
            return

        if msg.type == 'control_change':
            self._update_control_change(msg.control, msg.value)
        elif msg.type == 'program_change':
            self.program = msg.program
        elif msg.type == 'pitchwheel':
            self.pitchwheel = msg.pitch
        elif msg.type == 'aftertouch':
            self.channel_pressure = msg.value

    def _update_control_change(self, control, value):
        if control == BANK_SELECT_MSB:
            self.bank_msb = value
        elif control == BANK_SELECT_LSB:
            self.bank_lsb = value
        elif control == RPN_MSB:
            self._rpn_msb = value
            self._update_selection('rpn')
        elif control == RPN_LSB:
            self._rpn_lsb = value
            self._update_selection('rpn')
        elif control == NRPN_MSB:
            self._nrpn_msb = value
            self._update_selection('nrpn')
        elif control == NRPN_LSB:
            self._nrpn_lsb = value
            self._update_selection('nrpn')
        elif control == DATA_ENTRY_MSB:
            self._update_data_entry_msb(value)
        elif control == DATA_ENTRY_LSB:
            self._update_data_entry_lsb(value)
        elif control == DATA_INCREMENT:
            self._update_relative(value)
        elif control == DATA_DECREMENT:
            self._update_relative(-value)
        elif control == RESET_ALL_CONTROLLERS:
            self._reset_all_controllers()
        elif control in _CHANNEL_MODE_CONTROLS:
            return
        else:
            self.controllers[control] = value

    def _update_selection(self, kind):
        if kind == 'rpn':
            msb, lsb = self._rpn_msb, self._rpn_lsb
        else:
            msb, lsb = self._nrpn_msb, self._nrpn_lsb

        if msb is None or lsb is None:
            return

        number = (msb << 7) | lsb
        if self.selection == (kind, number):
            return

        self.selection = (kind, number)
        self._data_msb = None
        self._data_lsb = None

    def _active_parameter(self):
        """Return the selected parameter, or None for no/null selection."""
        if self.selection is None:
            return None
        if self.selection == ('rpn', RPN_NULL):
            return None
        return self.selection

    def _update_data_entry_msb(self, value):
        parameter = self._active_parameter()
        if parameter is None:
            return
        self._data_msb = value
        self._data_lsb = None
        self.parameters[parameter] = (value, 7)

    def _update_data_entry_lsb(self, value):
        parameter = self._active_parameter()
        if parameter is None or self._data_msb is None:
            return
        self._data_lsb = value
        self.parameters[parameter] = (
            (self._data_msb << 7) | value, 14)

    def _update_relative(self, delta):
        parameter = self._active_parameter()
        if parameter is None:
            return

        if parameter not in self.parameters:
            # Data increment/decrement is relative.  Without an
            # absolute value to apply it to, the resulting value is
            # unknown, so nothing is recorded rather than inventing a
            # baseline (the receiver's default is instrument
            # specific).
            return

        old_value, old_width = self.parameters[parameter]
        maximum = MAX_7BIT if old_width == 7 else MAX_14BIT
        value = min(max(old_value + delta, 0), maximum)
        self.parameters[parameter] = (value, old_width)
        self._data_msb = None
        self._data_lsb = None

    def _reset_all_controllers(self):
        # Reset All Controllers (RP-015) only resets controllers with a
        # defined reset value, plus pitch bend and channel pressure.
        # Bank select, program change, volume, pan and the RPN/NRPN
        # selection are explicitly left alone.
        for control, value in RESET_CONTROLLER_VALUES.items():
            self.controllers[control] = value
        self.pitchwheel = 0
        self.channel_pressure = 0

    # -- Output --------------------------------------------------------

    def messages(self):
        """Return the messages that recreate this channel's state.

        The returned messages have ``time=0`` and are ordered so that a
        bank select always comes before the program change it applies
        to.  Only bank select halves that the file actually sent are
        re-sent; a half it never touched is left at the receiver's
        value.
        """
        messages = []

        if self.bank_msb is not None:
            messages.append(
                _cc(self.channel, BANK_SELECT_MSB, self.bank_msb))
        if self.bank_lsb is not None:
            messages.append(
                _cc(self.channel, BANK_SELECT_LSB, self.bank_lsb))
        if self.program is not None:
            messages.append(Message('program_change', channel=self.channel,
                                    program=self.program, time=0))

        messages.extend(self._selection_messages())

        for control in sorted(self.controllers):
            messages.append(
                _cc(self.channel, control, self.controllers[control]))

        if self.pitchwheel is not None:
            messages.append(Message('pitchwheel', channel=self.channel,
                                    pitch=self.pitchwheel, time=0))
        if self.channel_pressure is not None:
            messages.append(Message('aftertouch', channel=self.channel,
                                    value=self.channel_pressure, time=0))

        return messages

    def _selection_messages(self):
        if self.selection is None:
            return []

        kind, number = self.selection

        if kind == 'rpn':
            msb_control, lsb_control = RPN_MSB, RPN_LSB
        else:
            msb_control, lsb_control = NRPN_MSB, NRPN_LSB

        if self.selection == ('rpn', RPN_NULL):
            # Reproduce the reset so the receiver does not keep an old
            # RPN selected.
            return [
                _cc(self.channel, RPN_MSB, MAX_7BIT),
                _cc(self.channel, RPN_LSB, MAX_7BIT),
            ]

        messages = [
            _cc(self.channel, msb_control, (number >> 7) & MAX_7BIT),
            _cc(self.channel, lsb_control, number & MAX_7BIT),
        ]

        entry = self.parameters.get(self.selection)
        if entry is not None:
            value, width = entry
            if width == 14:
                messages.append(
                    _cc(self.channel, DATA_ENTRY_MSB,
                        (value >> 7) & MAX_7BIT))
                messages.append(
                    _cc(self.channel, DATA_ENTRY_LSB, value & MAX_7BIT))
            else:
                messages.append(
                    _cc(self.channel, DATA_ENTRY_MSB, value))

        return messages

    # -- Convenience ---------------------------------------------------

    def copy(self):
        """Return an independent copy of the state."""
        other = self.__class__(self.channel)
        other.bank_msb = self.bank_msb
        other.bank_lsb = self.bank_lsb
        other.program = self.program
        other.controllers = dict(self.controllers)
        other.pitchwheel = self.pitchwheel
        other.channel_pressure = self.channel_pressure
        other.selection = self.selection
        other.parameters = dict(self.parameters)
        other._rpn_msb = self._rpn_msb
        other._rpn_lsb = self._rpn_lsb
        other._nrpn_msb = self._nrpn_msb
        other._nrpn_lsb = self._nrpn_lsb
        other._data_msb = self._data_msb
        other._data_lsb = self._data_lsb
        return other

    def __eq__(self, other):
        if not isinstance(other, ChannelChaseState):
            return NotImplemented
        return vars(self) == vars(other)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(channel={self.channel!r}, '
            f'bank_msb={self.bank_msb!r}, bank_lsb={self.bank_lsb!r}, '
            f'program={self.program!r}, controllers={self.controllers!r}, '
            f'pitchwheel={self.pitchwheel!r}, '
            f'channel_pressure={self.channel_pressure!r}, '
            f'selection={self.selection!r}, '
            f'parameters={self.parameters!r})'
        )


class ChaseState:
    """The controller state of a file at an absolute tick.

    Attributes:

    ``tick``
        The absolute target tick.

    ``channels``
        A dictionary mapping channel numbers to
        :class:`ChannelChaseState` objects.  Only channels that were
        touched by a message before the target tick are present.
    """

    def __init__(self, tick, channels=None):
        _check_tick(tick)
        self.tick = tick
        self.channels = dict(channels or {})

    def __getitem__(self, channel):
        return self.channels[channel]

    def __contains__(self, channel):
        return channel in self.channels

    def __iter__(self):
        return iter(sorted(self.channels))

    def __len__(self):
        return len(self.channels)

    def messages(self):
        """Return the chased messages for all channels as a MidiTrack."""
        messages = MidiTrack()
        for channel in sorted(self.channels):
            messages.extend(self.channels[channel].messages())
        return messages

    def copy(self):
        """Return an independent copy of the state."""
        return self.__class__(
            self.tick,
            {channel: state.copy()
             for channel, state in self.channels.items()},
        )

    def __eq__(self, other):
        if not isinstance(other, ChaseState):
            return NotImplemented
        return (self.tick == other.tick
                and self.channels == other.channels)

    def __repr__(self):
        return (f'{self.__class__.__name__}(tick={self.tick!r}, '
                f'channels={self.channels!r})')


def _check_channel_number(channel):
    if isinstance(channel, bool) or not isinstance(channel, Integral):
        raise TypeError(f'channel must be an int, got {channel!r}')
    if not 0 <= channel <= 15:
        raise ValueError(
            f'channel must be in range 0..15, got {channel!r}')


def chase_state(midifile, tick):
    """Collect the controller state of a MIDI file at an absolute tick.

    The state is built from every message in every track whose absolute
    tick is strictly less than ``tick``.  Messages on the same tick are
    processed by track index and then by position inside the track, so
    conflicting events have a stable, deterministic winner (the last
    one).

    Returns a :class:`ChaseState`.  Use :func:`chase_events` to get the
    corresponding message sequence directly.

    Raises ``TypeError`` for type 2 (asynchronous) files and for
    non-integer ticks, and ``ValueError`` for negative ticks or for a
    track whose messages add up to a negative absolute tick.
    """
    _check_tick(tick)

    if midifile.type == 2:
        raise TypeError(
            "can't chase controller state in a type 2 "
            "(asynchronous) file")

    # Collect only tracked events before the target.  Sorting by
    # (absolute tick, track index, message index) keeps the order of
    # messages that share a tick stable across tracks.
    tracked_types = frozenset((
        'control_change',
        'program_change',
        'pitchwheel',
        'aftertouch',
    ))
    events = []
    for track_index, track in enumerate(midifile.tracks):
        absolute_tick = 0
        for message_index, msg in enumerate(track):
            absolute_tick += msg.time
            if absolute_tick < 0:
                raise ValueError(
                    f'negative absolute tick in track {track_index} '
                    f'at message {message_index}')
            if absolute_tick < tick and msg.type in tracked_types:
                events.append(
                    (absolute_tick, track_index, message_index, msg))

    events.sort(key=lambda event: event[:3])

    states = {}
    for _absolute_tick, _track_index, _message_index, msg in events:
        channel = msg.channel
        try:
            state = states[channel]
        except KeyError:
            state = ChannelChaseState(channel)
            states[channel] = state

        state.update(msg)

    return ChaseState(tick, states)


def chase_events(midifile, tick):
    """Return the messages that recreate controller state at ``tick``.

    ``tick`` is an absolute tick from the beginning of the file.  The
    returned :class:`~mido.midifiles.tracks.MidiTrack` contains
    ``time=0`` messages that restore each channel's bank select,
    program, controllers, RPN/NRPN selection and value, pitch bend and
    channel pressure.  No note messages are included, so notes that
    started before the target are not retriggered.

    The state is taken from messages strictly before ``tick``; messages
    exactly at ``tick`` are intended for normal playback.  A target
    past the end of the file returns the final state.

    The file is not modified, so the function can be called any number
    of times.

    Raises ``TypeError`` for type 2 (asynchronous) files and for
    non-integer ticks, and ``ValueError`` for negative ticks or for a
    track whose messages add up to a negative absolute tick.
    """
    return chase_state(midifile, tick).messages()


#: Short alias for :func:`chase_events`.  Inside this module it does not
#: clash with anything; at package level the longer name is used so it
#: does not shadow the ``mido.midifiles.chase`` module.
chase = chase_events
