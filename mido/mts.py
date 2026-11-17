# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

"""
MIDI Tuning Standard (MTS) support.

This module implements the real-time *Single Note Tuning Change*
universal system exclusive message defined in the MIDI Manufacturers
Association "MIDI Tuning" specification::

    F0 7F <device ID> 08 02 tt ll [kk xx yy zz] ... F7

and the non-real-time variant which uses a ``7E`` header byte instead
of ``7F``.  The message allows the pitch of any subset of the 128 MIDI
key numbers to be adjusted on the fly while a performance is running.

A pitch is carried in a three byte frequency data word:

* the first byte is the equal-tempered semitone (the nearest semitone
  below the frequency),
* the next two bytes form a 14 bit fraction of that semitone in units
  of 100 / 16384 cents (about 0.0061 cents).

The word ``7F 7F 7F`` is reserved by the specification and means
"no change": a receiver must leave the stored tuning of that key
untouched.

References:

* `MIDI Tuning Updated Specification
  <https://midi.org/midi-tuning-updated-specification>`_
* `MIDI Tuning (MMA PDF mirror)
  <https://musescore.org/sites/musescore.org/files/2018-06/midituning.pdf>`_
"""

import math
from numbers import Integral, Real

from .messages import Message

# Universal system exclusive header bytes.
SYSEX_NON_REALTIME = 0x7E
SYSEX_REALTIME = 0x7F

#: Universal SysEx sub-ID#1 for the MIDI Tuning Standard.
MIDI_TUNING_STANDARD = 0x08

#: Universal SysEx sub-ID#2 for Single Note Tuning Change.
SINGLE_NOTE_TUNING_CHANGE = 0x02

#: Device ID that addresses every device.
DEVICE_ID_BROADCAST = 0x7F

#: Frequency data word reserved by the specification to mean "no change".
NO_CHANGE_FREQUENCY = (0x7F, 0x7F, 0x7F)

#: Maximum number of note changes that fit in one message.
MAX_NOTE_CHANGES = 127

#: Highest representable frequency data word (``7F 7F 7E``), in Hz.
MAX_FREQUENCY = 13289.73

#: Frequency of MIDI note 0 derived from A4 = 440 Hz.
NOTE_0_FREQUENCY = 440.0 * 2.0 ** (-69.0 / 12.0)

_FRACTION_STEPS = 1 << 14  # 14 bits, 16384 fractions of a semitone.


class NoteTuning:
    """A single key-to-pitch tuning change.

    ``note`` is the MIDI key number (0..127) that is retuned and
    ``frequency`` is the target pitch in hertz.  ``frequency`` is
    ``None`` when the frequency data word is the reserved "no change"
    value (``7F 7F 7F``).
    """

    __slots__ = ('frequency', 'note')

    def __init__(self, note, frequency):
        self.note = note
        self.frequency = frequency

    def __eq__(self, other):
        if not isinstance(other, NoteTuning):
            return NotImplemented
        return (self.note == other.note
                and _floats_equal(self.frequency, other.frequency))

    def __repr__(self):
        return 'NoteTuning(note={!r}, frequency={!r})'.format(
            self.note, self.frequency)

    def __hash__(self):
        return hash((self.note, self.frequency))


class SingleNoteTuningChange:
    """A decoded Single Note Tuning Change message.

    Attributes:

    * ``realtime``: ``True`` for a real-time message (header ``7F``),
      ``False`` for the non-real-time setup variant (header ``7E``).
    * ``device_id``: target device ID (0..127, ``0x7F`` broadcasts to
      all devices).
    * ``program``: tuning program number (0..127).
    * ``changes``: tuple of :class:`NoteTuning` entries, in the order
      they appear in the message.
    * ``time``: the message time, as carried by the source
      :class:`mido.Message`.
    """

    __slots__ = ('changes', 'device_id', 'program', 'realtime', 'time')

    def __init__(self, realtime, device_id, program, changes, time=0):
        self.realtime = realtime
        self.device_id = device_id
        self.program = program
        self.changes = tuple(changes)
        self.time = time

    def __eq__(self, other):
        if not isinstance(other, SingleNoteTuningChange):
            return NotImplemented
        return all(getattr(self, name) == getattr(other, name)
                   for name in self.__slots__)

    def __repr__(self):
        return ('SingleNoteTuningChange(realtime={!r}, device_id={!r}, '
                'program={!r}, changes={!r}, time={!r})').format(
                    self.realtime, self.device_id, self.program,
                    self.changes, self.time)


def _floats_equal(a, b):
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)


def _check_7bit(name, value):
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError('{} must be an integer'.format(name))
    if not 0 <= value <= 127:
        raise ValueError('{} must be in range 0..127'.format(name))


def _check_note(value):
    _check_7bit('note number', value)


def _check_frequency(frequency):
    if isinstance(frequency, bool) or not isinstance(frequency, Real):
        raise TypeError('frequency must be a number of hertz')
    frequency = float(frequency)
    if math.isnan(frequency) or math.isinf(frequency):
        raise ValueError('frequency must be a finite number')
    if frequency <= 0:
        raise ValueError('frequency must be greater than zero')
    return frequency


def frequency_to_data(frequency):
    """Encode a frequency in hertz as a MTS 3 byte frequency data word.

    Returns a tuple ``(semitone, fraction_msb, fraction_lsb)`` of 7 bit
    integers.  The frequency is rounded to the nearest representable
    value (about 0.0061 cents resolution) using round-half-to-even.

    Raises ``ValueError`` for non-positive, non-finite or
    non-representable frequencies (the representable range ends at the
    ``7F 7F 7E`` word, 13289.73 Hz; ``7F 7F 7F`` is reserved).
    """
    frequency = _check_frequency(frequency)

    # Continuous MIDI note number relative to A440.
    note_number = 69.0 + 12.0 * math.log2(frequency / 440.0)
    semitone = math.floor(note_number)

    # Fraction of the semitone, scaled to 14 bits.  Computing it from
    # the frequency ratio (instead of the fractional part of
    # note_number) avoids precision loss for high semitones.
    semitone_frequency = 440.0 * 2.0 ** ((semitone - 69.0) / 12.0)
    if frequency == semitone_frequency:
        fraction = 0.0
    else:
        cents = 1200.0 * math.log2(frequency / semitone_frequency)
        fraction = round(cents / 100.0 * _FRACTION_STEPS)

    if fraction == _FRACTION_STEPS:
        # Rounding reached the next semitone.
        fraction = 0
        semitone += 1

    if not 0 <= semitone <= 127:
        raise ValueError(
            'frequency {} Hz is outside the representable MTS range '
            '({:.4f} Hz..{:.2f} Hz)'.format(
                frequency, NOTE_0_FREQUENCY, MAX_FREQUENCY))

    msb, lsb = divmod(int(fraction), 128)
    word = (semitone, msb, lsb)

    if word == NO_CHANGE_FREQUENCY:
        # 7F 7F 7F is reserved and never represents a frequency.
        raise ValueError(
            'frequency {} Hz encodes to the reserved word 7F 7F 7F; '
            'use no_change=True to request "no change"'.format(frequency))

    return word


def data_to_frequency(data):
    """Decode a MTS 3 byte frequency data word to a frequency in hertz.

    ``data`` is a sequence of three 7 bit integers
    ``(semitone, fraction_msb, fraction_lsb)``.  Returns ``None`` for
    the reserved "no change" word (``7F 7F 7F``).

    Raises ``ValueError`` if any byte is outside the 0..127 range.
    """
    data = tuple(data)
    if len(data) != 3:
        raise ValueError('frequency data word must contain exactly 3 bytes')
    for index, byte in enumerate(data):
        if not isinstance(byte, Integral) or isinstance(byte, bool):
            raise TypeError(
                'frequency data byte {} must be an integer'.format(index))
        if not 0 <= byte <= 127:
            raise ValueError(
                'frequency data byte {} must be in range 0..127'.format(index))

    if data == NO_CHANGE_FREQUENCY:
        return None

    semitone, msb, lsb = data
    fraction = (msb << 7) | lsb
    note_number = semitone + fraction / float(_FRACTION_STEPS)
    return NOTE_0_FREQUENCY * 2.0 ** (note_number / 12.0)


def _coerce_change(change):
    """Return a (note, frequency) pair, allowing frequency=None."""
    if isinstance(change, NoteTuning):
        return change.note, change.frequency

    try:
        note, frequency = change
    except (TypeError, ValueError) as error:
        raise TypeError(
            'each tuning change must be a (note, frequency) pair or a '
            'NoteTuning object') from error

    return note, frequency


def single_note_tuning_change(changes, device_id=DEVICE_ID_BROADCAST,
                              program=0, realtime=True, time=0,
                              no_change=()):
    """Build a MTS Single Note Tuning Change sysex message.

    ``changes`` is a non-empty sequence of tuning changes.  Each change
    is either a :class:`NoteTuning` or a ``(note, frequency)`` pair
    where ``note`` is a MIDI key number (0..127) and ``frequency`` is
    the target pitch in hertz.  A frequency of ``None`` (or listing the
    note in ``no_change``) emits the reserved ``7F 7F 7F`` word that
    tells receivers to leave that key's tuning untouched.

    ``device_id`` selects the target device (0..127, default ``0x7F``
    which is broadcast to all devices) and ``program`` selects the
    tuning program (0..127).  Pass ``realtime=False`` to produce the
    non-real-time setup variant (header ``7E`` instead of ``7F``).

    At most :data:`MAX_NOTE_CHANGES` (127) changes fit in one message
    and note numbers may not be repeated.

    Returns a :class:`mido.Message` of type ``'sysex'`` whose ``time``
    attribute is set to ``time`` so it can be placed directly in a
    :class:`mido.MidiTrack`.
    """
    _check_7bit('device_id', device_id)
    _check_7bit('program', program)
    if not isinstance(realtime, bool):
        raise TypeError('realtime must be True or False')
    if not isinstance(no_change, (set, frozenset, tuple, list)):
        raise TypeError('no_change must be a sequence of note numbers')
    no_change = set(no_change)
    for note in no_change:
        _check_note(note)

    entries = list(changes)
    if len(entries) == 0:
        raise ValueError('at least one tuning change is required')
    if len(entries) > MAX_NOTE_CHANGES:
        raise ValueError(
            'a Single Note Tuning Change message can hold at most {} '
            'note changes, got {}'.format(MAX_NOTE_CHANGES, len(entries)))

    payload = [
        SYSEX_REALTIME if realtime else SYSEX_NON_REALTIME,
        device_id,
        MIDI_TUNING_STANDARD,
        SINGLE_NOTE_TUNING_CHANGE,
        program,
        len(entries),
    ]

    seen = set()
    for entry in entries:
        note, frequency = _coerce_change(entry)
        _check_note(note)
        if note in seen:
            raise ValueError(
                'duplicate note number {} in tuning changes'.format(note))
        seen.add(note)

        payload.append(note)
        if frequency is None or note in no_change:
            payload.extend(NO_CHANGE_FREQUENCY)
        else:
            payload.extend(frequency_to_data(frequency))

    return Message('sysex', data=payload, time=time)


def parse_single_note_tuning_change(message):
    """Decode a MTS Single Note Tuning Change sysex message.

    Accepts a :class:`mido.Message` of type ``'sysex'`` (or the raw
    sysex data bytes) and returns a :class:`SingleNoteTuningChange`.
    Both the real-time (``7F``) and non-real-time (``7E``) header
    variants are accepted.

    Raises ``ValueError`` if the message is not a Single Note Tuning
    Change message, is truncated, declares a change count that does not
    match its payload, contains non 7 bit data, holds no changes or more
    than 127 changes, or repeats a key number.
    """
    if isinstance(message, Message):
        if message.type != 'sysex':
            raise ValueError(
                'expected a sysex message, got {!r}'.format(message.type))
        data = tuple(message.data)
        time = message.time
    else:
        data = tuple(message)
        time = 0

    # Strip an optional F0/F7 wrapper so raw message bytes work too.
    # A mido sysex message stores the payload without either framing
    # byte.  Raw bytes that start with F0 must end with F7, otherwise
    # the message is truncated.
    if data and data[0] == 0xF0:
        if not data or data[-1] != 0xF7:
            raise ValueError(
                'truncated Single Note Tuning Change message: '
                'missing end byte F7')
        data = data[1:-1]

    if len(data) < 6:
        raise ValueError(
            'truncated Single Note Tuning Change message: need at least '
            '6 header bytes, got {}'.format(len(data)))

    header, device_id, sub_id_1, sub_id_2, program, count = data[:6]

    if header not in (SYSEX_REALTIME, SYSEX_NON_REALTIME):
        raise ValueError(
            'not a universal system exclusive message '
            '(header byte 0x{:02X}, expected 7E or 7F)'.format(header))
    if sub_id_1 != MIDI_TUNING_STANDARD:
        raise ValueError(
            'not a MIDI Tuning Standard message (sub-ID#1 0x{:02X}, '
            'expected 08)'.format(sub_id_1))
    if sub_id_2 != SINGLE_NOTE_TUNING_CHANGE:
        raise ValueError(
            'not a Single Note Tuning Change message (sub-ID#2 0x{:02X}, '
            'expected 02)'.format(sub_id_2))

    for name, value in (('device ID', device_id), ('tuning program',
                                                   program),
                        ('number of changes', count)):
        if not 0 <= value <= 127:
            raise ValueError(
                '{} byte 0x{:02X} is not valid 7 bit data'.format(
                    name, value))

    if count == 0:
        raise ValueError('Single Note Tuning Change message contains no '
                         'note changes')
    if count > MAX_NOTE_CHANGES:
        raise ValueError(
            'number of changes {} exceeds the maximum of {}'.format(
                count, MAX_NOTE_CHANGES))

    body = data[6:]
    expected_length = count * 4
    if len(body) < expected_length:
        raise ValueError(
            'truncated Single Note Tuning Change message: header declares '
            '{} changes ({} bytes) but only {} payload bytes remain'.format(
                count, expected_length, len(body)))
    if len(body) > expected_length:
        raise ValueError(
            'trailing data after Single Note Tuning Change payload: '
            '{} extra byte(s)'.format(len(body) - expected_length))

    changes = []
    seen = set()
    for index in range(count):
        note, semitone, fraction_msb, fraction_lsb = body[index * 4:
                                                           index * 4 + 4]
        for name, byte in (('note number', note),
                           ('frequency semitone byte', semitone),
                           ('frequency fraction MSB', fraction_msb),
                           ('frequency fraction LSB', fraction_lsb)):
            if not 0 <= byte <= 127:
                raise ValueError(
                    '{} 0x{:02X} is not valid 7 bit data'.format(name, byte))

        if note in seen:
            raise ValueError(
                'duplicate note number {} in tuning changes'.format(note))
        seen.add(note)

        frequency = data_to_frequency(
            (semitone, fraction_msb, fraction_lsb))
        changes.append(NoteTuning(note, frequency))

    return SingleNoteTuningChange(
        realtime=(header == SYSEX_REALTIME),
        device_id=device_id,
        program=program,
        changes=changes,
        time=time)
