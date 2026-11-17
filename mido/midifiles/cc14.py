# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

"""High-resolution (14-bit) control change view of a MIDI file.

Many controllers send their values as a pair of control change
messages: the most significant byte (MSB) on controllers 0..31 and
the least significant byte (LSB) on the corresponding controllers
32..63. This module provides a combined 14-bit view of those pairs.

``read_cc14(midifile)`` extracts the view from a MidiFile and
``write_cc14(midifile, curves)`` writes a (possibly modified) view
back as paired control change messages.

The view is a dictionary mapping ``(channel, control)`` tuples to a
curve, where ``control`` is the MSB controller number (0..31). A
curve is a list of ``(tick, value)`` points with absolute times in
ticks and values in 0..16383.

Rules for reading:

* Each point holds the value of the most recently received MSB and
  LSB parts: ``msb * 128 + lsb``. A part that has never been received
  counts as 0, so a lone MSB or LSB message still produces a point.
* Events are processed in a deterministic order: by absolute tick,
  then by track index, then by position inside the track. If several
  messages for the same ``(channel, control)`` pair land on the same
  tick (duplicates or cross-track conflicts) the later one in this
  order wins, and the curve holds a single point for that tick.
* Bank select (controllers 0 and 32) is treated as an ordinary
  14-bit pair like any other; it gets no special handling.
* Controllers 64..127 (sustain pedal and so on) are not part of any
  14-bit pair and are ignored.

Rules for writing:

* Only the ``(channel, control)`` pairs present in ``curves`` are
  touched. For each of them, all existing MSB and LSB messages on
  that channel are removed from every track and replaced by one MSB
  message immediately followed by one LSB message per curve point.
* The replacement messages are placed in the track that contained
  the pair's first original message (lowest track index). If the
  pair did not exist in the file, they go to the first track (a
  track is created if the file has none).
* New messages at a tick are inserted after all pre-existing
  messages at the same tick. The relative order of all other
  messages (notes, other controllers, meta messages) is preserved.
* Values must be integers in 0..16383 and ticks non-negative
  integers; anything else raises ValueError. Duplicate ticks in a
  curve are resolved before anything is written: the last point
  wins.
"""

from ..messages import Message
from .tracks import _to_abstime, _to_reltime

# Controllers 0..31 are the MSB parts and 32..63 the LSB parts.
NUM_PAIR_CONTROLLERS = 32
MAX_CC14_VALUE = 16383

__all__ = ['read_cc14', 'write_cc14']


def _pair_key(channel, control):
    """Map a control change to its ((channel, msb_control), part) key.

    ``part`` is 0 for the MSB message and 1 for the LSB message.
    Returns None for controllers that are not part of a 14-bit pair.
    """
    if control < NUM_PAIR_CONTROLLERS:
        return (channel, control), 0
    elif control < 2 * NUM_PAIR_CONTROLLERS:
        return (channel, control - NUM_PAIR_CONTROLLERS), 1
    else:
        return None


def read_cc14(midifile):
    """Extract 14-bit control change curves from a MidiFile.

    Returns a dictionary mapping ``(channel, control)`` to a list of
    ``(tick, value)`` points, sorted by tick. ``control`` is the MSB
    controller number (0..31) and ``value`` is in 0..16383.

    The returned curves are plain lists and may be modified freely;
    pass them (or a subset) to ``write_cc14()`` to write the changes
    back to the file.
    """
    # Collect events in a deterministic global order: absolute tick,
    # then track index, then position inside the track.
    events = []
    for track_index, track in enumerate(midifile.tracks):
        for pos, msg in enumerate(_to_abstime(track, skip_checks=True)):
            if msg.type == 'control_change':
                key = _pair_key(msg.channel, msg.control)
                if key is not None:
                    events.append((msg.time, track_index, pos,
                                   key[0], key[1], msg.value))
    events.sort(key=lambda event: (event[0], event[1], event[2]))

    # parts[(channel, control)] = [latest msb, latest lsb]
    parts = {}
    # points[(channel, control)] = {tick: value}
    points = {}

    for tick, _, _, key, part, value in events:
        latest = parts.setdefault(key, [0, 0])
        latest[part] = value
        # Later events at the same tick overwrite earlier ones.
        points.setdefault(key, {})[tick] = latest[0] * 128 + latest[1]

    return {key: sorted(tick_points.items())
            for key, tick_points in points.items()}


def _validate_curves(curves):
    """Validate curves and return a normalized copy.

    The normalized form maps (channel, control) to a list of
    (tick, value) points sorted by tick with duplicates resolved
    (last point wins).
    """
    normalized = {}

    for key, points in curves.items():
        try:
            channel, control = key
        except (TypeError, ValueError):
            raise ValueError(
                f'invalid cc14 key {key!r} (must be a (channel, control) '
                'tuple)') from None

        if not (isinstance(channel, int) and 0 <= channel <= 15):
            raise ValueError(
                f'invalid channel {channel!r} (must be in 0..15)')
        if not (isinstance(control, int)
                and 0 <= control < NUM_PAIR_CONTROLLERS):
            raise ValueError(
                f'invalid control {control!r} (must be in 0..31, the MSB '
                'controller of the pair)')

        tick_points = {}
        for point in points:
            try:
                tick, value = point
            except (TypeError, ValueError):
                raise ValueError(
                    f'invalid cc14 point {point!r} (must be a (tick, value) '
                    'tuple)') from None

            if not (isinstance(tick, int) and tick >= 0):
                raise ValueError(
                    f'invalid tick {tick!r} (must be a non-negative int)')
            if not (isinstance(value, int)
                    and 0 <= value <= MAX_CC14_VALUE):
                raise ValueError(
                    f'invalid cc14 value {value!r} (must be in '
                    f'0..{MAX_CC14_VALUE})')
            # Duplicate ticks: the last point wins.
            tick_points[tick] = value

        normalized[(channel, control)] = sorted(tick_points.items())

    return normalized


def write_cc14(midifile, curves):
    """Write 14-bit control change curves back to a MidiFile.

    ``curves`` is a dictionary as returned by ``read_cc14()`` (or a
    subset of it). Only the ``(channel, control)`` pairs present in
    the dictionary are modified; every other message in the file is
    left untouched, and the relative order of messages sharing a
    tick is preserved.

    The file is modified in place. The whole dictionary is validated
    before anything is written, so a ValueError leaves the file
    unchanged.
    """
    normalized = _validate_curves(curves)

    # Controllers to remove per channel: the MSB and LSB of each
    # pair that is being rewritten.
    remove = set()
    for channel, control in normalized:
        remove.add((channel, control))
        remove.add((channel, control + NUM_PAIR_CONTROLLERS))

    if not midifile.tracks and normalized:
        midifile.add_track()

    # Split every track into absolute-time messages, dropping the
    # messages that are being replaced, and remember which tracks
    # originally carried each pair.
    abs_tracks = []
    first_track = {}
    for track_index, track in enumerate(midifile.tracks):
        abs_track = []
        for msg in _to_abstime(track, skip_checks=True):
            if (msg.type == 'control_change'
                    and (msg.channel, msg.control) in remove):
                key = (msg.channel, msg.control % NUM_PAIR_CONTROLLERS)
                first_track.setdefault(key, track_index)
            else:
                abs_track.append(msg)
        abs_tracks.append(abs_track)

    # Build the replacement messages for each pair and add them to
    # their target track. Appending before the stable sort below
    # places them after all pre-existing messages at the same tick.
    for (channel, control), points in normalized.items():
        track_index = first_track.get((channel, control), 0)
        abs_track = abs_tracks[track_index]
        for tick, value in points:
            abs_track.append(Message('control_change',
                                     channel=channel, control=control,
                                     value=value >> 7, time=tick))
            abs_track.append(Message('control_change',
                                     channel=channel,
                                     control=control + NUM_PAIR_CONTROLLERS,
                                     value=value & 0x7f, time=tick))

    # Sort each track back into time order (stable, so messages
    # sharing a tick keep their relative order) and convert back to
    # delta times.
    for track, abs_track in zip(midifile.tracks, abs_tracks):
        abs_track.sort(key=lambda msg: msg.time)
        track[:] = _to_reltime(abs_track, skip_checks=True)

    # The merged track cache no longer reflects the tracks.
    del midifile.merged_track
