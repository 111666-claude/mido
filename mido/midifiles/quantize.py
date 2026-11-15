# SPDX-FileCopyrightText: 2026
#
# SPDX-License-Identifier: MIT

"""
Whole-file note quantization.

Note messages are moved onto a subdivision grid (optionally with swing)
while every other message keeps its original absolute time.  See
:func:`quantize_notes` for details.
"""

from fractions import Fraction

from .tracks import MidiTrack

# A MIDI delta time / variable-length quantity uses at most four bytes,
# which holds 28 bits.
MAX_VLQ_VALUE = 0x0FFFFFFF


def _validate_params(subdivision, swing, ticks_per_beat):
    """Validate quantization parameters.

    Raises TypeError for arguments of the wrong type and ValueError
    for arguments of the right type but outside the valid range.
    """
    if isinstance(subdivision, bool) or not isinstance(subdivision, int):
        raise TypeError(
            f'subdivision must be a positive integer, '
            f'not {type(subdivision).__name__}')
    if subdivision <= 0:
        raise ValueError(
            f'subdivision must be a positive integer, not {subdivision!r}')

    if isinstance(swing, bool) or not isinstance(swing, (int, float)):
        raise TypeError(
            f'swing must be a number, not {type(swing).__name__}')
    if not 0.0 < swing < 1.0:
        raise ValueError(
            f'swing must be strictly between 0 and 1, not {swing!r}')

    if isinstance(ticks_per_beat, bool) or not isinstance(
            ticks_per_beat, int):
        raise TypeError(
            'ticks_per_beat must be a positive integer, '
            f'not {type(ticks_per_beat).__name__}')
    if ticks_per_beat <= 0:
        raise ValueError(
            'ticks_per_beat must be a positive integer, '
            f'not {ticks_per_beat!r}')


def _time_signature_regions(tracks):
    """Return ``[(start_tick, numerator, denominator), ...]`` sorted.

    Time signature meta messages are collected from every track.  When
    several are defined at the same tick the one from the earliest
    track (and earliest position in that track) wins.  A default 4/4
    region starting at tick 0 is always included.
    """
    events = []
    for track_index, track in enumerate(tracks):
        now = 0
        for msg_index, msg in enumerate(track):
            now += msg.time
            if msg.type == 'time_signature':
                events.append(
                    (now, track_index, msg_index,
                     msg.numerator, msg.denominator))

    events.sort()

    regions = [(0, 4, 4)]
    for tick, _track, _msg, numerator, denominator in events:
        if tick == regions[-1][0]:
            # A later entry at the same tick replaces the earlier one
            # (sorting guarantees the earliest track entry comes first).
            regions[-1] = (tick, numerator, denominator)
        else:
            regions.append((tick, numerator, denominator))

    return regions


def _bar_length(numerator, denominator, ticks_per_beat):
    # A quarter note is one beat in tick terms regardless of the
    # notated denominator: a whole bar lasts
    # numerator / denominator * 4 quarter notes.
    return Fraction(numerator * 4 * ticks_per_beat, denominator)


def _iter_bars(regions, ticks_per_beat):
    """Yield ``(bar_start_tick, bar_length)`` pairs without an end.

    A time signature change takes effect on the first bar boundary at
    or after its tick, as recommended by the MIDI file specification.
    """
    position = Fraction(0)
    for index, (change_tick, numerator, denominator) in enumerate(regions):
        length = _bar_length(numerator, denominator, ticks_per_beat)

        if index > 0:
            # Finish the previous region's bars up to the boundary at
            # or after the change.  A change landing exactly on a
            # boundary needs no padding bars.
            prev_num, prev_den = regions[index - 1][1], regions[index - 1][2]
            prev_length = _bar_length(
                prev_num, prev_den, ticks_per_beat)
            while position < change_tick:
                yield position, prev_length
                position += prev_length

        yield position, length
        position += length

    # The final region continues indefinitely.
    last_num, last_den = regions[-1][1], regions[-1][2]
    last_length = _bar_length(last_num, last_den, ticks_per_beat)
    while True:
        yield position, last_length
        position += last_length


def _round_nearest(value):
    """Round a non-negative number to the nearest int (halves go up)."""
    fraction = Fraction(value)
    whole, remainder = divmod(fraction.numerator, fraction.denominator)
    return whole + (2 * remainder >= fraction.denominator)


class _Grid:
    """Subdivision grid whose swing phase restarts at every bar line."""

    def __init__(self, regions, ticks_per_beat, subdivision, swing):
        self._bars = []
        self._spacing = Fraction(ticks_per_beat, subdivision)
        self._swing = Fraction(str(swing))
        self._region_iter = _iter_bars(regions, ticks_per_beat)

    def _ensure_bars(self, tick):
        """Generate bars until the last generated bar starts past tick."""
        while not self._bars or self._bars[-1][0] <= tick:
            self._bars.append(next(self._region_iter))

    def _bar_for(self, tick):
        """Return the (start, length) of the bar containing tick."""
        self._ensure_bars(tick)

        for candidate_start, candidate_length in self._bars:
            if candidate_start <= tick < candidate_start + candidate_length:
                return candidate_start, candidate_length
            if candidate_start == tick:
                # Tick exactly on a bar boundary starts the new bar.
                return candidate_start, candidate_length

        # Fallback for ticks past the last generated (fractional) bar.
        return self._bars[-1]

    def quantize(self, tick):
        """Return the grid position (integer ticks) nearest to tick."""
        bar_start, bar_length = self._bar_for(tick)

        offset = Fraction(tick) - bar_start
        # Round to the nearest slot index rather than flooring, so an
        # offset sitting on (or a hair below) a grid line cannot pick
        # the neighbouring slot.
        slot = _round_nearest(offset / self._spacing)

        # Bars may hold a fractional number of slots.  A positive slot
        # index at or past the bar end means the tick belongs to the
        # next bar.
        if slot > 0 and slot * self._spacing >= bar_length:
            bar_start += bar_length
            offset = Fraction(tick) - bar_start
            slot = _round_nearest(offset / self._spacing)

        straight = bar_start + slot * self._spacing

        if slot % 2 == 1:
            # The second slot of each pair is pushed later (or earlier,
            # with swing < 0.5) within its two-slot cell.  At swing 0.5
            # this is exactly the straight grid position.
            position = (bar_start
                        + (slot - 1) * self._spacing
                        + 2 * self._swing * self._spacing)
        else:
            position = straight

        return _round_nearest(position)


def _is_note_off(msg):
    return (msg.type == 'note_off'
            or (msg.type == 'note_on' and msg.velocity == 0))


def _quantize_track(track, grid, strict):
    """Quantize one track.

    Returns ``[(order, tick, msg), ...]`` sorted on absolute tick with
    the original relative order preserved for ties.
    """
    absolute = []
    now = 0
    for index, msg in enumerate(track):
        now += msg.time
        absolute.append((index, now, msg))

    # LIFO stacks of open note_on messages keyed by (channel, note).
    # Last-in-first-out is the conventional pairing for overlapping
    # identical notes.
    open_notes = {}
    new_times = {}
    unmatched = []

    for index, tick, msg in absolute:
        if msg.type == 'note_on' and msg.velocity > 0:
            open_notes.setdefault((msg.channel, msg.note), []).append(
                (index, tick))
        elif _is_note_off(msg):
            stack = open_notes.get((msg.channel, msg.note))
            if stack:
                on_index, on_tick = stack.pop()
                if not stack:
                    del open_notes[(msg.channel, msg.note)]

                quantized_on = grid.quantize(on_tick)
                quantized_off = grid.quantize(tick)

                # Endpoints are quantized independently, but a note
                # must keep a strictly positive duration.
                if quantized_off <= quantized_on:
                    quantized_off = quantized_on + 1

                new_times[on_index] = quantized_on
                new_times[index] = quantized_off
            else:
                unmatched.append((index, tick, msg))
        # All other message types keep their original absolute time.

    still_open = [item for stack in open_notes.values() for item in stack]
    if (still_open or unmatched) and strict:
        problems = sorted(
            [(tick, msg) for _index, tick in still_open]
            + [(tick, msg) for _index, tick, msg in unmatched],
            key=lambda item: item[0])
        description = ', '.join(
            f'{msg.type} (channel={getattr(msg, "channel", "?")}, '
            f'note={getattr(msg, "note", "?")}) at tick {tick}'
            for tick, msg in problems)
        raise ValueError(
            f'cannot quantize: {len(problems)} note message(s) could not '
            f'be paired ({description})')

    # Lenient mode: unpaired note messages stay at their original time.
    for index, tick in still_open:
        new_times.setdefault(index, tick)
    for index, tick, _msg in unmatched:
        new_times.setdefault(index, tick)

    result = []
    for index, tick, msg in absolute:
        if msg.type == 'end_of_track':
            # Re-added at save time by fix_end_of_track().
            continue
        result.append((index, int(new_times.get(index, tick)), msg))

    # Stable ordering on (new tick, original index) guarantees that
    # every reconstructed delta time is non-negative.
    result.sort(key=lambda item: (item[1], item[0]))
    return result


def quantize_notes(mid, subdivision, swing=0.5, strict=True):
    """Return a copy of ``mid`` with all notes snapped to a time grid.

    Note-on and note-off events (including zero-velocity ``note_on``
    messages, which are equivalent to note offs) are quantized
    independently.  Every other message -- tempo, time signature,
    controller, program change, sysex and so on -- keeps its original
    absolute tick position.  The relative order of messages sharing a
    tick is preserved, and the result never contains negative delta
    times.

    Parameters
    ----------
    mid:
        The :class:`~mido.midifiles.midifiles.MidiFile` to quantize.
        It is left untouched; a new file is returned.
    subdivision:
        Grid slots per beat (quarter note).  ``4`` gives sixteenth
        notes, ``3`` gives eighth-note triplets.  Must be a positive
        integer.
    swing:
        Fraction of each two-slot cell taken by its first slot,
        strictly between ``0`` and ``1``.  ``0.5`` is a straight grid;
        larger values delay the off-grid slot (classic swing).
    strict:
        When true (the default), a note on without a matching note off
        (or vice versa) raises :class:`ValueError`.  When false,
        unpaired note messages are kept at their original time.

    Raises
    ------
    TypeError:
        When ``subdivision`` or ``swing`` has the wrong type.
    ValueError:
        For out-of-range ``subdivision`` or ``swing`` values, unpaired
        note messages in strict mode, or when a quantized position
        would exceed the range of a MIDI variable-length quantity.
    """
    _validate_params(subdivision, swing, mid.ticks_per_beat)

    if mid.type == 2:
        # Type 2 tracks run on independent timelines, so every track
        # is quantized against its own time signature messages.
        region_sets = [
            _time_signature_regions([track]) for track in mid.tracks
        ]
    else:
        shared_regions = _time_signature_regions(mid.tracks)
        region_sets = [shared_regions for _track in mid.tracks]

    new_tracks = []
    for track, regions in zip(mid.tracks, region_sets):
        grid = _Grid(regions, mid.ticks_per_beat, subdivision, swing)
        entries = _quantize_track(track, grid, strict)

        rebuilt = MidiTrack()
        previous = 0
        for _index, tick, msg in entries:
            if tick > MAX_VLQ_VALUE:
                raise ValueError(
                    f'quantized tick {tick} exceeds the maximum MIDI '
                    f'variable-length quantity of {MAX_VLQ_VALUE}')
            rebuilt.append(msg.copy(time=tick - previous))
            previous = tick
        new_tracks.append(rebuilt)

    return mid.__class__(
        type=mid.type,
        ticks_per_beat=mid.ticks_per_beat,
        charset=mid.charset,
        debug=mid.debug,
        clip=mid.clip,
        tracks=new_tracks,
    )
