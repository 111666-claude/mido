# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tempo ramps for MIDI files.

MIDI files can only encode tempo as a sequence of discrete
``set_tempo`` meta messages, each one holding a constant tempo in
microseconds per quarter note. A *tempo ramp* -- a continuous
accelerando or ritardando over a tick range -- therefore has to be
approximated with a bounded number of constant-tempo segments.

The planner in this module turns a target tempo curve (linear or an
easing function) into the smallest sequence of ``set_tempo`` events
whose accumulated playback time stays within a user supplied maximum
time error of the ideal curve.  :func:`add_tempo_ramp` inserts those
events into an existing :class:`~mido.midifiles.midifiles.MidiFile`
without moving any of the original messages.
"""

import math
from numbers import Integral, Real

from .meta import MetaMessage
from .midifiles import DEFAULT_TEMPO
from .tracks import MidiTrack
from .units import bpm2tempo, tempo2bpm

# Largest tempo value the set_tempo meta message can represent
# (24-bit unsigned, microseconds per quarter note).
MAX_TEMPO = 0xFFFFFF

# A non-linear target curve is integrated numerically on this fixed
# grid.  The grid is fine enough that the integration error is several
# orders of magnitude below any realistic time tolerance.
_GRID_N = 4096

# Hard safety cap on the number of generated set_tempo events.  The
# caller may lower it through the ``max_events`` argument.
DEFAULT_MAX_EVENTS = 1000


class TempoRampError(ValueError):
    """Raised when a tempo ramp cannot be built or inserted."""


def _curve_linear(u):
    return u


def _curve_ease_in(u):
    return u * u


def _curve_ease_out(u):
    return 1.0 - (1.0 - u) ** 2


def _curve_ease_in_out(u):
    return u * u * (3.0 - 2.0 * u)


CURVES = {
    'linear': _curve_linear,
    'ease_in': _curve_ease_in,
    'ease_out': _curve_ease_out,
    'ease_in_out': _curve_ease_in_out,
}


def _resolve_curve(curve):
    """Return a callable mapping u in [0, 1] to an eased position."""
    if isinstance(curve, str):
        try:
            return CURVES[curve]
        except KeyError as ke:
            names = ', '.join(sorted(CURVES))
            raise TempoRampError(
                f'unknown curve {curve!r} (valid names: {names})') from ke

    if not callable(curve):
        raise TempoRampError(
            'curve must be a name string or a callable taking one '
            'argument')

    # Validate the endpoints of user supplied easing functions.
    start = curve(0.0)
    end = curve(1.0)
    if not (isinstance(start, Real) and isinstance(end, Real)
            and abs(start) <= 1e-9 and abs(end - 1.0) <= 1e-9):
        raise TempoRampError(
            'custom curve must satisfy curve(0) == 0 and curve(1) == 1')
    return curve


def _check_positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) \
            or not value > 0:
        raise TempoRampError(f'{name} must be a positive number')


def _quantized_tempo(bpm):
    """Convert BPM to the integer microseconds-per-beat value MIDI stores."""
    tempo = bpm2tempo(bpm)
    if not 1 <= tempo <= MAX_TEMPO:
        raise TempoRampError(
            f'bpm {bpm} cannot be represented as a MIDI tempo '
            f'(would be {tempo} microseconds per beat, valid range is '
            f'1..{MAX_TEMPO})')
    return tempo


def _target_time_lookup(start_bpm, end_bpm, curve, ticks_per_beat,
                        span_ticks):
    """Build a function target_seconds(u) for elapsed time since u == 0.

    The integral over normalized position ``u`` is multiplied by the
    ramp length in ticks so the result is in seconds.  The linear case
    uses the closed-form integral of 1/BPM; every other curve is
    integrated numerically on a fixed grid.
    """
    scale = 60.0 * span_ticks / ticks_per_beat

    if curve is _curve_linear:
        if start_bpm == end_bpm:
            def target_seconds(u):
                return scale * u / start_bpm
        else:
            bpm_span = end_bpm - start_bpm

            def target_seconds(u):
                bpm_u = start_bpm + bpm_span * u
                return scale * math.log(bpm_u / start_bpm) / bpm_span

        return target_seconds

    us = [i / _GRID_N for i in range(_GRID_N + 1)]
    inverse_bpm = [1.0 / (start_bpm + (end_bpm - start_bpm) * curve(u))
                   for u in us]
    cumulative = [0.0]
    for i in range(_GRID_N):
        step = us[i + 1] - us[i]
        cumulative.append(
            cumulative[-1] + step * (inverse_bpm[i] + inverse_bpm[i + 1])
            / 2.0)

    def target_seconds(u):
        position = u * _GRID_N
        index = int(position)
        if index >= _GRID_N:
            return scale * cumulative[_GRID_N]
        fraction = position - index
        value = (cumulative[index]
                 + fraction * (cumulative[index + 1] - cumulative[index]))
        return scale * value

    return target_seconds


def compute_tempo_ramp(start_tick, end_tick, start_bpm, end_bpm,
                       ticks_per_beat, curve='linear',
                       max_time_error=0.01, max_events=DEFAULT_MAX_EVENTS):
    """Plan the ``set_tempo`` events that approximate a tempo ramp.

    Parameters
    ----------
    start_tick, end_tick:
        Absolute tick range of the ramp.  ``end_tick`` must be greater
        than ``start_tick`` and both must be non-negative integers.
    start_bpm, end_bpm:
        Tempo in beats per minute at the ramp boundaries.  Both must be
        positive and representable as MIDI tempo values.
    ticks_per_beat:
        PPQ resolution of the MIDI file (``MidiFile.ticks_per_beat``).
        Must be a positive integer.
    curve:
        Either one of the names in :data:`CURVES` (``'linear'``,
        ``'ease_in'``, ``'ease_out'``, ``'ease_in_out'``) or a
        callable with ``curve(0) == 0`` and ``curve(1) == 1``.
    max_time_error:
        Maximum allowed deviation in seconds between the playback time
        produced by the generated events and the ideal curve, measured
        at every ramp tick.  Must be positive.
    max_events:
        Upper bound on the number of returned events.  If the tolerance
        cannot be reached within this bound, :class:`TempoRampError` is
        raised instead of generating an unbounded number of events.

    Returns
    -------
    list of (tick, tempo) tuples:
        Absolute tick and tempo in microseconds per quarter note for
        every ``set_tempo`` event, sorted by tick and containing both
        ramp endpoints.
    """
    if (not isinstance(ticks_per_beat, Integral)
            or isinstance(ticks_per_beat, bool)
            or ticks_per_beat <= 0):
        raise TempoRampError(
            f'invalid PPQ {ticks_per_beat!r}: '
            'ticks_per_beat must be a positive integer')
    if (not isinstance(start_tick, Integral)
            or isinstance(start_tick, bool) or start_tick < 0):
        raise TempoRampError(
            f'start_tick must be a non-negative integer, got {start_tick!r}')
    if (not isinstance(end_tick, Integral) or isinstance(end_tick, bool)
            or end_tick <= start_tick):
        raise TempoRampError(
            'end_tick must be an integer greater than start_tick '
            f'(got start_tick={start_tick!r}, end_tick={end_tick!r})')
    _check_positive_number(start_bpm, 'start_bpm')
    _check_positive_number(end_bpm, 'end_bpm')
    _check_positive_number(max_time_error, 'max_time_error')
    if (not isinstance(max_events, Integral) or isinstance(max_events, bool)
            or max_events < 2):
        raise TempoRampError('max_events must be an integer >= 2')

    curve_fn = _resolve_curve(curve)

    def bpm_at(u):
        return start_bpm + (end_bpm - start_bpm) * curve_fn(u)

    # Validate that the whole curve (not just its endpoints) encodes a
    # positive, representable tempo.
    _quantized_tempo(start_bpm)
    _quantized_tempo(end_bpm)
    for i in range(1, _GRID_N):
        _quantized_tempo(bpm_at(i / _GRID_N))

    span_ticks = end_tick - start_tick
    target_seconds = _target_time_lookup(
        start_bpm, end_bpm, curve_fn, ticks_per_beat, span_ticks)

    # All timing arithmetic is done in integer *tempo units*
    # (tick * microseconds-per-beat).  One second is PPQ * 10**6 units,
    # so the actual playback clock is an exact integer and the tolerance
    # comparison never depends on accumulated floating-point rounding.
    units_per_second = ticks_per_beat * 1_000_000
    tolerance_units = max_time_error * units_per_second

    def target_units(tick):
        """Ideal elapsed time at ``tick``, expressed in tempo units."""
        u = (tick - start_tick) / span_ticks
        return target_seconds(u) * units_per_second

    def ideal_tempo_at_tick(tick):
        """Quantized MIDI tempo of the target curve at an absolute tick."""
        u = (tick - start_tick) / span_ticks
        return _quantized_tempo(bpm_at(u))

    # Greedy segment placement.  A segment opens at tick ``t0`` with the
    # accumulated actual clock ``clock_units`` and is played at one
    # constant integer tempo.  It is extended while the set of integer
    # tempos that keep every covered tick within the tolerance is
    # non-empty; the chosen tempo is the midpoint of that feasible range,
    # which centres the residual and leaves the most room for the next
    # segment.  Because the clock is carried in exact units, the bound
    # holds for the true cumulative playback time at every ramp tick.
    #
    # The first segment is pinned to the exact requested start tempo and
    # the final event to the exact end tempo, so the playback speed is
    # continuous with the tempo before and after the ramp.
    start_tempo = ideal_tempo_at_tick(start_tick)
    end_tempo = ideal_tempo_at_tick(end_tick)

    events = []
    t0 = start_tick
    clock_units = 0

    while t0 < end_tick:
        if not events:
            # Pin the opening segment to the exact start tempo.
            low = high = start_tempo
        else:
            low, high = 1, MAX_TEMPO
        chosen = None
        probe = t0 + 1
        while probe <= end_tick:
            distance = probe - t0
            target_here = target_units(probe)
            # An integer tempo T must satisfy
            # |clock_units + distance * T - target_here| <= tolerance.
            new_low = math.ceil(
                (target_here - tolerance_units - clock_units) / distance)
            new_high = math.floor(
                (target_here + tolerance_units - clock_units) / distance)
            low = max(low, new_low)
            high = min(high, new_high)
            if low > high:
                break
            chosen = (low + high) // 2
            probe += 1

        # ``probe`` is the first tick that no tempo can cover; the last
        # conforming tick closes the segment.
        last_good = probe - 1
        if chosen is None or last_good <= t0:
            if t0 + 1 <= end_tick:
                residual = abs(
                    clock_units + ideal_tempo_at_tick(t0 + 1)
                    - target_units(t0 + 1))
            else:
                residual = tolerance_units
            raise TempoRampError(
                f'cannot reach max_time_error={max_time_error} s: one '
                f'tick at {ticks_per_beat} ticks per beat already '
                f'deviates by {residual / units_per_second:.6f} s, which '
                'is beyond the resolution of this file. Increase '
                'ticks_per_beat or max_time_error.')

        if len(events) + 1 > max_events:
            raise TempoRampError(
                f'tempo ramp would need more than max_events={max_events} '
                f'set_tempo events to stay within '
                f'max_time_error={max_time_error} s. Increase '
                'max_time_error or max_events.')

        events.append((t0, chosen))
        clock_units += (last_good - t0) * chosen
        t0 = last_good

    # Guarantee the exact end tempo at the end tick.  When the final
    # segment already ends there, replace its tempo; otherwise append the
    # end event.  The exact verification below confirms the pinning keeps
    # the whole ramp inside the requested tolerance.
    if events[-1][0] == end_tick:
        events[-1] = (end_tick, end_tempo)
    elif end_tempo != events[-1][1]:
        if len(events) + 1 > max_events:
            raise TempoRampError(
                f'tempo ramp would need more than max_events={max_events} '
                f'set_tempo events to stay within '
                f'max_time_error={max_time_error} s. Increase '
                'max_time_error or max_events.')
        events.append((end_tick, end_tempo))

    # Drop any interior event whose tempo repeats the previous one; it
    # changes nothing and would only clutter the file.  The end tick is
    # always retained so the post-ramp tempo is explicitly defined.
    deduped = [events[0]]
    for tick, tempo in events[1:]:
        if tempo != deduped[-1][1] or tick == end_tick:
            deduped.append((tick, tempo))
    events = deduped

    # Exact verification: replay the finished event list in integer
    # units, applying each tempo to the ticks strictly after its
    # breakpoint (the MIDI convention), and confirm the cumulative clock
    # never leaves the tolerance band.  This guards the end-tempo pinning
    # and any accumulated rounding in the target-time integration.
    clock_units = 0
    for index, (tick, tempo) in enumerate(events):
        next_tick = events[index + 1][0] if index + 1 < len(events) \
            else end_tick
        for played in range(tick + 1, next_tick + 1):
            clock_units += tempo
            if abs(clock_units - target_units(played)) \
                    > tolerance_units + 1.0:
                raise TempoRampError(
                    'internal error: planned tempo map breaches the '
                    f'tolerance at tick {played}')

    return events


def _absolute_set_tempos(mid):
    """All set_tempo messages in the file as (absolute tick, track, msg)."""
    found = []
    for track_index, track in enumerate(mid.tracks):
        now = 0
        for msg in track:
            now += msg.time
            if msg.type == 'set_tempo':
                found.append((now, track_index, msg))
    found.sort(key=lambda item: item[0])
    return found


def add_tempo_ramp(mid, start_tick, end_tick, start_bpm, end_bpm,
                   curve='linear', max_time_error=0.01, track=0,
                   max_events=DEFAULT_MAX_EVENTS):
    """Insert a tempo ramp into an existing :class:`MidiFile`.

    The original notes and meta messages keep their absolute tick,
    their track and their relative order at shared ticks.  New
    ``set_tempo`` events are written to one chosen track and ordered
    *before* any original message at the same tick.  Delta times of the
    rewritten track are recomputed and are therefore non-negative.

    The tempo active at ``start_tick`` must already equal ``start_bpm``
    (within MIDI tempo quantization), and no existing ``set_tempo``
    event may sit strictly inside the ramp range; otherwise the ramp
    would contradict the file's current tempo map.

    Parameters
    ----------
    mid:
        The :class:`MidiFile` to modify in place.
    start_tick, end_tick, start_bpm, end_bpm, curve, max_time_error,
    max_events:
        See :func:`compute_tempo_ramp`.
    track:
        Index of the track that receives the new ``set_tempo`` events.
        Defaults to the first track, the conventional place for tempo
        events in type 1 files.

    Returns
    -------
    list of MetaMessage:
        The newly inserted ``set_tempo`` messages with their final
        delta times.  Pre-existing boundary events with a matching tempo
        are reused and not included.

    Raises
    ------
    TempoRampError:
        For type 2 files, bad arguments, overlapping ramps, tempo
        discontinuities, or an unachievable time tolerance.
    """
    if mid.type == 2:
        raise TempoRampError(
            'cannot add a tempo ramp to a type 2 (asynchronous) file')
    if (not isinstance(track, Integral) or isinstance(track, bool)
            or not 0 <= track < len(mid.tracks)):
        raise TempoRampError(
            f'track index {track!r} is out of range for a file with '
            f'{len(mid.tracks)} track(s)')

    planned = compute_tempo_ramp(
        start_tick, end_tick, start_bpm, end_bpm,
        mid.ticks_per_beat, curve=curve,
        max_time_error=max_time_error, max_events=max_events)

    start_tempo = planned[0][1]
    end_tempo = planned[-1][1]
    existing = _absolute_set_tempos(mid)

    # Pass 1: reject any tempo event strictly inside the ramp range.
    # This is checked before boundary mismatches so that an overlapping
    # ramp is always reported as an overlap, regardless of its values.
    for tick, _track_index, _msg in existing:
        if start_tick < tick < end_tick:
            raise TempoRampError(
                'overlapping tempo change: a set_tempo event already '
                f'exists at tick {tick}, inside the ramp '
                f'[{start_tick}, {end_tick}]')

    # Pass 2: boundary events must agree with the planned endpoint tempos.
    for tick, _track_index, msg in existing:
        if tick == start_tick and msg.tempo != start_tempo:
            raise TempoRampError(
                f'set_tempo at start tick {start_tick} has tempo '
                f'{msg.tempo} but the ramp starts with tempo '
                f'{start_tempo} ({start_bpm} BPM)')
        if tick == end_tick and msg.tempo != end_tempo:
            raise TempoRampError(
                f'set_tempo at end tick {end_tick} has tempo '
                f'{msg.tempo} but the ramp ends with tempo '
                f'{end_tempo} ({end_bpm} BPM)')

    # The tempo active at the ramp start must match its first event,
    # otherwise the ramp introduces a jump instead of a smooth change.
    # A matching boundary event is itself authoritative; otherwise the
    # active tempo is the last set_tempo strictly before the start (or
    # the MIDI default of 120 BPM when there is none).
    active_tempo = DEFAULT_TEMPO
    for tick, _track_index, msg in existing:
        if tick == start_tick:
            active_tempo = msg.tempo
            break
        if tick < start_tick:
            active_tempo = msg.tempo
        else:
            break
    if active_tempo != start_tempo:
        raise TempoRampError(
            'ramp would change the tempo discontinuously at tick '
            f'{start_tick}: the file plays at {active_tempo} us/beat '
            f'({tempo2bpm(active_tempo):.3f} BPM) but the ramp starts '
            f'with {start_tempo} us/beat ({start_bpm} BPM). Insert a '
            'matching set_tempo at the ramp start or adjust start_bpm.')

    # Boundary events with an identical tempo already encode the
    # endpoint; reuse them instead of duplicating the event.
    boundary_ticks = {tick for tick, _, _ in existing
                      if tick in (start_tick, end_tick)}
    new_events = [(tick, tempo) for tick, tempo in planned
                  if tick not in boundary_ticks]

    target_track = mid.tracks[track]

    # Flatten the target track to absolute ticks, tagging original
    # messages with rank 1 so that new events (rank 0) come first at a
    # shared tick while every original ordering is preserved.
    entries = []
    now = 0
    for msg in target_track:
        now += msg.time
        entries.append((now, 1, msg))
    for tick, tempo in new_events:
        entries.append((tick, 0, MetaMessage('set_tempo', tempo=tempo)))
    entries.sort(key=lambda entry: (entry[0], entry[1]))

    # Suppress a new event whose tempo equals the tempo already in force
    # at that tick (for example the first interior event repeating the
    # pre-ramp tempo); it would be a redundant set_tempo.
    def tempo_active_before(tick):
        active = DEFAULT_TEMPO
        for existing_tick, _track_index, msg in existing:
            if existing_tick < tick:
                active = msg.tempo
            else:
                break
        return active

    filtered_entries = []
    for tick, rank, msg in entries:
        # Suppress a new interior event whose tempo equals the tempo
        # already in force at that tick; it would be a redundant
        # set_tempo.  The ramp's start event is always kept so the ramp
        # has an explicit, self-contained beginning even when it matches
        # the file's default tempo.
        if (rank == 0 and msg.type == 'set_tempo'
                and tick != start_tick
                and msg.tempo == tempo_active_before(tick)):
            continue
        filtered_entries.append((tick, rank, msg))
    entries = filtered_entries

    rewritten = MidiTrack()
    last_tick = 0
    inserted = []
    for tick, rank, msg in entries:
        delta = tick - last_tick
        placed = msg.copy(time=delta)
        rewritten.append(placed)
        if rank == 0:
            inserted.append(placed)
        last_tick = tick

    mid.tracks[track] = rewritten
    # The merged track cache, if any, is now stale.
    del mid.merged_track

    return inserted
