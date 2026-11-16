# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

import io
import math

import pytest

import mido
from mido import (
    Message,
    MetaMessage,
    MidiFile,
    MidiTrack,
    TempoRampError,
    add_tempo_ramp,
    bpm2tempo,
    compute_tempo_ramp,
)

PPQ = 480
# Eight 4/4 bars at 480 ticks per beat.
RAMP_START = 0
RAMP_END = 8 * 4 * PPQ
TOLERANCE = 0.005  # 5 ms


def make_multitrack_file():
    """Two tracks of fixed-tempo notes plus a spare meta message."""
    mid = MidiFile(type=1, ticks_per_beat=PPQ)

    track0 = MidiTrack([
        MetaMessage('track_name', name='Tempo track', time=0),
        Message('note_on', note=60, velocity=64, time=0),
        Message('note_off', note=60, velocity=64, time=PPQ),
        Message('note_on', note=62, velocity=64, time=0),
        Message('note_off', note=62, velocity=64, time=PPQ),
        MetaMessage('marker', text='middle', time=2 * PPQ),
    ])
    track1 = MidiTrack([
        Message('note_on', note=72, velocity=64, time=PPQ),
        Message('note_off', note=72, velocity=64, time=PPQ),
        Message('note_on', note=74, velocity=64, time=2 * PPQ),
        Message('note_off', note=74, velocity=64, time=PPQ),
    ])
    mid.tracks.append(track0)
    mid.tracks.append(track1)
    return mid


def absolute_messages(track):
    now = 0
    result = []
    for msg in track:
        now += msg.time
        result.append((now, msg))
    return result


def tempo_timeline(mid):
    """set_tempo events as (absolute tick, tempo us/beat) over all tracks."""
    events = []
    for track in mid.tracks:
        for tick, msg in absolute_messages(track):
            if msg.type == 'set_tempo':
                events.append((tick, msg.tempo))
    return sorted(events)


def _eased(curve_name, u):
    if curve_name == 'linear':
        return u
    if curve_name == 'ease_in':
        return u * u
    if curve_name == 'ease_out':
        return 1.0 - (1.0 - u) ** 2
    if curve_name == 'ease_in_out':
        return u * u * (3.0 - 2.0 * u)
    raise AssertionError(curve_name)


def target_time(u, start_bpm, end_bpm, curve_name):
    """Elapsed seconds at normalized position u, integrated numerically."""
    if u == 0:
        return 0.0
    n = max(2000, round(20000 * u))
    total = 0.0
    previous = 1.0 / start_bpm
    for i in range(1, n + 1):
        x = u * i / n
        bpm = start_bpm + (end_bpm - start_bpm) * _eased(curve_name, x)
        value = 1.0 / bpm
        total += (previous + value) / 2.0 * (u / n)
        previous = value
    # Convert the integral over normalized position to seconds.
    return 60.0 * RAMP_END / PPQ * total


def actual_ramp_seconds(mid, start_bpm):
    """Playback time at every ramp tick under the file's tempo events.

    A ``set_tempo`` at tick T governs the ticks strictly after T, up to
    the next event, matching MIDI delta-time semantics.
    """
    timeline = tempo_timeline(mid)
    seconds = {0: 0.0}
    elapsed = 0.0
    for index, (tick, tempo) in enumerate(timeline):
        next_tick = (timeline[index + 1][0]
                     if index + 1 < len(timeline) else RAMP_END)
        for played_tick in range(tick + 1, next_tick + 1):
            elapsed += tempo * 1e-6 / PPQ
            seconds[played_tick] = elapsed
    return seconds


# ---------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------

def test_compute_ramp_endpoints_and_bounded_count():
    events = compute_tempo_ramp(
        RAMP_START, RAMP_END, 60, 180, PPQ,
        curve='linear', max_time_error=TOLERANCE)

    assert events[0][0] == RAMP_START
    assert events[-1][0] == RAMP_END
    # Endpoint tempos match the requested BPM after MIDI quantization.
    assert events[0][1] == bpm2tempo(60)
    assert events[-1][1] == bpm2tempo(180)
    assert len(events) <= 1000
    # Ticks are strictly increasing, tempos are valid 24-bit integers.
    ticks = [tick for tick, _ in events]
    assert ticks == sorted(set(ticks))
    assert all(1 <= tempo <= 0xffffff for _, tempo in events)


def test_linear_ramp_timing_within_tolerance_and_integral_endpoint():
    mid = make_multitrack_file()
    # The file defaults to 120 BPM, so the ramp starts there to keep the
    # preceding tempo continuous.
    inserted = add_tempo_ramp(
        mid, RAMP_START, RAMP_END, 120, 360,
        curve='linear', max_time_error=TOLERANCE)
    assert inserted  # something was inserted

    seconds = actual_ramp_seconds(mid, 120)
    for tick in range(0, RAMP_END + 1, 97):  # dense sweep, not just nodes
        u = tick / RAMP_END
        error = abs(seconds[tick] - target_time(u, 120, 360, 'linear'))
        assert error <= TOLERANCE + 1e-9

    # The integrated total duration must match the analytic integral.
    expected_total = target_time(1.0, 120, 360, 'linear')
    assert abs(seconds[RAMP_END] - expected_total) <= TOLERANCE
    # Sanity check on the analytic value itself (120 -> 360 BPM ramp):
    # 60 * span/PPQ * ln(3) / (360 - 120).
    assert expected_total == pytest.approx(
        60.0 * RAMP_END / PPQ * math.log(3.0) / 240.0, rel=1e-9)


def test_ease_in_out_ramp_timing_within_tolerance():
    mid = make_multitrack_file()
    add_tempo_ramp(
        mid, RAMP_START, RAMP_END, 120, 160,
        curve='ease_in_out', max_time_error=TOLERANCE)

    seconds = actual_ramp_seconds(mid, 120)
    for tick in range(0, RAMP_END + 1, 131):
        u = tick / RAMP_END
        error = abs(seconds[tick]
                    - target_time(u, 120, 160, 'ease_in_out'))
        assert error <= TOLERANCE + 1e-9

    expected_total = target_time(1.0, 120, 160, 'ease_in_out')
    assert abs(seconds[RAMP_END] - expected_total) <= TOLERANCE


def test_ease_curves_differ_from_linear():
    linear = compute_tempo_ramp(
        0, RAMP_END, 60, 240, PPQ, 'linear', max_time_error=TOLERANCE)
    eased = compute_tempo_ramp(
        0, RAMP_END, 60, 240, PPQ, 'ease_in', max_time_error=TOLERANCE)
    assert [t for t, _ in linear] != [t for t, _ in eased]


def test_constant_bpm_ramp_is_single_event():
    events = compute_tempo_ramp(
        100, 100 + 4 * PPQ, 100, 100, PPQ, max_time_error=0.001)
    assert events == [(100, bpm2tempo(100))]


def test_tighter_tolerance_needs_at_least_as_many_events():
    loose = compute_tempo_ramp(
        0, RAMP_END, 60, 200, PPQ, max_time_error=0.05)
    tight = compute_tempo_ramp(
        0, RAMP_END, 60, 200, PPQ, max_time_error=0.002)
    assert len(tight) > len(loose)


def test_custom_curve_callable():
    events = compute_tempo_ramp(
        0, RAMP_END, 90, 150, PPQ, curve=lambda u: u ** 3,
        max_time_error=TOLERANCE)
    assert events[0][1] == bpm2tempo(90)
    assert events[-1][1] == bpm2tempo(150)


def target_time_table(start_bpm, end_bpm, curve_name):
    """Ideal elapsed seconds at every ramp tick via one cumulative pass.

    The trapezoidal rule is O(RAMP_END) and its discretization error is
    quadratic in the step size, hence far below any tested tolerance;
    calling target_time() per tick would re-integrate from scratch and
    be quadratic.
    """
    ideal = [
        60.0 / (start_bpm + (end_bpm - start_bpm)
                * _eased(curve_name, tick / RAMP_END)) / PPQ
        for tick in range(RAMP_END + 1)
    ]
    table = [0.0]
    cumulative = 0.0
    for tick in range(RAMP_END):
        cumulative += (ideal[tick] + ideal[tick + 1]) / 2.0
        table.append(cumulative)
    return table


def test_every_integer_tick_is_within_tolerance_for_all_curves():
    # The strongest guarantee: walk every single ramp tick, not just a
    # sampled sweep, and confirm the cumulative error stays bounded.
    for curve_name in ('linear', 'ease_in', 'ease_out', 'ease_in_out'):
        events = compute_tempo_ramp(
            0, RAMP_END, 120, 240, PPQ, curve=curve_name,
            max_time_error=TOLERANCE)
        target = target_time_table(120, 240, curve_name)

        clock = 0.0
        worst = 0.0
        for index, (tick, tempo) in enumerate(events):
            next_tick = (events[index + 1][0]
                         if index + 1 < len(events) else RAMP_END)
            for played in range(tick + 1, next_tick + 1):
                clock += tempo * 1e-6 / PPQ
                worst = max(worst, abs(clock - target[played]))
        assert worst <= TOLERANCE + 1e-9, curve_name


def test_ritardando_tempos_are_broadly_decreasing_speed():
    # A deceleration must spend most of its life slowing down: the BPM
    # at successive breakpoints (ignoring the final end-pinned event) is
    # non-increasing.
    events = compute_tempo_ramp(
        0, RAMP_END, 200, 100, PPQ, curve='ease_out',
        max_time_error=TOLERANCE)
    bpms = [60e6 / tempo for _, tempo in events[:-1]]
    assert all(bpms[i] >= bpms[i + 1] - 1e-6
               for i in range(len(bpms) - 1))


def test_no_redundant_consecutive_tempo_events():
    events = compute_tempo_ramp(
        0, RAMP_END, 120, 200, PPQ, max_time_error=TOLERANCE)
    tempos = [tempo for _, tempo in events]
    assert all(left != right
               for left, right in zip(tempos, tempos[1:]))
    ticks = [tick for tick, _ in events]
    assert ticks == sorted(set(ticks))


# ---------------------------------------------------------------------
# Insertion semantics
# ---------------------------------------------------------------------

def test_original_messages_keep_ticks_track_and_order():
    mid = make_multitrack_file()

    def snapshot(track):
        # Compare absolute ticks plus message content, excluding the
        # delta time which is recomputed on insertion.
        return [(tick, msg.copy(time=0))
                for tick, msg in absolute_messages(track)]

    before = [snapshot(track) for track in mid.tracks]

    add_tempo_ramp(mid, 4 * PPQ, RAMP_END, 120, 240,
                   max_time_error=TOLERANCE)

    # Track 1 is untouched (ramp went to track 0).
    assert snapshot(mid.tracks[1]) == before[1]

    # Every original track-0 message is still there at the same absolute
    # tick and in the same relative order.
    original_after = [(tick, msg) for tick, msg
                      in absolute_messages(mid.tracks[0])
                      if msg.type != 'set_tempo']
    assert [(tick, msg.copy(time=0)) for tick, msg in original_after] \
        == before[0]

    # All delta times are non-negative integers.
    for track in mid.tracks:
        for msg in track:
            assert isinstance(msg.time, int)
            assert msg.time >= 0


def test_new_events_precede_original_messages_at_same_tick():
    mid = make_multitrack_file()
    # Place a ramp whose start collides with the note_on at tick PPQ.
    inserted = add_tempo_ramp(mid, PPQ, RAMP_END, 120, 200,
                              max_time_error=TOLERANCE)

    abs0 = absolute_messages(mid.tracks[0])
    for index, (tick, msg) in enumerate(abs0):
        if tick == PPQ:
            # First message at this tick is the inserted set_tempo,
            # followed by the original messages in their old order.
            assert msg.type == 'set_tempo'
            assert abs0[index + 1][0] == PPQ
            assert abs0[index + 1][1].type == 'note_off'
            assert abs0[index + 2][0] == PPQ
            assert abs0[index + 2][1].type == 'note_on'
            break
    else:  # pragma: no cover
        raise AssertionError('ramp start tick not found')

    assert all(msg.type == 'set_tempo' for msg in inserted)


def test_tempo_continuous_before_and_after_ramp():
    mid = make_multitrack_file()
    # File defaults to 120 BPM (no explicit set_tempo), so the ramp
    # must start at 120 to keep the preceding tempo continuous.
    add_tempo_ramp(mid, 4 * PPQ, RAMP_END, 120, 200,
                   max_time_error=TOLERANCE)

    timeline = tempo_timeline(mid)
    first_tick, first_tempo = timeline[0]
    assert first_tick == 4 * PPQ
    assert first_tempo == bpm2tempo(120)

    # No tempo event exists before the ramp: default tempo still rules.
    assert all(tick >= 4 * PPQ for tick, _ in timeline)
    # The last event fixes the tempo at 200 BPM, which then continues.
    assert timeline[-1] == (RAMP_END, bpm2tempo(200))


def test_explicit_matching_boundary_event_is_reused():
    mid = make_multitrack_file()
    mid.tracks[0].insert(
        0, MetaMessage('set_tempo', tempo=bpm2tempo(120), time=0))
    # Shift the following message back to relative time 0.
    mid.tracks[0][1] = mid.tracks[0][1].copy(time=0)

    inserted = add_tempo_ramp(mid, 0, RAMP_END, 120, 240,
                              max_time_error=TOLERANCE)
    ticks = {tick for tick, _ in tempo_timeline(mid)}
    assert 0 in ticks  # original event still present
    assert all(msg.time >= 0 for msg in inserted)
    # The reused boundary event is not returned as newly inserted.
    assert all(msg is not mid.tracks[0][0] for msg in inserted)


def test_adjacent_ramps_connect():
    mid = make_multitrack_file()
    middle = RAMP_END // 2
    add_tempo_ramp(mid, 0, middle, 120, 200, max_time_error=TOLERANCE)
    # The second ramp must start at the exact quantized boundary tempo.
    boundary_bpm = mido.tempo2bpm(bpm2tempo(200))
    add_tempo_ramp(mid, middle, RAMP_END, boundary_bpm, 120,
                   max_time_error=TOLERANCE)
    timeline = tempo_timeline(mid)
    assert timeline[0] == (0, bpm2tempo(120))
    # Both ramps agree on the shared boundary tick.
    boundary = [tempo for tick, tempo in timeline if tick == middle]
    assert boundary == [bpm2tempo(200)]
    assert timeline[-1] == (RAMP_END, bpm2tempo(120))


def test_roundtrip_preserves_tempo_map_and_notes():
    mid = make_multitrack_file()
    add_tempo_ramp(mid, 2 * PPQ, RAMP_END, 120, 220,
                   curve='ease_in_out', max_time_error=TOLERANCE)

    buffer = io.BytesIO()
    mid.save(file=buffer)
    buffer.seek(0)
    reloaded = MidiFile(file=buffer)

    assert reloaded.ticks_per_beat == PPQ
    assert len(reloaded.tracks) == len(mid.tracks)
    assert tempo_timeline(reloaded) == tempo_timeline(mid)

    for original_track, loaded_track in zip(mid.tracks, reloaded.tracks):
        original = [(tick, msg) for tick, msg
                    in absolute_messages(original_track)
                    if msg.type != 'end_of_track']
        loaded = [(tick, msg) for tick, msg
                  in absolute_messages(loaded_track)
                  if msg.type != 'end_of_track']
        assert original == loaded


def test_method_on_midifile_matches_function():
    mid = make_multitrack_file()
    inserted = mid.add_tempo_ramp(0, RAMP_END, 120, 180,
                                  max_time_error=TOLERANCE)
    assert inserted
    assert tempo_timeline(mid)[0] == (0, bpm2tempo(120))


# ---------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------

def test_overlapping_ramp_rejected():
    mid = make_multitrack_file()
    add_tempo_ramp(mid, 0, RAMP_END, 120, 200, max_time_error=TOLERANCE)
    # The new range starts strictly inside the existing ramp.
    with pytest.raises(TempoRampError, match='overlapping tempo change'):
        add_tempo_ramp(mid, 4 * PPQ, RAMP_END + PPQ, 200, 120,
                       max_time_error=TOLERANCE)


def test_non_positive_bpm_rejected():
    with pytest.raises(TempoRampError, match='start_bpm'):
        compute_tempo_ramp(0, 100, 0, 120, PPQ)
    with pytest.raises(TempoRampError, match='end_bpm'):
        compute_tempo_ramp(0, 100, 120, -5, PPQ)


def test_illegal_ppq_rejected():
    with pytest.raises(TempoRampError, match='PPQ'):
        compute_tempo_ramp(0, 100, 120, 120, 0)
    with pytest.raises(TempoRampError, match='PPQ'):
        compute_tempo_ramp(0, 100, 120, 120, 1.5)
    # bool is a subclass of int but is not a meaningful PPQ.
    with pytest.raises(TempoRampError, match='PPQ'):
        compute_tempo_ramp(0, 100, 120, 120, True)


def test_tolerance_too_small_triggers_event_cap():
    # The tolerance is achievable in principle but needs more events than
    # the cap allows.
    with pytest.raises(TempoRampError, match='max_events'):
        compute_tempo_ramp(0, RAMP_END, 60, 240, PPQ,
                           max_time_error=0.0005, max_events=5)


def test_tolerance_below_ppq_resolution_rejected():
    # A sub-microsecond tolerance on a one-tick ramp is below what any
    # quantized tempo map can deliver.
    with pytest.raises(TempoRampError, match='resolution'):
        compute_tempo_ramp(0, 1, 60, 240, PPQ, max_time_error=1e-12)


def test_discontinuous_start_bpm_rejected():
    mid = make_multitrack_file()
    # File plays the 120 BPM default, but the ramp claims 90 BPM.
    with pytest.raises(TempoRampError, match='discontinuously'):
        add_tempo_ramp(mid, 0, RAMP_END, 90, 180,
                       max_time_error=TOLERANCE)


def test_conflicting_boundary_event_rejected():
    mid = make_multitrack_file()
    mid.tracks[0].insert(
        0, MetaMessage('set_tempo', tempo=bpm2tempo(100), time=0))
    with pytest.raises(TempoRampError, match='start tick'):
        add_tempo_ramp(mid, 0, RAMP_END, 120, 200,
                       max_time_error=TOLERANCE)


def test_bad_tick_range_rejected():
    with pytest.raises(TempoRampError):
        compute_tempo_ramp(100, 100, 120, 120, PPQ)
    with pytest.raises(TempoRampError):
        compute_tempo_ramp(-1, 100, 120, 120, PPQ)


def test_unknown_curve_and_bad_callable_rejected():
    with pytest.raises(TempoRampError, match='unknown curve'):
        compute_tempo_ramp(0, 100, 120, 120, PPQ, curve='wobble')
    with pytest.raises(TempoRampError, match='curve'):
        compute_tempo_ramp(0, 100, 120, 120, PPQ, curve=lambda u: 2 * u)


def test_type2_file_rejected():
    mid = MidiFile(type=2, ticks_per_beat=PPQ)
    mid.add_track()
    with pytest.raises(TempoRampError, match='type 2'):
        add_tempo_ramp(mid, 0, 100, 120, 120)


def test_track_index_out_of_range():
    mid = make_multitrack_file()
    with pytest.raises(TempoRampError, match='track index'):
        add_tempo_ramp(mid, 0, 100, 120, 120, track=7)


def test_mido_namespace_exports():
    assert mido.compute_tempo_ramp is compute_tempo_ramp
    assert mido.add_tempo_ramp is add_tempo_ramp
    assert mido.TempoRampError is TempoRampError
    assert 'linear' in mido.CURVES and 'ease_in_out' in mido.CURVES
