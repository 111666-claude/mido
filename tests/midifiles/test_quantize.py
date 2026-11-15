# SPDX-FileCopyrightText: 2026
#
# SPDX-License-Identifier: MIT

import io

import pytest

from mido import (
    Message,
    MetaMessage,
    MidiFile,
    MidiTrack,
    quantize_notes,
)
from mido.midifiles.quantize import (
    _Grid,
    _time_signature_regions,
)

TPB = 480


def make_file(tracks, file_type=1, ticks_per_beat=TPB):
    return MidiFile(
        type=file_type,
        ticks_per_beat=ticks_per_beat,
        tracks=[MidiTrack(track) for track in tracks],
    )


def abs_ticks(track):
    """Return (absolute_tick, message) pairs for a track."""
    result = []
    now = 0
    for msg in track:
        now += msg.time
        result.append((now, msg))
    return result


def note_events(track):
    """Return ``(type, channel, note, absolute_tick)`` for note msgs."""
    return [
        (msg.type, msg.channel, msg.note, tick)
        for tick, msg in abs_ticks(track)
        if msg.type in ('note_on', 'note_off')
    ]


def rel_deltas(track):
    return [msg.time for msg in track]


def test_basic_grid_snapping():
    # Quarter-note grid (one slot per beat); spacing is 480 ticks.
    track = [
        Message('note_on', note=60, time=20),       # abs 20 -> 0
        Message('note_off', note=60, time=430),     # abs 450 -> 480
        Message('note_on', note=62, time=60),       # abs 510 -> 480
        Message('note_off', note=62, time=900),     # abs 1410 -> 1440
    ]
    mid = make_file([track])
    out = quantize_notes(mid, subdivision=1)

    assert note_events(out.tracks[0]) == [
        ('note_on', 0, 60, 0),
        ('note_off', 0, 60, 480),
        ('note_on', 0, 62, 480),
        ('note_off', 0, 62, 1440),
    ]


def test_round_half_up():
    # Half a slot (240 ticks) rounds up, never towards negative time.
    track = [
        Message('note_on', note=60, time=240),      # -> 480
        Message('note_off', note=60, time=240),     # abs 480 -> 480
    ]
    mid = make_file([track])
    out = quantize_notes(mid, subdivision=1)

    assert note_events(out.tracks[0]) == [
        ('note_on', 0, 60, 480),
        ('note_off', 0, 60, 481),  # positive duration enforced
    ]


def test_sixteenth_note_subdivision():
    spacing = TPB // 4  # 120 ticks
    track = [
        Message('note_on', note=60, time=125),      # -> 120
        Message('note_off', note=60, time=62),      # abs 187 -> 240
    ]
    mid = make_file([track])
    out = quantize_notes(mid, subdivision=4)

    assert note_events(out.tracks[0]) == [
        ('note_on', 0, 60, spacing),
        ('note_off', 0, 60, spacing * 2),
    ]


def test_positive_duration_when_endpoints_collide():
    track = [
        Message('note_on', note=60, time=10),
        Message('note_off', note=60, time=10),      # both quantize to 0
    ]
    mid = make_file([track])
    out = quantize_notes(mid, subdivision=1)
    events = note_events(out.tracks[0])

    assert events[0][3] == 0
    assert events[1][3] == 1


def test_velocity_zero_note_on_pairs_as_note_off():
    track = [
        Message('note_on', note=64, velocity=100, time=5),
        Message('note_on', note=64, velocity=0, time=TPB),
    ]
    mid = make_file([track])
    out = quantize_notes(mid, subdivision=1)

    assert note_events(out.tracks[0]) == [
        ('note_on', 0, 64, 0),
        ('note_on', 0, 64, TPB),  # message type preserved, time quantized
    ]


def test_overlapping_same_note_uses_lifo_pairing():
    # Two overlapping identical notes; LIFO pairing must match the
    # inner note first.
    track = [
        Message('note_on', note=60, time=10),       # A on
        Message('note_on', note=60, time=10),       # B on (abs 20)
        Message('note_off', note=60, time=TPB - 20),  # closes B at 480
        Message('note_off', note=60, time=TPB),       # closes A at 960
    ]
    mid = make_file([track])
    out = quantize_notes(mid, subdivision=1)
    events = note_events(out.tracks[0])

    on_times = [tick for typ, _ch, _n, tick in events if typ == 'note_on']
    off_times = [tick for typ, _ch, _n, tick in events if typ == 'note_off']
    assert on_times == [0, 0]
    # B's off at 480 and A's off at 960.
    assert off_times == [480, 960]


def test_channels_and_notes_are_pairing_keys():
    # Same note number on two channels must not cross-pair.
    track = [
        Message('note_on', note=60, channel=0, time=10),
        Message('note_on', note=60, channel=1, time=10),
        Message('note_off', note=60, channel=1, time=TPB - 20),
        Message('note_off', note=60, channel=0, time=TPB),
    ]
    mid = make_file([track])
    out = quantize_notes(mid, subdivision=1)
    events = note_events(out.tracks[0])

    assert [(ch, tick) for typ, ch, _n, tick in events
            if typ == 'note_off'] == [(1, TPB), (0, TPB * 2)]


def test_unpaired_note_off_raises_in_strict_mode():
    track = [Message('note_off', note=60, time=100)]
    mid = make_file([track])
    with pytest.raises(ValueError, match='could not be paired'):
        quantize_notes(mid, subdivision=1)


def test_unpaired_note_on_raises_in_strict_mode():
    track = [Message('note_on', note=60, time=100)]
    mid = make_file([track])
    with pytest.raises(ValueError, match='could not be paired'):
        quantize_notes(mid, subdivision=1)


def test_unpaired_notes_kept_in_lenient_mode():
    track = [
        Message('note_on', note=60, time=100),
        Message('note_off', note=62, time=100),
    ]
    mid = make_file([track])
    out = quantize_notes(mid, subdivision=1, strict=False)

    # Unpaired messages stay at their original absolute time.
    assert note_events(out.tracks[0]) == [
        ('note_on', 0, 60, 100),
        ('note_off', 0, 62, 200),
    ]
    assert all(delta >= 0 for delta in rel_deltas(out.tracks[0]))


def test_non_note_events_keep_absolute_time_and_order():
    track = [
        MetaMessage('set_tempo', tempo=400000, time=0),
        Message('control_change', control=7, value=100, time=333),
        Message('note_on', note=60, time=67),          # abs 400
        Message('program_change', program=5, time=80),  # abs 480
        Message('note_off', note=60, time=80),          # abs 560
    ]
    mid = make_file([track])
    out = quantize_notes(mid, subdivision=1)

    kept = {tick: msg for tick, msg in abs_ticks(out.tracks[0])}
    assert any(msg.type == 'set_tempo' and msg.tempo == 400000 and tick == 0
               for tick, msg in kept.items())
    assert any(msg.type == 'control_change' and tick == 333
               for tick, msg in kept.items())
    assert any(msg.type == 'program_change' and tick == 480
               for tick, msg in kept.items())

    # Note endpoints moved, everything else stayed put.
    assert note_events(out.tracks[0]) == [
        ('note_on', 0, 60, 480),
        ('note_off', 0, 60, 481),
    ]


def test_time_signature_change_resets_bar_grid():
    # 4/4 for one bar (1920 ticks), then 3/4 (1440 ticks per bar).
    track = [
        MetaMessage('time_signature', numerator=3, denominator=4,
                    time=1920),
        # Note just inside the 3/4 bar: 1930 with quarter grid -> 1920.
        Message('note_on', note=60, time=10),
        # abs 2180 (260 ticks into the bar) rounds up to the quarter
        # grid line at 2400.  The important check is that 2400 is a
        # grid line of the new 3/4 bar (1920 + one beat) rather than
        # the old 4/4 layout.
        Message('note_off', note=60, time=250),
    ]
    mid = make_file([track])
    regions = _time_signature_regions([mid.tracks[0]])
    assert regions == [(0, 4, 4), (1920, 3, 4)]

    out = quantize_notes(mid, subdivision=1)
    assert note_events(out.tracks[0]) == [
        ('note_on', 0, 60, 1920),
        ('note_off', 0, 60, 2400),
    ]


def test_grid_inside_changed_bar_with_eighth_subdivision():
    # With subdivision=2 the grid spacing is 240 ticks, so the first
    # grid line inside the 3/4 bar is exactly 2160.
    track = [
        MetaMessage('time_signature', numerator=3, denominator=4,
                    time=1920),
        Message('note_on', note=60, time=10),       # abs 1930 -> 1920
        Message('note_off', note=60, time=240),     # abs 2170 -> 2160
    ]
    mid = make_file([track])
    out = quantize_notes(mid, subdivision=2)
    assert note_events(out.tracks[0]) == [
        ('note_on', 0, 60, 1920),
        ('note_off', 0, 60, 2160),
    ]


def test_time_signature_change_tick_on_bar_boundary():
    # A change at 3840 (exactly two 4/4 bars) must not create a zero or
    # duplicate bar: the 3/4 region starts on the boundary itself.
    track = [
        MetaMessage('time_signature', numerator=3, denominator=4,
                    time=3840),
        Message('note_on', note=60, time=10),        # abs 3850 -> 3840
        Message('note_off', note=60, time=TPB - 10),  # abs 4320 stays
    ]
    mid = make_file([track])
    out = quantize_notes(mid, subdivision=1)
    assert note_events(out.tracks[0]) == [
        ('note_on', 0, 60, 3840),
        ('note_off', 0, 60, 4320),
    ]


def test_denominator_changes_bar_length():
    # 6/8 bar: 6 * (4/8) * 480 = 1440 ticks.
    track = [
        MetaMessage('time_signature', numerator=6, denominator=8, time=0),
        Message('note_on', note=60, time=1430),   # -> 1440
        Message('note_off', note=60, time=20),    # abs 1450 -> 1440+
    ]
    mid = make_file([track])
    out = quantize_notes(mid, subdivision=1)
    events = note_events(out.tracks[0])

    assert events[0][3] == 1440
    assert events[1][3] == 1441


def test_fractional_bar_keeps_grid_aligned():
    # 7/8 bars are 1680 ticks long (14 sixteenth slots of 120 ticks).
    grid = _Grid([(0, 7, 8)], TPB, subdivision=4, swing=0.5)
    assert grid.quantize(119) == 120
    assert grid.quantize(1679) == 1680   # end of the first 7/8 bar
    assert grid.quantize(1680) == 1680   # next bar starts on the line
    assert grid.quantize(1800) == 1800


def test_swing_below_half_pulls_odd_slots_earlier():
    # swing 1/3: odd slot lands 1/3 into the cell (80 ticks instead of
    # the straight 120).
    grid = _Grid([(0, 4, 4)], TPB, subdivision=4, swing=1 / 3)
    assert grid.quantize(120) == 80
    assert grid.quantize(240) == 240


def test_swing_delays_odd_slots():
    # Sixteenth grid (120-tick spacing), swing 2/3: odd slots sit at
    # 2/3 of the cell instead of 1/2.
    grid = _Grid([(0, 4, 4)], TPB, subdivision=4, swing=2 / 3)

    assert grid.quantize(0) == 0
    # Slot 1 straight position is 120; swung position is 160.
    assert grid.quantize(120) == 160
    assert grid.quantize(240) == 240
    assert grid.quantize(360) == 400


def test_swing_half_is_straight_grid():
    grid = _Grid([(0, 4, 4)], TPB, subdivision=4, swing=0.5)
    for tick in (0, 120, 240, 360, 480):
        assert grid.quantize(tick) == tick


def test_swing_phase_restarts_at_bar_line():
    # 4/4, sixteenth grid: a bar holds 16 slots. The first slot of the
    # next bar is even again and stays unshifted at 1920.
    grid = _Grid([(0, 4, 4)], TPB, subdivision=4, swing=2 / 3)
    assert grid.quantize(1920) == 1920
    # First odd slot of the new bar lands at 1920 + (2/3 * 240) = 2080.
    assert grid.quantize(2040) == 2080
    # Slot 2 is even and stays on the straight grid.
    assert grid.quantize(2160) == 2160


@pytest.mark.parametrize('bad_subdivision', [1.5, '4', True, None])
def test_subdivision_wrong_type(bad_subdivision):
    mid = make_file([[]])
    with pytest.raises(TypeError, match='subdivision'):
        quantize_notes(mid, subdivision=bad_subdivision)


@pytest.mark.parametrize('bad_subdivision', [0, -1, -16])
def test_subdivision_out_of_range(bad_subdivision):
    mid = make_file([[]])
    with pytest.raises(ValueError, match='subdivision'):
        quantize_notes(mid, subdivision=bad_subdivision)


@pytest.mark.parametrize('bad_swing', ['0.6', True, None])
def test_swing_wrong_type(bad_swing):
    mid = make_file([[]])
    with pytest.raises(TypeError, match='swing'):
        quantize_notes(mid, subdivision=4, swing=bad_swing)


@pytest.mark.parametrize('bad_swing', [0, 1, -0.5, 1.5])
def test_swing_out_of_range(bad_swing):
    mid = make_file([[]])
    with pytest.raises(ValueError, match='swing'):
        quantize_notes(mid, subdivision=4, swing=bad_swing)


def test_all_delta_times_non_negative_after_quantize():
    # Dense, deliberately messy input across several bars.
    track = []
    for start in (5, 500, 990, 2500, 3001):
        track.append(Message('note_on', note=60, time=start))
        track.append(Message('note_off', note=60, time=70))
    mid = make_file([track])
    out = quantize_notes(mid, subdivision=2, swing=0.65)

    assert all(msg.time >= 0 for msg in out.tracks[0])


def test_multi_track_non_note_events_keep_absolute_time():
    conductor = MidiTrack([
        MetaMessage('set_tempo', tempo=400000, time=0),
        MetaMessage('time_signature', numerator=3, denominator=4,
                    time=TPB * 4),
        MetaMessage('end_of_track', time=10),
    ])
    piano = MidiTrack([
        Message('note_on', note=60, time=77),
        Message('control_change', control=64, value=80, time=3),
        Message('note_off', note=60, time=TPB - 80),
    ])
    mid = MidiFile(type=1, ticks_per_beat=TPB, tracks=[conductor, piano])
    out = quantize_notes(mid, subdivision=1)

    conductor_ticks = abs_ticks(out.tracks[0])
    assert conductor_ticks[0][0] == 0
    assert conductor_ticks[1][0] == TPB * 4
    # end_of_track is recreated by fix_end_of_track() at save time.
    assert all(msg.type != 'end_of_track' for msg in out.tracks[0])
    buffer = io.BytesIO()
    out.save(file=buffer)
    buffer.seek(0)
    reloaded = MidiFile(file=buffer)
    assert reloaded.tracks[0][-1].type == 'end_of_track'

    # The control change inside the piano track did not move.
    piano_kept = {tick: msg for tick, msg in abs_ticks(out.tracks[1])}
    assert any(msg.type == 'control_change' and tick == 80
               for tick, msg in piano_kept.items())


def test_save_and_reload_roundtrip():
    track = [
        MetaMessage('track_name', name='quantized', time=0),
        Message('note_on', note=60, time=21),
        Message('note_off', note=60, time=TPB - 21),
        Message('note_on', note=67, time=13),
        Message('note_off', note=67, time=TPB - 13),
        MetaMessage('end_of_track', time=0),
    ]
    mid = make_file([track])
    quantized = quantize_notes(mid, subdivision=2, swing=0.6)

    buffer = io.BytesIO()
    quantized.save(file=buffer)
    buffer.seek(0)
    reloaded = MidiFile(file=buffer)

    assert reloaded.ticks_per_beat == TPB
    assert len(reloaded.tracks) == 1
    assert note_events(reloaded.tracks[0]) == \
        note_events(quantized.tracks[0])
    assert all(msg.time >= 0 for msg in reloaded.tracks[0])
    # The track name meta event survived the round trip unchanged.
    assert reloaded.tracks[0][0].type == 'track_name'
    assert reloaded.tracks[0][0].name == 'quantized'


def test_original_file_is_not_modified():
    track = MidiTrack([
        Message('note_on', note=60, time=33),
        Message('note_off', note=60, time=417),
    ])
    mid = make_file([track])
    before = [(msg.type, msg.note, msg.time) for msg in mid.tracks[0]]

    quantize_notes(mid, subdivision=1)

    after = [(msg.type, msg.note, msg.time) for msg in mid.tracks[0]]
    assert before == after


def test_method_entry_matches_function():
    mid = make_file([[
        Message('note_on', note=60, time=21),
        Message('note_off', note=60, time=459),
    ]])
    via_method = mid.quantize_notes(subdivision=1)
    via_function = quantize_notes(mid, subdivision=1)

    assert note_events(via_method.tracks[0]) == \
        note_events(via_function.tracks[0])


def test_type_two_file_quantizes_tracks_independently():
    track_a = MidiTrack([
        Message('note_on', note=60, time=30),
        Message('note_off', note=60, time=TPB - 30),
    ])
    track_b = MidiTrack([
        Message('note_on', note=72, time=470),
        Message('note_off', note=72, time=10),
    ])
    mid = MidiFile(type=2, ticks_per_beat=TPB,
                   tracks=[track_a, track_b])
    out = quantize_notes(mid, subdivision=1)

    assert out.type == 2
    assert note_events(out.tracks[0]) == [
        ('note_on', 0, 60, 0),
        ('note_off', 0, 60, TPB),
    ]
    assert note_events(out.tracks[1]) == [
        ('note_on', 0, 72, TPB),
        ('note_off', 0, 72, TPB + 1),
    ]


def test_type_zero_single_track():
    mid = make_file([[
        Message('note_on', note=60, time=19),
        Message('note_off', note=60, time=TPB - 19),
    ]], file_type=0)
    out = quantize_notes(mid, subdivision=1)

    assert out.type == 0
    assert note_events(out.tracks[0]) == [
        ('note_on', 0, 60, 0),
        ('note_off', 0, 60, TPB),
    ]


def test_quantized_tick_beyond_vlk_limit_raises(monkeypatch):
    # Shrink the representable limit so an ordinary quarter-grid note
    # quantizes past it, exercising the real bounds check.
    import mido.midifiles.quantize as quantize_module

    monkeypatch.setattr(quantize_module, 'MAX_VLQ_VALUE', 100)
    mid = make_file([[
        Message('note_on', note=60, time=10),
        Message('note_off', note=60, time=TPB - 10),
    ]])
    with pytest.raises(ValueError, match='variable-length quantity'):
        quantize_module.quantize_notes(mid, subdivision=1)
