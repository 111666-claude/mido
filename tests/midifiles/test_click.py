# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

import pytest

from mido import (
    Message,
    MetaMessage,
    MidiFile,
    MidiTrack,
    bpm2tempo,
    make_click_track,
    tick2second,
)

TPB = 480  # ticks per beat used by all fixtures


def note_on(tick, **kwargs):
    return Message('note_on', time=tick, **kwargs)


def make_file(tracks, type=1, ticks_per_beat=TPB):
    return MidiFile(type=type, ticks_per_beat=ticks_per_beat, tracks=tracks)


@pytest.fixture
def tempo_change_file():
    """4/4 with a tempo change from 120 to 240 BPM after two bars."""
    conductor = MidiTrack([
        MetaMessage('set_tempo', tempo=bpm2tempo(120), time=0),
        note_on(0, note=60, velocity=64),  # some content on the conductor
        Message('note_off', note=60, velocity=0, time=TPB),
        MetaMessage('set_tempo', tempo=bpm2tempo(240), time=7 * TPB),
        MetaMessage('end_of_track', time=8 * TPB),
    ])
    piano = MidiTrack([
        note_on(0, note=64, velocity=64),
        Message('note_off', note=64, velocity=0, time=16 * TPB),
        MetaMessage('end_of_track', time=0),
    ])
    return make_file([conductor, piano])


@pytest.fixture
def meter_change_file():
    """Two bars of 4/4, then 3/4 starting on the old beat grid."""
    conductor = MidiTrack([
        MetaMessage('time_signature', numerator=4, denominator=4, time=0),
        MetaMessage('time_signature', numerator=3, denominator=4,
                    time=8 * TPB),
        MetaMessage('end_of_track', time=12 * TPB),
    ])
    return make_file([conductor])


@pytest.fixture
def pickup_file():
    """4/4 where the music enters on beat 4 of the first bar (pickup)."""
    conductor = MidiTrack([
        MetaMessage('set_tempo', tempo=bpm2tempo(120), time=0),
        MetaMessage('end_of_track', time=8 * TPB),
    ])
    melody = MidiTrack([
        MetaMessage('track_name', name='melody', time=0),
        note_on(3 * TPB, note=67, velocity=80),  # pickup on beat 4
        Message('note_off', note=67, velocity=0, time=TPB),
        note_on(0, note=69, velocity=80),
        Message('note_off', note=69, velocity=0, time=4 * TPB),
        MetaMessage('end_of_track', time=0),
    ])
    return make_file([conductor, melody])


def click_notes(track):
    """(abs_tick, is_accent_note, velocity) for each note_on."""
    now = 0
    notes = []
    for msg in track:
        now += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            notes.append((now, msg.note, msg.velocity))
    return notes


def click_ticks(track):
    return [tick for tick, _, _ in click_notes(track)]


def click_seconds(mid, click_track, channel=9):
    """Playback time in seconds of every click note_on."""
    mid = MidiFile(type=mid.type, ticks_per_beat=mid.ticks_per_beat,
                   tracks=[track.copy() for track in mid.tracks]
                   + [click_track])
    now = 0.0
    seconds = []
    for msg in mid:
        now += msg.time
        if (msg.type == 'note_on' and msg.velocity > 0
                and getattr(msg, 'channel', None) == channel):
            seconds.append(now)
    return seconds


def seconds_at_ticks(mid, ticks):
    """Expected seconds for absolute ticks, walking the tempo map."""
    tempo = 500000
    now_tick = 0
    now_sec = 0.0
    changes = []
    pos = 0
    for msg in mid.merged_track:
        pos += msg.time
        if msg.type == 'set_tempo':
            changes.append((pos, msg.tempo))
    result = []
    for tick in ticks:
        for change_tick, new_tempo in changes:
            if change_tick <= now_tick or change_tick > tick:
                continue
            now_sec += tick2second(change_tick - now_tick, TPB, tempo)
            now_tick = change_tick
            tempo = new_tempo
        result.append(now_sec + tick2second(tick - now_tick, TPB, tempo))
    return result


# --- basic generation -------------------------------------------------

def test_default_click_track_4_4():
    mid = make_file([MidiTrack([
        MetaMessage('end_of_track', time=4 * TPB),
    ])])
    track = make_click_track(mid)
    assert click_ticks(track) == [0, TPB, 2 * TPB, 3 * TPB]
    notes = click_notes(track)
    # First beat of the bar is accented, the rest are weak.
    assert notes[0][1:] == (34, 96)
    assert all((note, vel) == (33, 64) for _, note, vel in notes[1:])


def test_returns_independent_track_and_does_not_touch_file(
        tempo_change_file):
    before = [repr(track) for track in tempo_change_file.tracks]
    num_tracks = len(tempo_change_file.tracks)

    track = make_click_track(tempo_change_file)

    assert isinstance(track, MidiTrack)
    assert len(tempo_change_file.tracks) == num_tracks
    assert [repr(t) for t in tempo_change_file.tracks] == before
    # Editing the click track must not affect the file.
    track.append(note_on(0, note=1, velocity=1))
    assert [repr(t) for t in tempo_change_file.tracks] == before


def test_track_is_editable_and_named(tempo_change_file):
    track = make_click_track(tempo_change_file)
    assert track.name == 'click'
    track.name = 'metronome'
    assert track.name == 'metronome'
    # All messages are plain notes/meta with non-negative deltas.
    assert all(msg.time >= 0 for msg in track)


def test_custom_accent_and_weak(tempo_change_file):
    track = make_click_track(tempo_change_file, channel=0,
                             accent_note=76, accent_velocity=100,
                             accent_duration=30,
                             note=77, velocity=50, duration=10)
    notes = click_notes(track)
    accents = [(note, vel) for tick, note, vel in notes
               if tick % (4 * TPB) == 0]
    weak = [(note, vel) for tick, note, vel in notes
            if tick % (4 * TPB) != 0]
    assert accents and all(accent == (76, 100) for accent in accents)
    assert weak and all(w == (77, 50) for w in weak)

    # Durations: note_off lands exactly accent_duration/duration later.
    now = 0
    events = []
    for msg in track:
        now += msg.time
        if msg.type in ('note_on', 'note_off'):
            events.append((now, msg.type, msg.note))
    ons = [e for e in events if e[1] == 'note_on']
    offs = [e for e in events if e[1] == 'note_off']
    assert offs[0][0] - ons[0][0] == 30
    assert offs[1][0] - ons[1][0] == 10
    assert all(msg.channel == 0 for msg in track if not msg.is_meta)


def test_default_duration_is_ticks_per_beat_div_8(tempo_change_file):
    track = make_click_track(tempo_change_file)
    now = 0
    on_tick = off_tick = None
    for msg in track:
        now += msg.time
        if msg.type == 'note_on' and on_tick is None:
            on_tick = now
        elif msg.type == 'note_off' and off_tick is None:
            off_tick = now
    assert off_tick - on_tick == TPB // 8


# --- start offset and end tick ----------------------------------------

def test_start_and_end_half_open(tempo_change_file):
    track = make_click_track(tempo_change_file, start=TPB, end=5 * TPB)
    assert click_ticks(track) == [TPB, 2 * TPB, 3 * TPB, 4 * TPB]


def test_start_mid_beat_emits_nothing_before_next_beat(tempo_change_file):
    track = make_click_track(tempo_change_file, start=TPB + 1,
                             end=2 * TPB + 1)
    assert click_ticks(track) == [2 * TPB]


def test_end_defaults_to_longest_track(tempo_change_file):
    track = make_click_track(tempo_change_file)
    # Piano track ends at 16 * TPB.
    assert click_ticks(track)[-1] == 15 * TPB


def test_zero_length_interval_has_no_notes(tempo_change_file):
    track = make_click_track(tempo_change_file, start=4 * TPB, end=4 * TPB)
    assert click_notes(track) == []
    assert [msg.type for msg in track] == ['track_name', 'end_of_track']


def test_empty_file():
    track = make_click_track(make_file([MidiTrack()]))
    assert click_notes(track) == []


# --- tempo changes ----------------------------------------------------

def test_tempo_change_keeps_tick_grid(tempo_change_file):
    track = make_click_track(tempo_change_file)
    ticks = click_ticks(track)
    assert ticks == [i * TPB for i in range(16)]
    # Accents every 4 beats regardless of the tempo change.
    accents = [tick for tick, note, _ in click_notes(track) if note == 34]
    assert accents == [0, 4 * TPB, 8 * TPB, 12 * TPB]


def test_tempo_change_click_seconds(tempo_change_file):
    track = make_click_track(tempo_change_file)
    ticks = click_ticks(track)
    seconds = click_seconds(tempo_change_file, track)
    assert seconds == pytest.approx(seconds_at_ticks(tempo_change_file,
                                                     ticks))
    # Spot check: tempo change to 240 BPM at tick 8 * TPB.
    assert seconds[8] == pytest.approx(8 * 0.5)          # 8 beats @ 120
    assert seconds[9] == pytest.approx(8 * 0.5 + 0.25)   # then 0.25 s/beat


# --- time signature changes -------------------------------------------

def test_meter_change_resets_bar(meter_change_file):
    track = make_click_track(meter_change_file)
    ticks = click_ticks(track)
    # 4/4 for 2 bars, then 3/4: beat grid is unbroken (same denominator).
    assert ticks == [i * TPB for i in range(20)]
    accents = [tick for tick, note, _ in click_notes(track) if note == 34]
    # Accents at bar lines: 0, 4, 8 (4/4) then every 3 beats.
    assert accents == [0, 4 * TPB, 8 * TPB, 11 * TPB, 14 * TPB, 17 * TPB]


def test_meter_change_off_beat_grid():
    mid = make_file([MidiTrack([
        MetaMessage('time_signature', numerator=3, denominator=4,
                    time=3 * TPB),
        MetaMessage('end_of_track', time=6 * TPB),
    ])])
    track = make_click_track(mid)
    # New bar starts at the event tick, even mid-bar.
    assert click_ticks(track) == [0, TPB, 2 * TPB, 3 * TPB, 4 * TPB,
                                  5 * TPB, 6 * TPB, 7 * TPB, 8 * TPB]
    accents = [tick for tick, note, _ in click_notes(track) if note == 34]
    assert accents == [0, 3 * TPB, 6 * TPB]


def test_same_tick_meta_applies_before_click():
    # The 3/4 event sits exactly where a 4/4 weak beat would fall.
    mid = make_file([MidiTrack([
        MetaMessage('time_signature', numerator=3, denominator=4,
                    time=4 * TPB),
        MetaMessage('end_of_track', time=6 * TPB),
    ])])
    track = make_click_track(mid)
    notes = {tick: note for tick, note, _ in click_notes(track)}
    # The click at 4 * TPB uses the new signature and is accented.
    assert notes[4 * TPB] == 34
    # And the next accent follows after 3 beats, not 4.
    accents = [tick for tick, note, _ in click_notes(track) if note == 34]
    assert accents == [0, 4 * TPB, 7 * TPB]


def test_compound_meter_6_8():
    mid = make_file([MidiTrack([
        MetaMessage('time_signature', numerator=6, denominator=8, time=0),
        MetaMessage('end_of_track', time=6 * TPB),
    ])])
    track = make_click_track(mid)
    eighth = TPB // 2
    assert click_ticks(track) == [i * eighth for i in range(12)]
    accents = [tick for tick, note, _ in click_notes(track) if note == 34]
    assert accents == [0, 6 * eighth]


# --- pickup (anacrusis) -------------------------------------------------

def test_pickup_start_offset_gives_weak_beats_until_barline(pickup_file):
    # The pickup enters on beat 4 of the bar anchored at tick 0.
    track = make_click_track(pickup_file, start=3 * TPB, end=8 * TPB)
    notes = click_notes(track)
    assert [tick for tick, _, _ in notes] == [3 * TPB] + [
        i * TPB for i in range(4, 8)]
    # The pickup click itself is weak; the downbeat lands on the barline.
    assert notes[0][1] == 33
    assert notes[1][1] == 34  # tick 4 * TPB: bar 2 downbeat


def test_pickup_click_seconds(pickup_file):
    track = make_click_track(pickup_file, start=3 * TPB)
    seconds = click_seconds(pickup_file, track)
    # 120 BPM the whole way: 0.5 s per beat, pickup click at 1.5 s.
    assert seconds[:2] == pytest.approx([1.5, 2.0])


# --- multi-track conflicts --------------------------------------------

def test_conflicting_signatures_same_tick_last_track_wins():
    track_a = MidiTrack([
        MetaMessage('time_signature', numerator=3, denominator=4,
                    time=2 * TPB),
        MetaMessage('end_of_track', time=10 * TPB),
    ])
    track_b = MidiTrack([
        MetaMessage('time_signature', numerator=5, denominator=4,
                    time=2 * TPB),
        MetaMessage('end_of_track', time=10 * TPB),
    ])
    mid = make_file([track_a, track_b])
    track = make_click_track(mid)
    accents = [tick for tick, note, _ in click_notes(track) if note == 34]
    # 5/4 (the later track) wins: accents at 0, 2, 7.
    assert accents == [0, 2 * TPB, 7 * TPB]


def test_signatures_on_conductor_track_only(tempo_change_file):
    # Meta events spread over tracks are all honored.
    mid = make_file([
        MidiTrack([
            MetaMessage('time_signature', numerator=2, denominator=4,
                        time=0),
            MetaMessage('end_of_track', time=8 * TPB),
        ]),
        MidiTrack([
            note_on(0, note=60, velocity=64),
            Message('note_off', note=60, velocity=0, time=8 * TPB),
            MetaMessage('end_of_track', time=0),
        ]),
    ])
    track = make_click_track(mid)
    accents = [tick for tick, note, _ in click_notes(track) if note == 34]
    assert accents == [0, 2 * TPB, 4 * TPB, 6 * TPB]


# --- invalid input ------------------------------------------------------

def test_type_2_raises_type_error():
    mid = make_file([MidiTrack(), MidiTrack()], type=2)
    with pytest.raises(TypeError):
        make_click_track(mid)


@pytest.mark.parametrize('kwargs', [
    {'start': -1},
    {'start': 1.5},
    {'start': '0'},
    {'end': -1},
    {'end': 1.5},
    {'start': 4 * TPB, 'end': 2 * TPB},
    {'channel': -1},
    {'channel': 16},
    {'accent_note': 128},
    {'note': -1},
    {'accent_velocity': 0},
    {'velocity': 128},
    {'accent_duration': 0},
    {'duration': -5},
])
def test_invalid_arguments_raise_value_error(tempo_change_file, kwargs):
    with pytest.raises(ValueError):
        make_click_track(tempo_change_file, **kwargs)
