# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

import io

from pytest import approx, raises

from mido.midifiles import MidiFile, MidiTrack, extract_lyrics, format_lyrics
from mido.midifiles.lyrics import LyricFragment
from mido.midifiles.meta import MetaMessage
from mido.midifiles.units import tick2second

TPB = 480
DEFAULT_TEMPO = 500000


def make_file(type=1, tracks=None, ticks_per_beat=TPB):
    return MidiFile(type=type, ticks_per_beat=ticks_per_beat,
                    tracks=tracks or [])


def lyrics_track(*events):
    """Build a track from (delta_ticks, type, kwargs) tuples."""
    track = MidiTrack()
    for delta, mtype, kwargs in events:
        track.append(MetaMessage(mtype, time=delta, **kwargs))
    return track


def test_empty_file_gives_empty_timeline():
    mid = make_file(tracks=[MidiTrack()])
    assert extract_lyrics(mid) == []
    assert mid.lyrics() == []


def test_empty_lyrics_text_is_dropped():
    mid = make_file(tracks=[lyrics_track(
        (0, 'lyrics', {'text': ''}),
        (0, 'lyrics', {'text': '/'}),  # marker only, no text
        (10, 'lyrics', {'text': 'hi'}),
    )])
    fragments = extract_lyrics(mid)
    assert [(f.text, f.line) for f in fragments] == [('hi', 0)]


def test_missing_tempo_uses_default():
    mid = make_file(tracks=[lyrics_track(
        (TPB, 'lyrics', {'text': 'one'}),
        (TPB, 'lyrics', {'text': 'two'}),
    )])
    fragments = extract_lyrics(mid)
    # 120 BPM: one beat (TPB ticks) is 0.5 seconds.
    assert [f.seconds for f in fragments] == [approx(0.5), approx(1.0)]
    assert [f.tick for f in fragments] == [TPB, 2 * TPB]


def test_two_tracks_with_tempo_change():
    """Tempo track + lyrics track: seconds follow the global tempo map."""
    tempo_track = lyrics_track(
        (0, 'set_tempo', {'tempo': 500000}),
        (TPB, 'set_tempo', {'tempo': 250000}),
    )
    words = lyrics_track(
        (0, 'lyrics', {'text': 'Hel'}),
        (TPB, 'lyrics', {'text': 'lo'}),
        (TPB, 'lyrics', {'text': '/wor'}),
        (TPB, 'lyrics', {'text': 'ld'}),
    )
    mid = make_file(type=1, tracks=[tempo_track, words])

    fragments = extract_lyrics(mid)

    assert [f.track for f in fragments] == [1, 1, 1, 1]
    assert [f.tick for f in fragments] == [0, TPB, 2 * TPB, 3 * TPB]
    # First beat at 120 BPM (0.5 s), then the tempo doubles (0.25 s/beat).
    assert [f.seconds for f in fragments] == [approx(0.0), approx(0.5),
                                              approx(0.75), approx(1.0)]
    # '/' starts a new line and is stripped from the text.
    assert [f.text for f in fragments] == ['Hel', 'lo', 'wor', 'ld']
    assert [f.line for f in fragments] == [0, 0, 1, 1]
    assert all(f.type == 'lyrics' for f in fragments)


def test_type2_patterns_are_independent():
    """Each track of a type 2 file gets its own tempo map."""
    track0 = lyrics_track(
        (0, 'set_tempo', {'tempo': 500000}),
        (TPB, 'lyrics', {'text': 'slow'}),
    )
    track1 = lyrics_track(
        (0, 'set_tempo', {'tempo': 250000}),
        (TPB, 'lyrics', {'text': 'fast'}),
    )
    mid = make_file(type=2, tracks=[track0, track1])

    fragments = extract_lyrics(mid)

    # Grouped by track, each with its own tempo.
    assert [(f.track, f.text) for f in fragments] == [(0, 'slow'),
                                                      (1, 'fast')]
    assert fragments[0].seconds == approx(0.5)
    assert fragments[1].seconds == approx(0.25)


def test_same_tick_order_is_stable():
    tempo = lyrics_track((0, 'set_tempo', {'tempo': 500000}))
    track1 = lyrics_track(
        (TPB, 'lyrics', {'text': 'a'}),
        (0, 'lyrics', {'text': 'b'}),
    )
    track2 = lyrics_track(
        (TPB, 'lyrics', {'text': 'c'}),
    )
    mid = make_file(type=1, tracks=[tempo, track1, track2])

    fragments = extract_lyrics(mid)
    # Same tick: track number, then position inside the track.
    assert [(f.track, f.text) for f in fragments] == [(1, 'a'), (1, 'b'),
                                                      (2, 'c')]


def test_include_text_events():
    mid = make_file(tracks=[lyrics_track(
        (0, 'text', {'text': 'comment'}),
        (0, 'lyrics', {'text': 'sing'}),
    )])
    assert [f.text for f in extract_lyrics(mid)] == ['sing']
    fragments = extract_lyrics(mid, include_text=True)
    assert [(f.type, f.text) for f in fragments] == [('text', 'comment'),
                                                     ('lyrics', 'sing')]


def test_line_splitting_markers_and_newlines():
    mid = make_file(tracks=[lyrics_track(
        (0, 'lyrics', {'text': 'one'}),
        (0, 'lyrics', {'text': '\\two'}),    # backslash starts a line
        (0, 'lyrics', {'text': 'three\nfour\r\nfive\r\rsix'}),
    )])
    fragments = extract_lyrics(mid)
    assert [(f.text, f.line) for f in fragments] == [
        ('one', 0),
        ('two', 1),
        ('three', 1),  # no leading marker: continues the current line
        ('four', 2),
        ('five', 3),
        ('six', 4),
    ]


def test_repeated_separators_collapse():
    mid = make_file(tracks=[lyrics_track(
        (0, 'lyrics', {'text': 'one\n\n\n\ntwo'}),
        (0, 'lyrics', {'text': '//three'}),
    )])
    fragments = extract_lyrics(mid)
    # Runs of separators are a single line break; no empty fragments.
    assert [(f.text, f.line) for f in fragments] == [
        ('one', 0),
        ('two', 1),
        ('three', 2),
    ]


def test_bad_text_control_characters_removed():
    mid = make_file(tracks=[lyrics_track(
        (0, 'lyrics', {'text': '\x00he\x07llo\x1f\x7f'}),
    )])
    fragments = extract_lyrics(mid)
    assert [f.text for f in fragments] == ['hello']


def test_invalid_time_base_raises():
    for ticks_per_beat in (0, -25, -40):  # 0 and SMPTE divisions
        mid = make_file(ticks_per_beat=ticks_per_beat,
                        tracks=[lyrics_track((0, 'lyrics', {'text': 'x'}))])
        with raises(ValueError):
            extract_lyrics(mid)


def test_smpte_file_from_bytes_raises():
    # Header with SMPTE division 0xE728 (-25 fps, 40 ticks/frame).
    data = bytes.fromhex(
        '4d5468640000000600010001e728'
        '4d54726b0000000900ff05017800ff2f00'
    )
    mid = MidiFile(file=io.BytesIO(data))
    assert mid.ticks_per_beat < 0
    with raises(ValueError):
        extract_lyrics(mid)


def test_nonpositive_tempo_raises():
    mid = make_file(tracks=[lyrics_track(
        (0, 'set_tempo', {'tempo': 0}),
        (0, 'lyrics', {'text': 'x'}),
    )])
    with raises(ValueError):
        extract_lyrics(mid)


def test_abnormally_large_tick():
    huge = 2 ** 40
    mid = make_file(tracks=[lyrics_track(
        (huge, 'lyrics', {'text': 'far away'}),
    )])
    (fragment,) = extract_lyrics(mid)
    assert fragment.tick == huge
    assert fragment.seconds == approx(
        tick2second(huge, TPB, DEFAULT_TEMPO))


def test_file_is_not_modified():
    track = lyrics_track(
        (0, 'set_tempo', {'tempo': 400000}),
        (TPB, 'lyrics', {'text': '/hello\nworld'}),
    )
    mid = make_file(tracks=[track])
    before = [(msg.type, msg.time, msg.bytes()) for msg in mid.tracks[0]]

    extract_lyrics(mid)
    mid.lyrics(include_text=True)

    after = [(msg.type, msg.time, msg.bytes()) for msg in mid.tracks[0]]
    assert before == after
    assert len(mid.tracks) == 1


def test_format_lyrics_preview():
    mid = make_file(tracks=[lyrics_track(
        (0, 'lyrics', {'text': '/Hel'}),
        (TPB, 'lyrics', {'text': 'lo'}),
    )])
    fragments = mid.lyrics()
    preview = format_lyrics(fragments)
    lines = preview.splitlines()
    assert len(lines) == 2
    assert 'track 0' in lines[0] and 'line 0' in lines[0]
    assert 'Hel' in lines[0] and 'lo' in lines[1]
    assert str(fragments[0]).endswith('Hel')


def test_fragment_fields():
    fragment = LyricFragment(track=2, tick=10, seconds=0.5,
                             text='la', type='lyrics', line=3)
    assert fragment.track == 2
    assert fragment.tick == 10
    assert fragment.seconds == 0.5
    assert fragment.text == 'la'
    assert fragment.type == 'lyrics'
    assert fragment.line == 3
