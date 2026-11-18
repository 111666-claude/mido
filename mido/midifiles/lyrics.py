# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Lyrics timeline extraction for karaoke MIDI files.

Karaoke files scatter ``lyrics`` meta messages (and sometimes ``text``
meta messages) across several tracks. This module collects them into a
single, directly previewable timeline of lyric fragments with absolute
tick and second timestamps, track numbers and line boundaries.

Line boundary conventions (the de-facto karaoke standard):

* A fragment starting with ``/`` starts a new line (paragraph).
* A fragment starting with ``\\`` starts a new line (within a paragraph).
* Newlines (``\\n``, ``\\r`` or ``\\r\\n``) inside the text split the
  fragment into lines.
* Runs of consecutive separators (e.g. ``\\n\\n`` or ``//``) collapse
  into a single line break; empty fragments are never emitted.

Tempo handling:

* Type 0 and 1 files use one global tempo map built from the
  ``set_tempo`` events of all tracks.
* Type 2 (asynchronous) files keep each pattern independent: every
  track gets its own tempo map built only from that track's
  ``set_tempo`` events, and fragments are grouped by track.
* If no tempo is given the MIDI default of 500000 microseconds per
  beat (120 BPM) is used.

The input file is never modified.
"""

from dataclasses import dataclass

from .midifiles import DEFAULT_TEMPO
from .units import tick2second

# Karaoke line-start markers are '/' and '\'; a run of them at the
# start of a fragment collapses into a single line break.


@dataclass(frozen=True)
class LyricFragment:
    """One piece of lyric text on the timeline.

    Attributes:
        track: Index of the track the fragment came from.
        tick: Absolute time in ticks.
        seconds: Absolute time in seconds (via the tempo map).
        text: The lyric text with line markers and control
            characters removed.
        type: Meta message type, ``'lyrics'`` or ``'text'``.
        line: Line number (0-based) the fragment belongs to.
    """
    track: int
    tick: int
    seconds: float
    text: str
    type: str
    line: int

    def __str__(self):
        return (f'[{self.seconds:9.3f}s track {self.track} '
                f'line {self.line}] {self.text}')


def _clean_text(text):
    """Normalize newlines and strip other control characters.

    This makes the result deterministic for malformed text: ``\\r\\n``
    and ``\\r`` become ``\\n`` and any other C0/DEL control characters
    are removed.
    """
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return ''.join(char for char in text
                   if char == '\n' or (ord(char) >= 32 and ord(char) != 127))


def _pieces_of(text):
    """Split cleaned text into (starts_new_line, text) pieces.

    A leading run of ``/``/``\\`` markers and every newline start a
    new line. Consecutive separators collapse into one line break and
    empty pieces are dropped.
    """
    starts_new_line = False
    stripped = text.lstrip('/\\')
    if stripped != text:
        # A run of leading line markers collapses into one line break.
        starts_new_line = True
        text = stripped

    pieces = []
    for part in text.split('\n'):
        if part:
            pieces.append((starts_new_line, part))
        # Any separator (or run of them) makes the next non-empty
        # piece start a new line.
        starts_new_line = True
    return pieces


class _TempoMap:
    """Seconds lookup for one sequence of (tick, tempo) changes."""

    def __init__(self, tempo_events):
        # tempo_events: list of (absolute tick, tempo), sorted by tick
        # with stable order for equal ticks.
        self._events = tempo_events

    def seconds_at(self, tick, ticks_per_beat):
        tempo = DEFAULT_TEMPO
        last_tick = 0
        seconds = 0.0
        for event_tick, event_tempo in self._events:
            if event_tick >= tick:
                break
            seconds += tick2second(event_tick - last_tick,
                                   ticks_per_beat, tempo)
            last_tick = event_tick
            tempo = event_tempo
        seconds += tick2second(tick - last_tick, ticks_per_beat, tempo)
        return seconds


def _check_time_base(midifile):
    ticks_per_beat = midifile.ticks_per_beat
    if not isinstance(ticks_per_beat, int) or ticks_per_beat <= 0:
        raise ValueError(
            f'cannot compute lyrics timeline for time division '
            f'{ticks_per_beat!r}: SMPTE or invalid time bases are not '
            f'supported')
    return ticks_per_beat


def _collect_events(midifile, include_text):
    """Collect (track, tick, order, type, text) lyric events.

    ``order`` is the position inside the track so that events at the
    same tick keep their original order. Tracks are iterated in file
    order, so a stable sort by tick yields (tick, track, order).
    """
    wanted = ('lyrics', 'text') if include_text else ('lyrics',)
    events = []
    for track_number, track in enumerate(midifile.tracks):
        tick = 0
        for order, msg in enumerate(track):
            tick += msg.time
            if msg.type in wanted:
                events.append((track_number, tick, order,
                               msg.type, msg.text))
    return events


def _collect_tempos(track):
    tempos = []
    tick = 0
    for msg in track:
        tick += msg.time
        if msg.type == 'set_tempo':
            if msg.tempo <= 0:
                raise ValueError(
                    f'invalid tempo {msg.tempo} (must be positive)')
            tempos.append((tick, msg.tempo))
    return tempos


def extract_lyrics(midifile, include_text=False):
    """Extract a lyrics timeline from a MIDI file.

    Returns a list of :class:`LyricFragment` sorted by absolute tick
    (stable for equal ticks: track number, then position in track).
    For type 2 files the fragments are grouped by track instead, since
    the patterns are not synchronized.

    Pass ``include_text=True`` to also include ``text`` meta messages.

    The file is only read, never modified. A file with no lyrics
    yields an empty list.
    """
    ticks_per_beat = _check_time_base(midifile)

    events = _collect_events(midifile, include_text)

    if midifile.type == 2:
        # Each pattern is independent: per-track tempo map, and the
        # output stays grouped by track (collection order already is
        # track-major with non-decreasing ticks inside each track).
        tempo_maps = [_TempoMap(_collect_tempos(track))
                      for track in midifile.tracks]
        ordered = events
    else:
        # One global tempo map from all tracks, stable-sorted by tick.
        tempo_events = []
        for track in midifile.tracks:
            tempo_events.extend(_collect_tempos(track))
        tempo_events.sort(key=lambda event: event[0])
        tempo_map = _TempoMap(tempo_events)
        tempo_maps = [tempo_map] * len(midifile.tracks)
        # Stable sort: equal ticks keep (track, order) file order.
        ordered = sorted(events, key=lambda event: event[1])

    fragments = []
    line = 0
    for track_number, tick, _order, msg_type, text in ordered:
        for starts_new_line, piece in _pieces_of(_clean_text(text)):
            if starts_new_line and fragments:
                line += 1
            fragments.append(LyricFragment(
                track=track_number,
                tick=tick,
                seconds=tempo_maps[track_number].seconds_at(
                    tick, ticks_per_beat),
                text=piece,
                type=msg_type,
                line=line,
            ))
    return fragments


def format_lyrics(fragments):
    """Format a lyrics timeline as a printable preview string."""
    return '\n'.join(str(fragment) for fragment in fragments)
