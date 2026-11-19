# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Generate a metronome (click) track from a MIDI file.

The click track is derived from the ``time_signature`` meta messages
of a type 0 or type 1 file. Clicks are placed on the tick grid, so
``set_tempo`` changes automatically stretch them in time during
playback; the file itself is never modified.
"""

from fractions import Fraction
from numbers import Integral

from ..messages import Message
from .meta import MetaMessage
from .tracks import MidiTrack

# Files without a time_signature meta message are in 4/4.
DEFAULT_TIME_SIGNATURE = (4, 4)


def _check_int(name, value, low, high):
    if (not isinstance(value, Integral) or isinstance(value, bool)
            or not low <= value <= high):
        raise ValueError(f'{name} must be an integer in range {low}..{high}')
    return value


def _to_tick(pos):
    # pos is a Fraction. Positions are exact for every power-of-two
    # denominator that divides 4 * ticks_per_beat; anything more exotic
    # is rounded half up to the nearest tick (deterministic).
    if pos.denominator == 1:
        return pos.numerator
    return int(pos + Fraction(1, 2))


def _time_signature_events(mid):
    """Absolute ticks and values of all time_signature changes.

    Events are returned in merged-track order. When several tracks
    carry conflicting time_signature events at the same tick, the one
    from the later track wins (merge_tracks() sorts stably, so events
    at one tick keep track order and the last one applied is the one
    that counts).
    """
    events = []
    now = 0
    for msg in mid.merged_track:
        now += msg.time
        if msg.type == 'time_signature':
            events.append((now, msg.numerator, msg.denominator))
    return events


def _click_ticks(mid, start, end):
    """Yield (tick, is_accent) for every click in [start, end)."""
    tpb = mid.ticks_per_beat
    anchor = 0
    numerator, denominator = DEFAULT_TIME_SIGNATURE

    # A sentinel at `end` closes the last segment.
    changes = _time_signature_events(mid) + [(end, None, None)]

    for change_tick, num, den in changes:
        # Emit clicks for the current signature in [anchor, change_tick).
        # A time_signature event starts a new bar, so the click landing
        # exactly on the event tick already uses the new signature.
        beat = Fraction(tpb * 4, denominator)
        limit = min(change_tick, end)
        pos = Fraction(anchor)
        k = 0
        while pos < limit:
            if pos >= start:
                yield _to_tick(pos), k % numerator == 0
            pos += beat
            k += 1

        if num is None:
            break
        anchor = change_tick
        numerator, denominator = num, den


def make_click_track(mid, start=0, end=None, *,
                     channel=9,
                     accent_note=34, accent_velocity=96,
                     accent_duration=None,
                     note=33, velocity=64, duration=None,
                     name='click'):
    """Create a metronome click track for a MIDI file.

    Returns a new MidiTrack with one note per beat, derived from the
    file's time_signature meta messages. The file and its tracks are
    not modified; append the returned track yourself:

        mid.tracks.append(make_click_track(mid))

    The first beat of every bar gets the accent (``accent_note``,
    ``accent_velocity``, ``accent_duration``), all other beats get
    ``note``, ``velocity`` and ``duration``. Durations are in ticks
    and default to ticks_per_beat // 8.

    ``start`` and ``end`` are absolute ticks. Clicks are generated for
    the half-open interval [start, end); ``end=None`` means the end of
    the longest track. start == end yields a track with no notes.

    Behavior notes:

    * Bars are anchored at tick 0 and at every time_signature change.
      If the piece starts with a pickup (anacrusis), the clicks before
      the first barline are weak beats; a time_signature event always
      starts a new (accented) bar.
    * A time_signature event takes effect before a click on the same
      tick: the click uses the new signature and is accented.
    * Compound meters (6/8, 12/8, ...) click once per denominator
      unit, e.g. six eighth-note clicks per 6/8 bar.
    * Conflicting time_signature events at the same tick in different
      tracks are resolved deterministically: the event from the later
      track wins.
    * Type 2 (asynchronous) files have no shared timeline and raise
      TypeError.
    """
    if mid.type == 2:
        raise TypeError(
            "can't generate click track for type 2 (asynchronous) file")

    if isinstance(start, bool) or not isinstance(start, Integral) or start < 0:
        raise ValueError('start must be a non-negative integer tick')

    if end is None:
        end = max((sum(msg.time for msg in track) for track in mid.tracks),
                  default=0)
    elif isinstance(end, bool) or not isinstance(end, Integral) or end < 0:
        raise ValueError('end must be a non-negative integer tick')

    if end < start:
        raise ValueError('end must not be before start')

    _check_int('channel', channel, 0, 15)
    _check_int('accent_note', accent_note, 0, 127)
    _check_int('note', note, 0, 127)
    _check_int('accent_velocity', accent_velocity, 1, 127)
    _check_int('velocity', velocity, 1, 127)

    default_duration = max(1, mid.ticks_per_beat // 8)
    if accent_duration is None:
        accent_duration = default_duration
    if duration is None:
        duration = default_duration
    _check_int('accent_duration', accent_duration, 1, 2 ** 31 - 1)
    _check_int('duration', duration, 1, 2 ** 31 - 1)

    events = []
    for tick, is_accent in _click_ticks(mid, start, end):
        if is_accent:
            note_number, vel, dur = accent_note, accent_velocity, accent_duration
        else:
            note_number, vel, dur = note, velocity, duration
        events.append((tick, 1, Message('note_on', channel=channel,
                                        note=note_number, velocity=vel)))
        events.append((tick + dur, 0, Message('note_off', channel=channel,
                                              note=note_number, velocity=0)))

    # note_off (order 0) sorts before note_on (order 1) at the same
    # tick so overlapping durations never leave a stuck note.
    events.sort(key=lambda event: (event[0], event[1]))

    track = MidiTrack()
    if name is not None:
        track.append(MetaMessage('track_name', name=name, time=0))
    now = 0
    for tick, _, msg in events:
        msg.time = tick - now
        now = tick
        track.append(msg)
    track.append(MetaMessage('end_of_track', time=0))
    return track
