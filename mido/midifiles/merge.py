# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

"""
Merge two MIDI files by actual playback time.

This is different from merge_tracks(), which simply concatenates the
tracks of one file. merge_midifiles() takes two files that may have
different ticks_per_beat and different tempo maps, computes the
absolute playback time (in seconds) of every event in the secondary
file using that file's own tempo map, and places the events on the
primary file's timeline. The result is a new type 1 MidiFile that can
be saved with MidiFile.save().
"""

from bisect import bisect_right
from fractions import Fraction
from numbers import Integral

from .meta import MetaMessage
from .midifiles import DEFAULT_TEMPO, MidiFile
from .tracks import MidiTrack, fix_end_of_track

# Largest delta time that fits in a MIDI variable length quantity.
MAX_TICK = 0x0FFFFFFF

# Meta message types in the secondary file that would corrupt the
# primary file's timeline if they were copied. They are only used to
# compute playback times in the secondary file and are then dropped.
_TIMELINE_META_TYPES = frozenset(['set_tempo', 'time_signature'])


def _check_file(mid, role):
    if mid.type == 2:
        raise TypeError(
            f"can't merge type 2 (asynchronous) file ({role} file)")
    if not isinstance(mid.ticks_per_beat, Integral) \
            or mid.ticks_per_beat <= 0:
        raise ValueError(
            f'invalid ticks_per_beat {mid.ticks_per_beat!r} in {role} file'
            ' (must be a positive integer; SMPTE time division is not'
            ' supported)')
    for track in mid.tracks:
        for msg in track:
            if not isinstance(msg.time, Integral) or msg.time < 0:
                raise ValueError(
                    f'invalid message time {msg.time!r} in {role} file'
                    ' (delta times must be non-negative integers)')


def _tempo_segments(mid):
    """Return [(tick, tempo)] of tempo changes, sorted by tick.

    Always starts with (0, DEFAULT_TEMPO). If several set_tempo
    messages share a tick the last one (in file order) wins.
    """
    changes = []
    order = 0
    for track in mid.tracks:
        now = 0
        for msg in track:
            now += msg.time
            if msg.type == 'set_tempo':
                changes.append((now, order, msg.tempo))
                order += 1
    changes.sort(key=lambda change: (change[0], change[1]))

    segments = [(0, DEFAULT_TEMPO)]
    for tick, _, tempo in changes:
        if tick == segments[-1][0]:
            segments[-1] = (tick, tempo)
        else:
            segments.append((tick, tempo))
    return segments


def _seconds_map(mid):
    """Return [(start_tick, start_seconds, tempo)] for a file.

    Times are exact fractions so that rounding to ticks happens only
    once, at the very end.
    """
    ticks_per_beat = mid.ticks_per_beat
    segments = []
    last_tick = 0
    last_seconds = Fraction(0)
    last_tempo = DEFAULT_TEMPO
    for tick, tempo in _tempo_segments(mid):
        last_seconds += (Fraction(tick - last_tick) * last_tempo
                         / (ticks_per_beat * 1000000))
        last_tick = tick
        segments.append((tick, last_seconds, tempo))
        last_tempo = tempo
    return segments


def _track_seconds(mid, track, segments):
    """Yield the absolute playback time in seconds of each message."""
    ticks_per_beat = mid.ticks_per_beat
    now = 0
    segment = 0
    for msg in track:
        now += msg.time
        while segment + 1 < len(segments) \
                and segments[segment + 1][0] <= now:
            segment += 1
        start_tick, start_seconds, tempo = segments[segment]
        yield (start_seconds + Fraction(now - start_tick) * tempo
               / (ticks_per_beat * 1000000))


def _round_to_tick(value):
    # Round a non-negative Fraction to the nearest tick (half up).
    # The rounding error is at most half a tick.
    return int(value + Fraction(1, 2))


class _TickConverter:
    """Convert seconds to ticks on the primary file's timeline."""

    def __init__(self, mid):
        self._segments = _seconds_map(mid)
        self._starts = [segment[1] for segment in self._segments]
        self._ticks_per_beat = mid.ticks_per_beat

    def tick(self, seconds):
        index = bisect_right(self._starts, seconds) - 1
        start_tick, start_seconds, tempo = self._segments[index]
        return _round_to_tick(
            start_tick + (seconds - start_seconds) * self._ticks_per_beat
            * 1000000 / tempo)


def _used_channels(mid):
    used = set()
    for track in mid.tracks:
        for msg in track:
            if not msg.is_meta and hasattr(msg, 'channel'):
                used.add(msg.channel)
    return used


def _resolve_channel_map(secondary_channels, primary_channels, channel_map):
    if channel_map is None:
        channel_map = {}

    mapping = {}
    for source, target in channel_map.items():
        for value in (source, target):
            if not isinstance(value, Integral) or not 0 <= value <= 15:
                raise ValueError(
                    f'invalid channel {value!r} in channel_map'
                    ' (must be an integer in 0..15)')
        mapping[source] = target

    resolved = {channel: mapping.get(channel, channel)
                for channel in sorted(secondary_channels)}

    targets = list(resolved.values())
    if len(set(targets)) != len(targets):
        raise ValueError(
            'channel map conflict: several secondary channels map to the'
            ' same channel')

    conflicts = sorted(set(targets) & primary_channels)
    if conflicts:
        raise ValueError(
            'channel map conflict: secondary channels {} collide with'
            ' channels used by the primary file; pass a channel_map that'
            ' moves them to free channels'.format(
                [channel for channel, target in resolved.items()
                 if target in conflicts]))
    return resolved


def merge_midifiles(primary, secondary, channel_map=None):
    """Merge two MIDI files into one type 1 file by playback time.

    The primary file provides the timeline: the returned file has its
    ticks_per_beat, tempo and time signature, and its tracks are copied
    verbatim. Every event of the secondary file is converted to
    absolute seconds using the secondary file's own tempo map and then
    placed on the primary timeline, with a rounding error of at most
    one output tick. set_tempo and time_signature meta messages of the
    secondary file are used for the conversion but not copied.

    channel_map optionally remaps channels of the secondary file, e.g.
    {0: 4} moves its channel 0 events to channel 4. If a secondary
    channel would end up on a channel already used by the primary file
    (after mapping), ValueError is raised.

    Returns a new MidiFile of type 1. The input files are not modified.

    Raises TypeError for type 2 (asynchronous) files and ValueError for
    invalid ticks_per_beat, negative or non-integer delta times, event
    times outside the representable range and channel map conflicts.
    """
    _check_file(primary, 'primary')
    _check_file(secondary, 'secondary')

    resolved_channels = _resolve_channel_map(
        _used_channels(secondary), _used_channels(primary), channel_map)

    merged = MidiFile(type=1, ticks_per_beat=primary.ticks_per_beat)

    # The primary tracks are copied unchanged: they already live on
    # the output timeline.
    for track in primary.tracks:
        merged.tracks.append(MidiTrack(msg.copy() for msg in track))

    converter = _TickConverter(primary)
    secondary_segments = _seconds_map(secondary)

    for track in secondary.tracks:
        events = []
        for order, (msg, seconds) in enumerate(
                zip(track, _track_seconds(secondary, track,
                                          secondary_segments))):
            if msg.type in _TIMELINE_META_TYPES:
                continue
            tick = converter.tick(seconds)
            if tick > MAX_TICK:
                raise ValueError(
                    f'message time out of range: {msg!r} in the secondary'
                    f' file lands on tick {tick}, which does not fit in a'
                    ' MIDI file')
            overrides = {'time': tick}
            if not msg.is_meta and hasattr(msg, 'channel'):
                overrides['channel'] = resolved_channels[msg.channel]
            events.append((tick, order, msg.copy(skip_checks=True,
                                                 **overrides)))

        # Stable sort: events that land on the same tick keep their
        # original order. Sorting also guarantees non-negative deltas
        # after rounding.
        events.sort(key=lambda event: (event[0], event[1]))

        new_track = MidiTrack()
        last_tick = 0
        for tick, _, msg in events:
            new_track.append(msg.copy(skip_checks=True,
                                      time=tick - last_tick))
            last_tick = tick
        merged.tracks.append(MidiTrack(
            fix_end_of_track(new_track, skip_checks=True)))

    return merged
