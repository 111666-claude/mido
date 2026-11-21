# SPDX-FileCopyrightText: 2026 Mido contributors
#
# SPDX-License-Identifier: MIT

"""
Convert between Standard MIDI File formats.

Type 0 files store every message in one single track while type 1 files
split the data into several synchronous tracks. The functions in this
module convert between the two without mutating the source file.

Type 2 (asynchronous) files can not be converted, because their tracks
do not share a common time line.
"""

from numbers import Integral

from .tracks import MidiTrack, merge_tracks


def _check_times(mid):
    """Validate delta times before merging or splitting tracks."""
    for track in mid.tracks:
        for msg in track:
            if not isinstance(msg.time, Integral):
                raise ValueError(
                    'message time must be int in MIDI file')
            if msg.time < 0:
                raise ValueError(
                    'message time must be non-negative in MIDI file')


def _unique_name(name, used):
    """Return a track name that does not collide with the ones in ``used``.

    The set ``used`` is updated in place.
    """
    if name not in used:
        used.add(name)
        return name

    suffix = 2
    while f'{name} ({suffix})' in used:
        suffix += 1
    unique = f'{name} ({suffix})'
    used.add(unique)
    return unique


def _to_abstime_events(tracks):
    """Yield ``(absolute_tick, message)`` pairs in playback order.

    Messages are copied, so the returned events can be rearranged into
    new tracks without touching the source objects. Stable ordering keeps
    same-tick events in track order, matching ``merge_tracks()``.
    """
    events = []
    for track in tracks:
        now = 0
        for msg in track:
            now += msg.time
            events.append((now, msg.copy(skip_checks=True)))
    events.sort(key=lambda event: event[0])
    return events


def to_type0(mid):
    """Return a type 0 (single track) copy of the MIDI file.

    All tracks are merged into one track while global timing and the
    order of events occurring on the same tick are preserved.

    Raises:
        TypeError: if ``mid`` is a type 2 file.
        ValueError: if a message has a non-integer or negative time.
    """
    if mid.type == 2:
        raise TypeError(
            "can't convert type 2 (asynchronous) file to type 0")

    _check_times(mid)

    merged = merge_tracks(mid.tracks, skip_checks=True)

    return mid.__class__(
        type=0,
        ticks_per_beat=mid.ticks_per_beat,
        charset=mid.charset,
        debug=mid.debug,
        clip=mid.clip,
        tracks=[merged],
    )


def to_type1(mid):
    """Return a type 1 (multi track) copy of the MIDI file.

    Channel messages are split into one track per MIDI channel.

    A dedicated first track holds the global meta messages (tempo, time
    signature, key signature and friends) as well as everything that has
    no channel: SysEx dumps, unknown meta messages and other non-channel
    data. Existing ``end_of_track`` messages are removed and a single one
    is written at the end of every track on save, exactly like a freshly
    created track.

    No messages are added or rewritten: when two source tracks carry the
    same ``track_name`` meta the duplicate is renamed in place with a
    numeric suffix (for example ``Piano (2)``), so the split stays
    deterministic and reversible.

    Raises:
        TypeError: if ``mid`` is a type 2 file.
        ValueError: if a message has a non-integer or negative time.
    """
    if mid.type == 2:
        raise TypeError(
            "can't convert type 2 (asynchronous) file to type 1")

    _check_times(mid)

    events = _to_abstime_events(mid.tracks)

    # The first track is dedicated to global and channel-less events.
    conductor = MidiTrack()
    channel_tracks = {channel: MidiTrack() for channel in range(16)}
    targets = [conductor] + list(channel_tracks.values())
    last_tick = [0] * len(targets)
    used_names = set()

    for tick, msg in events:
        if msg.type == 'end_of_track':
            # Each new track gets a fresh end_of_track on save.
            continue

        if msg.is_meta or not hasattr(msg, 'channel'):
            index = 0
        else:
            index = msg.channel + 1

        if msg.type == 'track_name':
            if msg.name in used_names:
                # Rename the conflicting meta in place instead of
                # dropping it or adding a new message.
                msg.name = _unique_name(msg.name, used_names)
            else:
                used_names.add(msg.name)

        target = targets[index]
        target.append(msg.copy(skip_checks=True, time=tick - last_tick[index]))
        last_tick[index] = tick

    # Drop empty tracks, including the conductor when there is nothing
    # global to store; a type 1 file with no events simply has no tracks.
    tracks = [track for track in targets if track]

    return mid.__class__(
        type=1,
        ticks_per_beat=mid.ticks_per_beat,
        charset=mid.charset,
        debug=mid.debug,
        clip=mid.clip,
        tracks=tracks,
    )
