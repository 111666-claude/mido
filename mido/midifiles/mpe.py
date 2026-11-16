# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

"""
MPE (MIDI Polyphonic Expression) support for MIDI files.

An MPE zone consists of one master channel and a set of member
channels. Each note is played on its own member channel, so per-note
expression (pitch bend and channel pressure) is sent as ordinary
channel messages on that note's channel. This module extracts those
per-note expression trajectories from a MidiFile (with absolute
ticks) and writes edited trajectories back while preserving
everything else (tracks, CC and meta messages, same-tick order).

References:

https://www.midi.org/specifications-old/item/mpe-midi-polyphonic-expression
"""

from ..messages import MAX_PITCHWHEEL, MIN_PITCHWHEEL, Message

# Message types that carry per-note MPE data on member channels.
_MANAGED_TYPES = frozenset(['note_on', 'note_off', 'pitchwheel', 'aftertouch'])

# Sort order for generated messages that share a tick: end old notes
# first (so a reused channel is free), then expression, then new notes.
_GENERATED_ORDER = {'note_off': 0, 'pitchwheel': 1, 'aftertouch': 1, 'note_on': 2}


class MPEZone:
    """An MPE zone: one master channel and a set of member channels.

    member_channels is an iterable of channel numbers (0..15) used
    for per-note messages. The master channel carries zone-wide
    messages and is never used for notes.
    """

    def __init__(self, master_channel, member_channels):
        member_channels = tuple(member_channels)
        for channel in (master_channel,) + member_channels:
            if not 0 <= channel <= 15:
                raise ValueError(f'channel must be in range 0..15, got {channel}')
        if not member_channels:
            raise ValueError('an MPE zone needs at least one member channel')
        if master_channel in member_channels:
            raise ValueError(
                f'master channel {master_channel} is also a member channel')
        self.master_channel = master_channel
        self.member_channels = member_channels

    @classmethod
    def lower(cls, num_members=15):
        """Lower zone: master channel 0, member channels 1..num_members."""
        return cls(0, range(1, num_members + 1))

    @classmethod
    def upper(cls, num_members=15):
        """Upper zone: master channel 15, member channels 15-num_members..14."""
        return cls(15, range(15 - num_members, 15))

    def __repr__(self):
        return (f'{self.__class__.__name__}(master_channel={self.master_channel}, '
                f'member_channels={self.member_channels!r})')


class MPENote:
    """One note in an MPE zone with its expression trajectories.

    Attributes:

    note, velocity -- note number and note-on velocity.
    channel -- member channel the note plays (played) on. Updated by
        write_mpe_notes() if the channel had to be reallocated.
    track -- index of the MidiFile track the note belongs to.
    start, end -- absolute ticks of note-on and note-off.
    pitch, pressure -- expression trajectories: lists of
        (absolute_tick, value) tuples in time order. Values are
        pitchwheel (-8192..8191) and aftertouch (0..127) values.
        Events received before the note-on (a common MPE idiom) are
        included with their original ticks.

    All attributes may be edited before writing the notes back with
    write_mpe_notes().
    """

    def __init__(self, note, velocity=64, channel=None, track=0,
                 start=0, end=0, pitch=None, pressure=None):
        self.note = note
        self.velocity = velocity
        self.channel = channel
        self.track = track
        self.start = start
        self.end = end
        self.pitch = list(pitch) if pitch is not None else []
        self.pressure = list(pressure) if pressure is not None else []

    @property
    def duration(self):
        return self.end - self.start

    def __eq__(self, other):
        if not isinstance(other, MPENote):
            return NotImplemented
        return vars(self) == vars(other)

    def __repr__(self):
        return (f'{self.__class__.__name__}(note={self.note}, '
                f'velocity={self.velocity}, channel={self.channel}, '
                f'track={self.track}, start={self.start}, end={self.end}, '
                f'pitch={self.pitch!r}, pressure={self.pressure!r})')


def _validate_zones(zones):
    zones = list(zones)
    if not zones:
        raise ValueError('no MPE zones configured')

    used = {}
    for zone in zones:
        for channel in (zone.master_channel,) + zone.member_channels:
            if channel in used:
                raise ValueError(
                    f'channel {channel} is used by more than one MPE zone')
            used[channel] = zone
    return zones


def _member_channel_map(zones):
    """Map each member channel to its zone."""
    return {channel: zone for zone in zones for channel in zone.member_channels}


def extract_mpe_notes(midifile, zones, on_conflict='close', on_orphan='ignore'):
    """Extract per-note MPE expression from a MidiFile.

    zones is a list of MPEZone objects describing the master/member
    channel layout. note_on/note_off, pitchwheel and aftertouch
    (channel pressure) messages on member channels are collected into
    MPENote objects with absolute ticks. Messages on master channels
    and on channels outside all zones are left alone.

    Returns a list of MPENote in the order the notes started.

    on_conflict selects what happens when a note-on arrives on a
    member channel that already has a sounding note (a conflict,
    since MPE allows only one note per member channel):

        'close' (default) -- end the sounding note at this tick and
            start the new one.
        'error' -- raise ValueError.

    on_orphan selects what happens when a note-off arrives on a
    member channel with no sounding note:

        'ignore' (default) -- the note-off is dropped.
        'error' -- raise ValueError.

    Notes still sounding at the end of a track are ended at the
    track's final tick.
    """
    if on_conflict not in ('close', 'error'):
        raise ValueError(f'invalid on_conflict policy: {on_conflict!r}')
    if on_orphan not in ('ignore', 'error'):
        raise ValueError(f'invalid on_orphan policy: {on_orphan!r}')

    zones = _validate_zones(zones)
    member = _member_channel_map(zones)

    notes = []

    for track_index, track in enumerate(midifile.tracks):
        tick = 0
        active = {}    # channel -> MPENote currently sounding
        pending = {}   # channel -> expression events before its note-on

        for msg in track:
            tick += msg.time

            if msg.is_meta or msg.type not in _MANAGED_TYPES:
                continue
            if msg.channel not in member:
                continue

            channel = msg.channel

            if msg.type == 'note_on' and msg.velocity > 0:
                if channel in active:
                    if on_conflict == 'error':
                        raise ValueError(
                            f'conflicting note-on for note {msg.note} on '
                            f'member channel {channel} at tick {tick}: '
                            f'note {active[channel].note} is still sounding')
                    # 'close': end the sounding note at this tick.
                    finished = active.pop(channel)
                    finished.end = tick
                    notes.append(finished)

                note = MPENote(note=msg.note, velocity=msg.velocity,
                               channel=channel, track=track_index,
                               start=tick, end=tick)
                # Attach expression received before the note-on.
                for kind, event_tick, value in pending.pop(channel, []):
                    getattr(note, kind).append((event_tick, value))
                active[channel] = note

            elif msg.type == 'note_off' or msg.type == 'note_on':
                # note_off, or note_on with velocity 0.
                if channel in active:
                    finished = active.pop(channel)
                    finished.end = tick
                    notes.append(finished)
                elif on_orphan == 'error':
                    raise ValueError(
                        f'orphan note-off for note {msg.note} on member '
                        f'channel {channel} at tick {tick}: no note is '
                        f'sounding')
                # 'ignore': drop the orphan note-off.

            else:
                # pitchwheel or aftertouch.
                kind = 'pitch' if msg.type == 'pitchwheel' else 'pressure'
                value = msg.pitch if msg.type == 'pitchwheel' else msg.value
                if channel in active:
                    getattr(active[channel], kind).append((tick, value))
                else:
                    pending.setdefault(channel, []).append((kind, tick, value))

        # Close notes still sounding at the end of the track.
        for note in active.values():
            note.end = tick
            notes.append(note)

    notes.sort(key=lambda note: (note.start, note.track, note.channel))
    return notes


def _clip_value(value, low, high, name, strict):
    if not low <= value <= high:
        if strict:
            raise ValueError(f'{name} value {value} out of range {low}..{high}')
        return min(max(value, low), high)
    return value


def _assign_channels(notes, member_channels):
    """Assign a free member channel to each note (in place).

    A note keeps its current channel if the channel is free for the
    note's whole duration; otherwise the first free member channel is
    used. A channel is free again at the tick the previous note ends,
    so channels are reused without notes overlapping (and thus without
    their expression messages leaking into each other).
    """
    member_channels = tuple(member_channels)
    free_at = {channel: -1 for channel in member_channels}

    for note in sorted(notes, key=lambda n: (n.start, n.end, n.note)):
        if note.end < note.start:
            raise ValueError(
                f'note {note.note} ends at tick {note.end} '
                f'before it starts (tick {note.start})')

        candidates = [note.channel] if note.channel in free_at else []
        candidates += [ch for ch in member_channels if ch != note.channel]

        for channel in candidates:
            if free_at[channel] <= note.start:
                note.channel = channel
                free_at[channel] = note.end
                break
        else:
            raise ValueError(
                f'no free member channel for note {note.note} at tick '
                f'{note.start}: all {len(member_channels)} member channels '
                f'are in use')


def write_mpe_notes(midifile, zones, notes, strict=False):
    """Write (possibly edited) MPE notes back into a MidiFile.

    All note_on/note_off, pitchwheel and aftertouch messages on member
    channels of the given zones are removed and regenerated from the
    MPENote objects. Everything else is preserved: the track
    structure, messages on master and non-zone channels, CC and meta
    messages, and the relative order of messages that share a tick.
    Kept messages at a tick come before regenerated ones; among
    regenerated messages note-offs come first, then expression events,
    then note-ons, so a member channel reused at the same tick is
    released before it is reused.

    Each note is written to the track its track attribute points to
    and to the member channel stored in its channel attribute (which
    is updated in place if reallocation was needed).

    Expression values outside their valid MIDI range (pitch
    -8192..8191, pressure 0..127, note/velocity 0..127) are clipped to
    the nearest valid value. Pass strict=True to raise ValueError
    instead.

    The MidiFile is modified in place.
    """
    zones = _validate_zones(zones)
    member = _member_channel_map(zones)
    member_channels = set(member)
    notes = list(notes)

    _assign_channels(notes, member_channels)

    # Generate replacement messages, grouped by target track.
    generated = {}  # track index -> list of (tick, msg)
    for note in notes:
        track_index = note.track or 0
        if not 0 <= track_index < len(midifile.tracks):
            raise ValueError(
                f'note {note.note} belongs to track {track_index} but the '
                f'file has {len(midifile.tracks)} track(s)')

        note_number = _clip_value(note.note, 0, 127, 'note', strict)
        velocity = _clip_value(note.velocity, 0, 127, 'velocity', strict)

        events = generated.setdefault(track_index, [])
        events.append((note.start, Message(
            'note_on', channel=note.channel, note=note_number,
            velocity=velocity)))
        for tick, value in note.pitch:
            events.append((tick, Message(
                'pitchwheel', channel=note.channel,
                pitch=_clip_value(value, MIN_PITCHWHEEL, MAX_PITCHWHEEL,
                                  'pitch', strict))))
        for tick, value in note.pressure:
            events.append((tick, Message(
                'aftertouch', channel=note.channel,
                value=_clip_value(value, 0, 127, 'pressure', strict))))
        events.append((note.end, Message(
            'note_off', channel=note.channel, note=note_number, velocity=0)))

    for track_index, track in enumerate(midifile.tracks):
        # Keep everything that is not MPE-managed, with absolute ticks.
        tick = 0
        kept = []
        for index, msg in enumerate(track):
            tick += msg.time
            if (not msg.is_meta and msg.type in _MANAGED_TYPES
                    and msg.channel in member_channels):
                continue
            kept.append((tick, (0, index), msg))

        events = kept + [
            (tick, (1, _GENERATED_ORDER[msg.type], seq), msg)
            for seq, (tick, msg) in enumerate(generated.get(track_index, []))
        ]
        events.sort(key=lambda event: (event[0], event[1]))

        # Convert back to delta times.
        track[:] = _to_delta_times(events)

    del midifile.merged_track  # uncache merged track
    return midifile


def _to_delta_times(events):
    messages = []
    previous_tick = 0
    for tick, _, msg in events:
        msg.time = tick - previous_tick
        previous_tick = tick
        messages.append(msg)
    return messages
