# SPDX-FileCopyrightText: 2024
#
# SPDX-License-Identifier: MIT

"""
Fragmented (escape) SysEx message handling in standard MIDI files.

A single SysEx message can be split over several escape events in a
track: an ``F0`` event starts the message and one or more ``F7``
events continue it. Other MIDI events (for example notes) may appear
between the fragments without ending the message. This module merges
those fragments back into whole SysEx messages while keeping the
information needed to write the original fragmentation back out
byte-for-byte.

See the MIDI File Format specification on "escape sequences".
"""

from ..messages import Message

#: Largest payload accepted when merging fragments. Matches the read
#: limit used elsewhere in the MIDI file reader.
MAX_SYSEX_LENGTH = 1000000


class SysexError(Exception):
    """Base class for fragmented SysEx errors."""


class OrphanSysexError(SysexError):
    """An F7 fragment with no preceding F0 in the same track."""


class NestedSysexError(SysexError):
    """An F0 fragment (or embedded F0 byte) inside an open SysEx."""


class UnterminatedSysexError(SysexError):
    """An F0 fragment never reaches an F7 terminator."""


class SysexTooLongError(SysexError):
    """The merged payload exceeds the allowed length."""


class SysExFragment:
    """One raw SysEx escape event as stored in a track.

    Attributes:
        track_index: index of the track containing the fragment.
        index: position of the corresponding ``sysex`` message in the
            track's message list.
        delta: delta time in ticks as written before the status byte.
        tick: absolute tick at which the fragment starts.
        status: the escape status byte, ``0xf0`` (start) or ``0xf7``
            (continuation / escape).
        raw: raw payload bytes inside the length field, exactly as
            they appear in the file (including any leading ``F0`` or
            trailing ``F7``). Writable, so the fragmentation can be
            edited before saving.
        terminated: whether the fragment ends with ``F7``.
        message: the stripped ``sysex`` :class:`~mido.Message` living
            in the track.
    """

    __slots__ = (
        'track_index',
        'index',
        'delta',
        'tick',
        'status',
        'raw',
        'terminated',
        'message',
    )

    def __init__(self, track_index, index, delta, tick, status, raw,
                 terminated, message):
        self.track_index = track_index
        self.index = index
        self.delta = delta
        self.tick = tick
        self.status = status
        self.raw = raw
        self.terminated = terminated
        self.message = message

    @property
    def payload(self):
        """Payload bytes without the leading F0 / trailing F7 framing."""
        data = list(self.raw)
        if data and data[0] == 0xf0:
            data = data[1:]
        if data and data[-1] == 0xf7:
            data = data[:-1]
        return data

    def __repr__(self):
        return ('SysExFragment(track_index={}, index={}, tick={}, '
                'status=0x{:02x}, terminated={}, {} raw byte{})').format(
                    self.track_index,
                    self.index,
                    self.tick,
                    self.status,
                    self.terminated,
                    len(self.raw),
                    '' if len(self.raw) == 1 else 's',
                )


class SysExMessage:
    """A complete SysEx message reconstructed from one or more fragments.

    Attributes:
        fragments: list of :class:`SysExFragment` objects, in file
            order. All fragments belong to the same track.
        data: merged payload bytes (without the F0/F7 framing bytes).
        start_tick: absolute tick of the first fragment.
        end_tick: absolute tick of the last fragment.
        terminated: whether the last fragment ends with F7.
        contains_f0: whether an embedded F0 byte was found in any
            fragment's raw data (a nested start).
    """

    __slots__ = ('fragments', 'data', 'start_tick', 'end_tick',
                 'terminated', 'contains_f0')

    def __init__(self, fragments, data, terminated, contains_f0):
        self.fragments = fragments
        self.data = data
        self.start_tick = fragments[0].tick
        self.end_tick = fragments[-1].tick
        self.terminated = terminated
        self.contains_f0 = contains_f0

    @property
    def track_index(self):
        return self.fragments[0].track_index

    def as_message(self):
        """Return the merged message as a regular mido ``sysex`` Message."""
        return Message('sysex', data=self.data, time=self.start_tick)

    def __repr__(self):
        return ('SysExMessage(track_index={}, fragments={}, '
                'start_tick={}, end_tick={}, terminated={}, '
                '{} payload byte{})').format(
                    self.track_index,
                    len(self.fragments),
                    self.start_tick,
                    self.end_tick,
                    self.terminated,
                    len(self.data),
                    '' if len(self.data) == 1 else 's',
                )


def get_sysex_fragments(track, track_index=0):
    """Return the :class:`SysExFragment` list recorded for a track.

    Tracks loaded from a MIDI file carry their fragmentation
    metadata. For tracks built in memory (or edited in a way that
    invalidated the metadata) the fragments are synthesised from
    their ``sysex`` messages so the same API works everywhere.
    """
    fragments = getattr(track, 'sysex_fragments', None)
    sysex_count = sum(1 for msg in track if msg.type == 'sysex')
    if fragments is not None and len(fragments) == sysex_count:
        return fragments

    # Synthesise standard, single-event fragments.
    fragments = []
    tick = 0
    for index, msg in enumerate(track):
        tick += msg.time
        if msg.type == 'sysex':
            raw = [0xf0] + list(msg.data) + [0xf7]
            fragments.append(SysExFragment(
                track_index=track_index,
                index=index,
                delta=msg.time,
                tick=tick,
                status=0xf0,
                raw=raw,
                terminated=True,
                message=msg,
            ))
    return fragments


def group_sysex_fragments(track, track_index=0, max_length=MAX_SYSEX_LENGTH,
                          allow_orphan=False, allow_nested=False,
                          allow_unterminated=False):
    """Group a track's fragments into whole :class:`SysExMessage` objects.

    Merging never spans tracks: every group is contained in the track
    passed in. Other MIDI events are simply skipped here and stay
    untouched in the track, so their order is preserved.

    By default malformed input raises the matching subclass of
    :class:`SysexError`. Pass the corresponding ``allow_*`` flag to
    get a result object describing the condition instead.
    """
    messages = []
    open_fragments = None
    open_data = []
    contains_f0 = False

    def close(terminated):
        messages.append(SysExMessage(
            list(open_fragments), list(open_data), terminated,
            contains_f0))

    for fragment in get_sysex_fragments(track, track_index):
        payload = fragment.payload
        # Any F0 inside the escape event other than the framing F0 of
        # an F0 start fragment is an embedded (nested) start.
        embedded_f0 = (
            0xf0 in fragment.raw[1:]
            if fragment.status == 0xf0
            else 0xf0 in fragment.raw
        )

        if fragment.status == 0xf7 and open_fragments is None:
            # Orphan continuation.
            if not allow_orphan:
                raise OrphanSysexError(
                    'F7 SysEx fragment at tick {} has no preceding F0 '
                    'in track {}'.format(fragment.tick, track_index))
            messages.append(SysExMessage(
                [fragment], payload, fragment.terminated, embedded_f0))
            continue

        if fragment.status == 0xf0 and open_fragments is not None:
            # Nested start: the previous message is left without a
            # terminator.
            if not allow_nested:
                raise NestedSysexError(
                    'F0 SysEx fragment at tick {} starts while the '
                    'SysEx started at tick {} is still open in '
                    'track {}'.format(fragment.tick,
                                      open_fragments[0].tick,
                                      track_index))
            close(terminated=False)
            open_fragments = None
            open_data = []
            contains_f0 = False

        if open_fragments is None:
            open_fragments = [fragment]
        else:
            open_fragments.append(fragment)
        open_data.extend(payload)
        contains_f0 = contains_f0 or embedded_f0

        if len(open_data) > max_length:
            raise SysexTooLongError(
                'SysEx payload starting at tick {} is {} bytes, '
                'exceeds maximum of {}'.format(
                    open_fragments[0].tick, len(open_data), max_length))

        if embedded_f0 and not allow_nested:
            raise NestedSysexError(
                'embedded F0 byte inside SysEx fragment at tick {} '
                'in track {}'.format(fragment.tick, track_index))

        if fragment.terminated:
            close(terminated=True)
            open_fragments = None
            open_data = []
            contains_f0 = False

    if open_fragments is not None:
        if not allow_unterminated:
            raise UnterminatedSysexError(
                'SysEx starting at tick {} in track {} has no F7 '
                'terminator before the end of the track'.format(
                    open_fragments[0].tick, track_index))
        close(terminated=False)

    return messages


def iter_sysex_messages(track, track_index=0, **kwargs):
    """Yield complete :class:`SysExMessage` objects from one track.

    ``F0`` start fragments and their ``F7`` continuation fragments are
    merged; other MIDI events between them do not affect the grouping.
    Keyword arguments are forwarded to
    :func:`group_sysex_fragments`.
    """
    for message in group_sysex_fragments(track, track_index, **kwargs):
        yield message


def iter_track_events(track, track_index=0, **kwargs):
    """Yield a track's events in order, with SysEx fragments merged.

    Every non-SysEx message is yielded as-is, preserving the original
    ordering and delta times. Each merged :class:`SysExMessage` is
    yielded in the position of its last fragment (the point where the
    whole message is complete), with ``time`` set to the absolute
    start tick of the message.
    """
    sysex_messages = group_sysex_fragments(
        track, track_index, **kwargs)
    emitted_on = {
        id(message.fragments[-1]): message
        for message in sysex_messages
    }
    fragments = get_sysex_fragments(track, track_index)

    for msg in track:
        fragment = next((f for f in fragments if f.message is msg), None)
        if fragment is None:
            yield msg
        elif id(fragment) in emitted_on:
            yield emitted_on[id(fragment)]


def iter_file_sysex(midifile, **kwargs):
    """Yield all complete :class:`SysExMessage` objects of a MIDI file.

    Each yielded value carries its ``track_index``. Fragment groups
    never cross track boundaries: an ``F7`` continuation in another
    track is reported as an orphan there.
    """
    for track_index, track in enumerate(midifile.tracks):
        for message in iter_sysex_messages(
                track, track_index=track_index, **kwargs):
            yield message
