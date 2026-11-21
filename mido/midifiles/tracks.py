# SPDX-FileCopyrightText: 2016 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

from .meta import MetaMessage


class MidiTrack(list):
    def __new__(cls, iterable=()):
        # list subclasses are also created through __new__ (e.g. in
        # copy(), slicing and __add__) with __init__ bypassed, so the
        # version counter is set up here. Items themselves are filled
        # in by __init__ (or by list directly on the bypassed paths).
        self = super().__new__(cls)
        self._version = 0
        return self

    def __init__(self, iterable=()):
        super().__init__(iterable)

    def _bump_version(self):
        self._version += 1

    @property
    def name(self):
        """Name of the track.

        This will return the name from the first track_name meta
        message in the track, or '' if there is no such message.

        Setting this property will update the name field of the first
        track_name message in the track. If no such message is found,
        one will be added to the beginning of the track with a delta
        time of 0."""
        for message in self:
            if message.type == 'track_name':
                return message.name
        else:
            return ''

    @name.setter
    def name(self, name):
        # Find the first track_name message and modify it.
        for message in self:
            if message.type == 'track_name':
                message.name = name
                return
        else:
            # No track name found, add one.
            self.insert(0, MetaMessage('track_name', name=name, time=0))

    def copy(self):
        return self.__class__(self)

    def __getitem__(self, index_or_slice):
        # Retrieve item from the MidiTrack
        lst = list.__getitem__(self, index_or_slice)
        if isinstance(index_or_slice, int):
            # If an index was provided, return the list element
            return lst
        else:
            # Otherwise, construct a MidiTrack to return.
            # TODO: this make a copy of the list. Is there a better way?
            return self.__class__(lst)

    def __add__(self, other):
        return self.__class__(list.__add__(self, other))

    def __mul__(self, other):
        return self.__class__(list.__mul__(self, other))

    def __repr__(self):
        if len(self) == 0:
            messages = ''
        elif len(self) == 1:
            messages = f'[{self[0]}]'
        else:
            messages = '[\n  {}]'.format(',\n  '.join(repr(m) for m in self))
        return f'{self.__class__.__name__}({messages})'

    # Every list operation that can change the contents bumps the
    # version so cached merged tracks are invalidated.
    def append(self, message):
        super().append(message)
        self._bump_version()

    def extend(self, iterable):
        super().extend(iterable)
        self._bump_version()

    def insert(self, index, message):
        super().insert(index, message)
        self._bump_version()

    def remove(self, message):
        super().remove(message)
        self._bump_version()

    def pop(self, index=-1):
        message = super().pop(index)
        self._bump_version()
        return message

    def clear(self):
        super().clear()
        self._bump_version()

    def sort(self, *args, **kwargs):
        super().sort(*args, **kwargs)
        self._bump_version()

    def reverse(self):
        super().reverse()
        self._bump_version()

    def __setitem__(self, index_or_slice, value):
        super().__setitem__(index_or_slice, value)
        self._bump_version()

    def __delitem__(self, index_or_slice):
        super().__delitem__(index_or_slice)
        self._bump_version()

    def __iadd__(self, iterable):
        result = super().__iadd__(iterable)
        self._bump_version()
        return result

    def __imul__(self, count):
        result = super().__imul__(count)
        self._bump_version()
        return result


def _to_abstime(messages):
    """Convert messages to absolute time."""
    now = 0
    for msg in messages:
        now += msg.time
        yield msg.copy(time=now)


def _to_reltime(messages):
    """Convert messages to relative time."""
    now = 0
    for msg in messages:
        delta = msg.time - now
        yield msg.copy(time=delta)
        now = msg.time


def fix_end_of_track(messages):
    """Remove all end_of_track messages and add one at the end.

    This is used by merge_tracks() and MidiFile.save()."""
    # Accumulated delta time from removed end of track messages.
    # This is added to the next message.
    accum = 0

    for msg in messages:
        if msg.type == 'end_of_track':
            accum += msg.time
        else:
            if accum:
                delta = accum + msg.time
                yield msg.copy(time=delta)
                accum = 0
            else:
                yield msg

    yield MetaMessage('end_of_track', time=accum)


def merge_tracks(tracks):
    """Returns a MidiTrack object with all messages from all tracks.

    The messages are returned in playback order with delta times
    as if they were all in one track.
    """
    messages = []
    for track in tracks:
        messages.extend(_to_abstime(track))

    messages.sort(key=lambda msg: msg.time)

    return MidiTrack(fix_end_of_track(_to_reltime(messages)))
