# SPDX-FileCopyrightText: 2017 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

import itertools
from mido.messages import Message
from mido.midifiles.meta import MetaMessage
from mido.midifiles.tracks import MidiTrack

zip = getattr(itertools, 'izip', zip)


def test_track_slice():
    track = MidiTrack()

    # Slice should return MidiTrack object.
    assert isinstance(track[::], MidiTrack)


def test_track_name():
    name1 = MetaMessage('track_name', name='name1')
    name2 = MetaMessage('track_name', name='name2')

    # The track should use the first name it finds.
    track = MidiTrack([name1, name2])
    assert track.name == name1.name


def test_track_repr():
    track = MidiTrack([
        Message('note_on', channel=1, note=2, time=3),
        Message('note_off', channel=1, note=2, time=3),
    ])
    track_eval = eval(repr(track))
    for m1, m2 in zip(track, track_eval):
        assert m1 == m2


def test_track_mutations_bump_version():
    # Every list operation that changes the contents must bump the
    # version used to invalidate cached merged tracks.
    track = MidiTrack([Message('note_on', time=0)])
    version = track._version

    track.append(Message('note_off', time=1))
    assert track._version > version
    version = track._version

    track.extend([Message('control_change', control=1, time=1)])
    assert track._version > version
    version = track._version

    track.insert(0, Message('pitchwheel', time=0))
    assert track._version > version
    version = track._version

    track[0] = Message('pitchwheel', pitch=1, time=0)
    assert track._version > version
    version = track._version

    track[0:1] = [Message('pitchwheel', pitch=2, time=0)]
    assert track._version > version
    version = track._version

    track.remove(track[0])
    assert track._version > version
    version = track._version

    del track[0]
    assert track._version > version
    version = track._version

    track.append(Message('note_on', time=0))
    del track[0:1]
    assert track._version > version
    version = track._version

    track.append(Message('note_on', time=0))
    track.pop()
    assert track._version > version
    version = track._version

    track += [Message('note_on', time=0)]
    assert track._version > version
    version = track._version

    track *= 2
    assert track._version > version
    version = track._version

    track.sort(key=lambda msg: msg.type)
    assert track._version > version
    version = track._version

    track.reverse()
    assert track._version > version
    version = track._version

    track.clear()
    assert track._version > version


def test_track_read_only_access_keeps_version():
    track = MidiTrack([Message('note_on', time=0)])
    version = track._version

    # Reading, slicing and copying must not invalidate anything.
    _ = track[0]
    _ = track[:]
    _ = list(track)
    _ = len(track)
    _ = track.copy()
    assert track._version == version
