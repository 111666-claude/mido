# SPDX-FileCopyrightText: 2016 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

from .meta import KeySignatureError, MetaMessage, UnknownMetaMessage
from .midifiles import MidiFile
from .tempo_ramp import (
    CURVES,
    DEFAULT_MAX_EVENTS,
    TempoRampError,
    add_tempo_ramp,
    compute_tempo_ramp,
)
from .tracks import MidiTrack, merge_tracks
from .units import bpm2tempo, second2tick, tempo2bpm, tick2second

__all__ = [
    "CURVES",
    "DEFAULT_MAX_EVENTS",
    "KeySignatureError",
    "MetaMessage",
    "MidiFile",
    "MidiTrack",
    "TempoRampError",
    "UnknownMetaMessage",
    "add_tempo_ramp",
    "bpm2tempo",
    "compute_tempo_ramp",
    "merge_tracks",
    "second2tick",
    "tempo2bpm",
    "tick2second",
]
