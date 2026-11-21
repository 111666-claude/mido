#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

"""
Split one input between several outputs by channel and message type.

Drums (channel 9) go to one device, notes on channels 0 and 1 are moved
to channel 2 and sent to a synth, while SysEx and clock are broadcast
to both devices on explicit default paths.

Run with::

    python routing.py INPORT DRUMS SYNTH
"""
import sys

import mido
from mido.routing import Router, RoutingRule

if len(sys.argv) != 4:
    sys.exit(__doc__)

inport = mido.open_input(sys.argv[1])
drums = mido.open_output(sys.argv[2])
synth = mido.open_output(sys.argv[3])

router = Router(
    [
        RoutingRule([drums], channels=9),
        RoutingRule([synth], types=['note_on', 'note_off'],
                    channels=[0, 1], remap=2),
    ],
    # SysEx and clock are never caught by the channel rules above, so
    # they take these explicit paths.
    sysex=[drums, synth],
    realtime=[drums, synth],
    name='split',
)

print(f'Routing {inport.name} to {drums.name} and {synth.name}')
try:
    for message in inport:
        router.send(message)
except KeyboardInterrupt:
    pass
finally:
    router.close()
    inport.close()
    drums.close()
    synth.close()
