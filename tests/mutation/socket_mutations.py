# SPDX-FileCopyrightText: 2026
#
# SPDX-License-Identifier: MIT

"""Deliberately broken variants of the MIDI-over-TCP chain.

The loopback suite in ``tests/test_socket_loopback.py`` is only useful if
it actually fails when a link in the chain is broken.  This module
provides a pytest plugin that monkeypatches one specific link at a time,
selected by the ``MIDO_SOCKET_MUTATION`` environment variable:

``frag_parser_reset``
    SocketPort discards the Parser/Tokenizer state at the start of every
    ``_receive`` call.  A MIDI message whose status byte and data bytes
    arrive in different TCP segments can no longer be reassembled.
    -> breaks the SocketPort._receive -> Parser/Tokenizer link.

``eof_no_close``
    SocketPort ignores the empty read (peer EOF) and never closes itself.
    Disconnected ports leak and are never pruned by PortServer.
    -> breaks the EOF detection / port-cleanup link.

``broadcast_first_only``
    PortServer sends each message only to the first live port instead of
    fanning it out to every connected client.
    -> breaks the PortServer -> MultiPort broadcast link.

The plugin is enabled from ``tests/test_socket_mutations.py`` via a
subprocess so that each mutation is applied to a fresh interpreter and
cannot leak between tests.
"""

import os

from mido import sockets


def _apply_frag_parser_reset():
    """Break stream reassembly across TCP segment boundaries."""
    original_receive = sockets.SocketPort._receive

    def mutated_receive(self, block=True):
        # A fresh parser on every poll forgets running status and any
        # partially received message, so split status/data bytes never
        # join back into a whole MIDI message.
        from mido.parser import Parser
        self._parser = Parser()
        self._messages = self._parser.messages
        return original_receive(self, block)

    sockets.SocketPort._receive = mutated_receive


def _apply_eof_no_close():
    """Break peer-disconnect detection: an empty read is ignored."""
    def mutated_receive(self, block=True):
        from mido.sockets import _is_readable
        while _is_readable(self._socket):
            try:
                byte = self._rfile.read(1)
            except OSError as err:
                raise OSError(err.args[1])
            if len(byte) == 0:
                # Deliberately swallow the EOF: do not self.close().
                break
            else:
                self._parser.feed_byte(ord(byte))
                self._bytes_consumed += 1

    sockets.SocketPort._receive = mutated_receive


def _apply_broadcast_first_only():
    """Break fan-out: deliver only to the first live port."""
    def mutated_send(self, message):
        self._update_ports()
        for port in self.ports:
            if not port.closed:
                port.send(message)
                break  # one client only; the rest are starved

    sockets.PortServer._send = mutated_send


MUTATIONS = {
    'frag_parser_reset': _apply_frag_parser_reset,
    'eof_no_close': _apply_eof_no_close,
    'broadcast_first_only': _apply_broadcast_first_only,
}


def pytest_configure(config):  # pragma: no cover - exercised in subprocess
    name = os.environ.get('MIDO_SOCKET_MUTATION')
    if name:
        if name not in MUTATIONS:
            raise RuntimeError(
                f'unknown mutation {name!r}; choose from '
                f'{sorted(MUTATIONS)}')
        MUTATIONS[name]()
