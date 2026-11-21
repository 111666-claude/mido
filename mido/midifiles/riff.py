# SPDX-FileCopyrightText: 2026
#
# SPDX-License-Identifier: MIT

"""
RIFF RMID container support.

Some old games store a Standard MIDI File (SMF) inside a RIFF container
with the form type ``RMID``::

    RIFF (little endian size)
      RMID
      LIST ...        # optional chunks, e.g. LIST/INFO tags
      data (the SMF)

RIFF chunks use little endian sizes and are padded with one byte when
their data has an odd length. The embedded SMF itself is unchanged and
still uses big endian chunk sizes.

Only the first top-level ``data`` chunk is the MIDI stream. This module
extracts it so the regular SMF parser can read it, and can wrap an SMF
back into an RMID container.
"""

import struct
from io import BytesIO

RIFF_ID = b'RIFF'
RMID_ID = b'RMID'
DATA_ID = b'data'
LIST_ID = b'LIST'

# Every RIFF chunk has a 4-byte id followed by a 4-byte little endian
# size field.
CHUNK_HEADER_SIZE = 8
# The RIFF form type (RMID) is the first four bytes of the RIFF chunk
# data, as is the list type inside a LIST chunk.
TYPE_SIZE = 4


class RiffError(OSError):
    """A malformed RIFF/RMID container.

    ``offset`` is the byte position in the input stream where the
    problem was detected, or ``None`` when there is no specific
    position.
    """

    def __init__(self, message, offset=None):
        if offset is not None:
            message = f'{message} (at byte offset {offset})'
        super().__init__(message)
        self.offset = offset


def _read_exact(infile, size, offset, what):
    """Read exactly ``size`` bytes or raise with the failing offset."""
    data = infile.read(size)
    if len(data) != size:
        raise RiffError(f'truncated {what}', offset + len(data))
    return data


def _read_chunk_header(infile, end):
    """Read a chunk header, returning ``(name, size, start, data_start)``.

    ``end`` is the offset just past the enclosing chunk's data, so
    chunks that claim to extend past it can be rejected instead of
    being trusted.
    """
    start = infile.tell()
    header = infile.read(CHUNK_HEADER_SIZE)
    if len(header) == 0:
        # A clean end of the enclosing chunk or file.
        return None
    if len(header) < CHUNK_HEADER_SIZE:
        raise RiffError('truncated chunk header', start + len(header))

    name, size = struct.unpack('<4sL', header)
    if start + CHUNK_HEADER_SIZE + size > end:
        raise RiffError(
            f"{name!r} chunk size {size} extends past end of container "
            f"(container ends at byte offset {end})",
            start,
        )
    return name, size, start, start + CHUNK_HEADER_SIZE


def _skip_pad(infile, end):
    """Skip the single odd-byte pad of an odd-sized RIFF chunk."""
    if infile.tell() % 2:
        if infile.tell() >= end:
            raise RiffError('missing odd-byte pad', infile.tell())
        _read_exact(infile, 1, infile.tell(), 'odd-byte pad')


def _walk_chunks(infile, end, data_chunks, top_level):
    """Walk the chunks of a RIFF or LIST chunk.

    The embedded SMF is appended to ``data_chunks`` when a top-level
    ``data`` chunk is found. LIST chunks are descended into so that
    nested chunk sizes are validated against their container.
    """
    while infile.tell() < end:
        header = _read_chunk_header(infile, end)
        if header is None:
            raise RiffError('truncated chunk header', infile.tell())
        name, size, start, data_start = header

        if name == LIST_ID:
            # The list type (e.g. INFO) is part of the LIST chunk data,
            # followed by the nested chunks. The LIST chunk itself is a
            # RIFF chunk, so it too is padded when its size is odd.
            _read_exact(infile, TYPE_SIZE, data_start, 'LIST type')
            list_end = data_start + size
            _walk_chunks(infile, list_end, data_chunks, top_level=False)
            _skip_pad(infile, end)
        elif name == DATA_ID and top_level:
            if data_chunks:
                raise RiffError(
                    'multiple data chunks found',
                    start,
                )
            data_chunks.append(
                _read_exact(infile, size, data_start, 'data chunk'))
        else:
            # Unknown chunks (including data chunks nested in a LIST)
            # are skipped but still size-validated.
            _read_exact(infile, size, data_start, f'{name!r} chunk')

        _skip_pad(infile, end)

    return data_chunks


class _PrefixedStream:
    """Stream that replays ``prefix`` bytes before delegating to infile.

    This makes an already-read RIFF marker transparent to the SMF
    parser without requiring the underlying stream to be seekable.
    """

    def __init__(self, prefix, infile):
        self._prefix = prefix
        self._infile = infile
        self._pos = 0

    def read(self, size=-1):
        if self._pos < len(self._prefix):
            head = self._prefix[self._pos:]
            if size is not None and size >= 0:
                head = head[:size]
                rest_size = size - len(head)
            else:
                rest_size = -1
            self._pos += len(head)
            return head + self._infile.read(rest_size)
        return self._infile.read(size)

    def tell(self):
        if self._pos >= len(self._prefix):
            return self._infile.tell()
        return self._pos

    def seek(self, offset, whence=0):
        # Only rewinding to the start is needed by the SMF reader.
        if whence == 0 and offset == 0:
            self._pos = 0
            return 0
        raise OSError('only seek(0) is supported')


def read_rmid_smf(infile):
    """Extract the embedded SMF from an RMID stream.

    If the input is not a RIFF container a stream positioned at its
    beginning is returned (wrapped when the original stream is not
    seekable), so the caller can parse it as an ordinary SMF.

    Raises :class:`RiffError` with a byte offset for malformed RIFF
    data.
    """
    start = infile.tell()
    marker = infile.read(4)
    if len(marker) < 4 or marker != RIFF_ID:
        # Empty or non-RIFF input. Replay the consumed bytes so the
        # regular SMF parser sees the whole stream.
        if marker:
            return _PrefixedStream(marker, infile)
        return None

    size_data = _read_exact(infile, 4, start + 4, 'RIFF size')
    (riff_size,) = struct.unpack('<L', size_data)
    # riff_size counts everything after the id and size field,
    # including the form type and all contained chunks.
    riff_end = start + CHUNK_HEADER_SIZE + riff_size

    form_type = _read_exact(infile, TYPE_SIZE,
                            start + CHUNK_HEADER_SIZE, 'form type')
    if form_type != RMID_ID:
        raise RiffError(
            f'RIFF form type {form_type!r} is not {RMID_ID!r}',
            start + CHUNK_HEADER_SIZE,
        )

    data_chunks = _walk_chunks(infile, riff_end, [], top_level=True)
    if not data_chunks:
        raise RiffError('RMID container has no data chunk', riff_end)

    return BytesIO(data_chunks[0])


def build_rmid(smf):
    """Wrap a Standard MIDI File in an RMID container.

    The RIFF size is little endian and an odd-sized embedded file gets
    the required single-byte pad.
    """
    data = bytes(smf)
    pad = b'\x00' if len(data) % 2 else b''

    # 4 bytes form type + data chunk header + data + optional pad.
    riff_size = TYPE_SIZE + CHUNK_HEADER_SIZE + len(data) + len(pad)

    return b''.join((
        RIFF_ID,
        struct.pack('<L', riff_size),
        RMID_ID,
        DATA_ID,
        struct.pack('<L', len(data)),
        data,
        pad,
    ))
