# SPDX-FileCopyrightText: 2024 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Optional resource budgets for reading untrusted MIDI files.

A malicious MIDI file can declare gigantic tracks or SysEx messages,
or contain a variable-length quantity (VLQ) that never terminates.
Parsing such a file can exhaust memory or take forever.

Pass a :class:`ReadBudget` to ``MidiFile(..., budget=...)`` to cap the
resources the parser is allowed to use.  Every limit is checked before
the corresponding bytes are read or memory is allocated.  When a limit
is exceeded the parser raises :class:`SecurityError`, which reports the
zero-based track number and the absolute byte offset where the limit
was hit.
"""

from numbers import Integral


class SecurityError(OSError):
    """Raised when a MIDI file exceeds a :class:`ReadBudget`.

    Attributes:
        track: zero-based track number, or ``None`` while the file
            header is being read.
        offset: absolute byte offset where the limit was exceeded.
    """

    def __init__(self, message, track=None, offset=None):
        super().__init__(message)
        self.track = track
        self.offset = offset


class ReadBudget:
    """Resource limits applied while parsing a MIDI file.

    Every limit is optional and defaults to ``None`` (unlimited), so a
    budget only enforces the limits you set.  Limits are checked before
    the data is read or allocated.

    Args:
        max_bytes: total number of bytes that may be read from the
            input stream.
        max_tracks: maximum number of tracks the file may declare.
        max_track_length: maximum declared track chunk length in bytes.
        max_event_length: maximum length of a single event payload
            (meta message or SysEx message), in bytes.
        max_events: maximum total number of events across all tracks.
        max_vlq_length: maximum number of bytes a single
            variable-length quantity (delta time or event length) may
            use.  The MIDI specification allows at most 4 bytes, which
            is the default.  Pass ``None`` to disable the cap.
    """

    def __init__(self, *, max_bytes=None, max_tracks=None,
                 max_track_length=None, max_event_length=None,
                 max_events=None, max_vlq_length=4):
        for name, value in [
            ('max_bytes', max_bytes),
            ('max_tracks', max_tracks),
            ('max_track_length', max_track_length),
            ('max_event_length', max_event_length),
            ('max_events', max_events),
        ]:
            self._check_limit(name, value)

        if (max_vlq_length is not None
                and (not isinstance(max_vlq_length, Integral)
                     or max_vlq_length < 1)):
            raise ValueError(
                'max_vlq_length must be a positive integer or None')

        self.max_bytes = max_bytes
        self.max_tracks = max_tracks
        self.max_track_length = max_track_length
        self.max_event_length = max_event_length
        self.max_events = max_events
        self.max_vlq_length = max_vlq_length

        # Running counters, updated by the parser.
        self.bytes_read = 0
        self.events_read = 0

    @staticmethod
    def _check_limit(name, value):
        if (value is not None
                and (not isinstance(value, Integral) or value < 0)):
            raise ValueError(
                f'{name} must be a non-negative integer or None')

    def check_bytes(self, reader, additional):
        """Check that reading ``additional`` bytes stays within budget."""
        if (self.max_bytes is not None
                and reader.bytes_read + additional > self.max_bytes):
            reader.fail(
                f'reading {additional} more byte(s) would exceed the '
                f'maximum of {self.max_bytes} byte(s)')

    def check_track_count(self, reader, count):
        if self.max_tracks is not None and count > self.max_tracks:
            reader.fail(
                f'file declares {count} track(s), exceeding the maximum '
                f'of {self.max_tracks}')

    def check_track_length(self, reader, size):
        if self.max_track_length is not None \
                and size > self.max_track_length:
            reader.fail(
                f'track length {size} exceeds the maximum of '
                f'{self.max_track_length} byte(s)')

    def check_event_length(self, reader, size, kind='event'):
        if self.max_event_length is not None \
                and size > self.max_event_length:
            reader.fail(
                f'{kind} length {size} exceeds the maximum event length '
                f'of {self.max_event_length} byte(s)')

    def check_event(self, reader):
        """Count one parsed event and enforce the total event limit."""
        self.events_read += 1
        if self.max_events is not None \
                and self.events_read > self.max_events:
            reader.fail(
                f'event count exceeds the maximum of {self.max_events} '
                f'event(s)')

    def check_vlq(self, reader, octets):
        """Enforce the VLQ length cap before reading another byte.

        ``octets`` is the number of VLQ bytes already consumed.
        """
        if (self.max_vlq_length is not None
                and octets >= self.max_vlq_length):
            reader.fail(
                f'variable-length quantity exceeds the maximum length of '
                f'{self.max_vlq_length} byte(s)')


class _CountingReader:
    """Wrap a binary stream, counting bytes and enforcing a budget.

    This works with non-seekable streams: when ``tell()`` is not
    supported the internal byte counter is used as the position, so no
    up-front read of the whole stream is required.
    """

    def __init__(self, infile, budget=None):
        self.file = infile
        self.budget = budget
        self.bytes_read = 0
        # Context for error messages, set by the parser.
        self.track = None
        # Absolute position just past the end of the current track.
        self.track_end = None

    def read(self, size=-1):
        if self.budget is not None and size is not None and size >= 0:
            self.budget.check_bytes(self, size)

        data = self.file.read(size)
        self.bytes_read += len(data)

        if self.budget is not None:
            self.budget.bytes_read = self.bytes_read
            # Catches short reads and unbounded read() calls that the
            # pre-check above cannot cover.
            if (self.budget.max_bytes is not None
                    and self.bytes_read > self.budget.max_bytes):
                self.fail(
                    f'total input exceeds the maximum of '
                    f'{self.budget.max_bytes} byte(s)')

        return data

    def tell(self):
        """Return the absolute byte position in the input."""
        try:
            return self.file.tell()
        except (AttributeError, OSError):
            return self.bytes_read

    def fail(self, message):
        raise SecurityError(message, track=self.track,
                            offset=self.tell())

    def check_fits_track(self, size, kind):
        """Check that a declared payload fits inside the current track."""
        if self.track_end is not None:
            remaining = self.track_end - self.bytes_read
            if size > remaining:
                self.fail(
                    f'declared {kind} length {size} exceeds remaining '
                    f'input ({remaining} byte(s)) in track')
