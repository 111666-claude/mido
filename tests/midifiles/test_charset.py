# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for per-file, thread-isolated charset handling of meta text.

The charset used while reading or writing a MidiFile must be isolated
between concurrent files instead of being a plain module global that
threads can overwrite in each other's faces.
"""
import io
import threading

import pytest

from mido import MidiFile, MidiTrack
from mido.midifiles import meta
from mido.midifiles.meta import (
    MetaMessage,
    UnknownMetaMessage,
    decode_string,
    encode_string,
    meta_charset,
)

# The documented default charset. 'é' is 0xE9 in Latin-1 but
# 0xC3 0xA9 in UTF-8, which makes the two easy to tell apart.
E_ACUTE = 'é'
LATIN1_EACUTE = [0xe9]
UTF8_EACUTE = [0xc3, 0xa9]


def active_charset_encodes_latin1():
    """Observe the active charset through documented encode behavior."""
    return encode_string(E_ACUTE) == LATIN1_EACUTE


def active_charset_encodes_utf8():
    return encode_string(E_ACUTE) == UTF8_EACUTE


def make_midi(text, charset):
    """Build a minimal type-0 MIDI file with one lyrics meta event."""
    payload = text.encode(charset)
    body = bytearray([0x00, 0xff, 0x05, len(payload)]) + payload
    body += bytearray([0x00, 0xff, 0x2f, 0x00])  # end_of_track
    out = bytearray(b'MThd')
    out += (6).to_bytes(4, 'big')
    out += (0).to_bytes(2, 'big')       # type 0
    out += (1).to_bytes(2, 'big')       # one track
    out += (480).to_bytes(2, 'big')     # ticks per beat
    out += b'MTrk' + len(body).to_bytes(4, 'big') + body
    return bytes(out)


UTF8_RAW = make_midi(E_ACUTE, 'utf-8')       # payload C3 A9
LATIN1_RAW = make_midi(E_ACUTE, 'latin1')    # payload E9


class _BarrierFile:
    """File wrapper synchronizing two threads around the payload read.

    read() #8 returns the single lyric payload byte. Both threads are
    held there, inside their meta_charset() context, until each has
    decoded the payload; only then may either leave that context.
    """

    def __init__(self, wrapped, enter_barrier, leave_barrier):
        self._wrapped = wrapped
        self._enter_barrier = enter_barrier
        self._leave_barrier = leave_barrier
        self._reads = 0

    def read(self, size):
        self._reads += 1
        data = self._wrapped.read(size)
        if self._reads == 8:
            self._enter_barrier.wait()
            self._leave_barrier.wait()
        return data

    def tell(self):
        return self._wrapped.tell()


def test_default_charset_is_latin1():
    assert encode_string(E_ACUTE) == LATIN1_EACUTE
    assert decode_string(LATIN1_EACUTE) == E_ACUTE


def test_metacharset_restores_on_normal_exit():
    assert active_charset_encodes_latin1()
    with meta_charset('utf-8'):
        assert active_charset_encodes_utf8()
    assert active_charset_encodes_latin1()


def test_metacharset_restores_on_exception():
    with pytest.raises(RuntimeError):
        with meta_charset('utf-8'):
            assert active_charset_encodes_utf8()
            raise RuntimeError('boom')
    assert active_charset_encodes_latin1()


def test_nested_metacharset_like_a_stack():
    with meta_charset('utf-8'):
        assert active_charset_encodes_utf8()
        with meta_charset('latin1'):
            assert active_charset_encodes_latin1()
        assert active_charset_encodes_utf8()
    assert active_charset_encodes_latin1()


def test_nested_metacharset_restores_on_inner_exception():
    with meta_charset('utf-8'):
        with pytest.raises(ValueError):
            with meta_charset('latin1'):
                raise ValueError('boom')
        assert active_charset_encodes_utf8()
    assert active_charset_encodes_latin1()


def test_invalid_charset_name_raises_lookup_error():
    # The bad name is rejected when the context is entered, before any
    # string conversion happens, so the error points at the charset
    # configuration rather than at an unrelated encode/decode call.
    with pytest.raises(LookupError):
        with meta_charset('this-is-not-a-codec'):
            pass
    # A failed entry must not change the active charset.
    assert active_charset_encodes_latin1()


def test_invalid_charset_name_raises_from_midifile():
    with pytest.raises(LookupError):
        MidiFile(file=io.BytesIO(UTF8_RAW), charset='this-is-not-a-codec')

    # The charset is only validated when it is actually used, i.e.
    # while reading or writing.
    mid = MidiFile(charset='this-is-not-a-codec')
    with pytest.raises(LookupError):
        mid.save(file=io.BytesIO())
    assert active_charset_encodes_latin1()


def test_unencodable_character_raises_with_context():
    # 'é' cannot be encoded in plain ASCII.
    mid = MidiFile(charset='ascii')
    mid.tracks.append(MidiTrack([
        MetaMessage('lyrics', text=E_ACUTE),
        MetaMessage('end_of_track'),
    ]))
    with pytest.raises(UnicodeEncodeError):
        mid.save(file=io.BytesIO())


def test_undecodable_bytes_raise_inside_file_charset():
    # C3 A9 is not valid ASCII, so an ASCII-scoped read must surface
    # the codec error instead of silently producing mojibake.
    with pytest.raises(UnicodeDecodeError):
        MidiFile(file=io.BytesIO(UTF8_RAW), charset='ascii')


def test_unknown_meta_message_keeps_raw_data():
    # 0x7d is not assigned by the MIDI spec and must round-trip
    # untouched regardless of the active charset.
    payload = [0x01, 0xc3, 0xa9, 0xe9]
    raw = bytearray(b'MThd') + (6).to_bytes(4, 'big')
    raw += (0).to_bytes(2, 'big') + (1).to_bytes(2, 'big')
    raw += (480).to_bytes(2, 'big')
    body = bytearray([0x00, 0xff, 0x7d, len(payload)]) + bytearray(payload)
    body += bytearray([0x00, 0xff, 0x2f, 0x00])
    raw += b'MTrk' + len(body).to_bytes(4, 'big') + body

    for charset in ('latin1', 'utf-8'):
        mid = MidiFile(file=io.BytesIO(bytes(raw)), charset=charset)
        msg = mid.tracks[0][0]
        assert isinstance(msg, UnknownMetaMessage)
        assert list(msg.data) == payload

        out = io.BytesIO()
        mid.save(file=out)
        assert bytes(raw) in out.getvalue()


def test_single_thread_charset_round_trip():
    """Single-threaded regression: each file keeps its own charset."""
    utf8_mid = MidiFile(file=io.BytesIO(UTF8_RAW), charset='utf-8')
    latin1_mid = MidiFile(file=io.BytesIO(LATIN1_RAW), charset='latin1')

    assert utf8_mid.tracks[0][0].text == E_ACUTE
    assert latin1_mid.tracks[0][0].text == E_ACUTE

    # Saving under the same charsets must reproduce the original bytes.
    utf8_out, latin1_out = io.BytesIO(), io.BytesIO()
    utf8_mid.save(file=utf8_out)
    latin1_mid.save(file=latin1_out)
    assert utf8_out.getvalue() == UTF8_RAW
    assert latin1_out.getvalue() == LATIN1_RAW


def test_metamessage_uses_default_charset_without_file_context():
    # No MidiFile context active -> the documented latin1 default.
    assert MetaMessage('text', text=E_ACUTE).bytes()[-1] == 0xe9
    msg = MetaMessage.from_bytes([0xff, 0x01, 0x01, 0xe9])
    assert msg.text == E_ACUTE


def test_concurrent_reads_keep_their_own_charset():
    """Two threads parse UTF-8 and Latin-1 files behind barriers."""
    enter_barrier = threading.Barrier(2)
    leave_barrier = threading.Barrier(2)
    results = {}
    errors = []

    def worker(name, charset, raw):
        try:
            mid = MidiFile(
                file=_BarrierFile(io.BytesIO(raw),
                                  enter_barrier, leave_barrier),
                charset=charset)
            results[name] = mid.tracks[0][0].text
        except Exception as exc:  # report any thread failure
            errors.append(exc)

    threads = [
        threading.Thread(target=worker,
                         args=('utf8', 'utf-8', UTF8_RAW)),
        threading.Thread(target=worker,
                         args=('latin1', 'latin1', LATIN1_RAW)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert results == {'utf8': E_ACUTE, 'latin1': E_ACUTE}
    # Threads must not leak their charset into the surrounding context.
    assert active_charset_encodes_latin1()


def test_concurrent_writes_keep_their_own_charset():
    """Two threads save UTF-8 and Latin-1 files behind barriers."""
    real_encode = meta.encode_string
    enter_barrier = threading.Barrier(2)
    leave_barrier = threading.Barrier(2)

    def synchronized_encode(string):
        enter_barrier.wait()
        try:
            return real_encode(string)
        finally:
            leave_barrier.wait()

    outputs = {}
    errors = []

    def worker(name, charset):
        try:
            mid = MidiFile(charset=charset)
            mid.tracks.append(MidiTrack([
                MetaMessage('lyrics', text=E_ACUTE),
                MetaMessage('end_of_track'),
            ]))
            out = io.BytesIO()
            # Park both threads inside meta_charset(), right at string
            # encoding time, so a shared global would be overwritten.
            meta.encode_string = synchronized_encode
            try:
                mid.save(file=out)
            finally:
                meta.encode_string = real_encode
            outputs[name] = out.getvalue()
        except Exception as exc:  # report any thread failure
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=('utf8', 'utf-8')),
        threading.Thread(target=worker, args=('latin1', 'latin1')),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert bytes(UTF8_EACUTE) in outputs['utf8']
    assert bytes(LATIN1_EACUTE) in outputs['latin1']
    assert bytes(UTF8_EACUTE) not in outputs['latin1']
    assert active_charset_encodes_latin1()
