# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: CC0-1.0

"""Tests for the MIDI Tuning Standard Single Note Tuning Change support.

Reference vectors come from the MMA "MIDI Tuning" specification and
from the `tune` crate (https://crates.io/crates/tune), whose MTS
implementation is validated against the same specification.
"""

import math

from pytest import raises

from mido import (
    MAX_NOTE_CHANGES,
    NO_CHANGE_FREQUENCY,
    Message,
    MidiFile,
    MidiTrack,
    NoteTuning,
    data_to_frequency,
    frequency_to_data,
    parse_single_note_tuning_change,
    single_note_tuning_change,
)
from mido.mts import NOTE_0_FREQUENCY

# Frequency data word examples printed in the MMA specification.
# The strict A440-derived formula matches most table entries to the
# four decimals printed by the specification.
SPEC_FREQUENCY_WORDS = [
    ((0x00, 0x00, 0x00), 8.1758),    # C, normal tuning of MIDI key 0
    ((0x01, 0x00, 0x00), 8.6620),
    ((0x0C, 0x00, 0x00), 16.3516),
    ((0x3C, 0x00, 0x00), 261.6256),  # middle C
    ((0x44, 0x7F, 0x7F), 439.9984),  # one step below A440
    ((0x45, 0x00, 0x00), 440.0000),  # A440
    ((0x45, 0x00, 0x01), 440.0016),
    ((0x78, 0x00, 0x00), 8372.0181),  # normal tuning of MIDI key 120
]

# Entries whose printed value in the 1990s specification table differs
# from the strict formula by a few units in the last decimal; they are
# still checked against the formula's own value so regressions show.
SPEC_TABLE_ROUNDED_ENTRIES = [
    ((0x3D, 0x00, 0x00), 277.1826),  # spec table prints 277.1827
    ((0x7F, 0x00, 0x00), 12543.8540),  # spec table prints 12543.8800
    ((0x7F, 0x7F, 0x7E), 13289.6566),  # spec table prints 13289.7300
]


def test_spec_frequency_words_decode():
    for word, hertz in SPEC_FREQUENCY_WORDS + SPEC_TABLE_ROUNDED_ENTRIES:
        assert round(data_to_frequency(word), 4) == hertz


def test_spec_frequency_words_round_trip():
    for word, _ in SPEC_FREQUENCY_WORDS:
        assert frequency_to_data(data_to_frequency(word)) == word


def test_no_change_word_is_reserved():
    assert data_to_frequency(NO_CHANGE_FREQUENCY) is None


def test_top_of_range_word():
    # 7F 7F 7E is the highest usable word ("top of range" in the spec).
    top = data_to_frequency((0x7F, 0x7F, 0x7E))
    assert frequency_to_data(top) == (0x7F, 0x7F, 0x7E)
    assert top > NOTE_0_FREQUENCY * 2 ** (127 / 12.0)


def test_reserved_word_cannot_be_produced_from_a_frequency():
    # A frequency mapping onto 7F 7F 7F must raise instead of silently
    # emitting the reserved "no change" word.
    reserved_frequency = data_to_frequency((0x7F, 0x7F, 0x7E))
    with raises(ValueError):
        frequency_to_data(reserved_frequency * 1.0001)


def test_frequency_resolution_is_better_than_one_cent():
    # One 14-bit step at A440 is 100/16384 cents, about 0.0061 cents.
    a440 = data_to_frequency((0x45, 0x00, 0x00))
    one_step = data_to_frequency((0x45, 0x00, 0x01))
    step_in_cents = 1200 * math.log2(one_step / a440)
    assert abs(step_in_cents - 100.0 / 16384.0) < 1e-9
    assert step_in_cents < 0.01


def test_frequency_to_data_rejects_bad_inputs():
    for bad in (0, -1.0, float('nan'), float('inf'), float('-inf')):
        with raises(ValueError):
            frequency_to_data(bad)
    for bad in ('440', None, True):
        with raises(TypeError):
            frequency_to_data(bad)


def test_frequency_to_data_out_of_range():
    with raises(ValueError):
        frequency_to_data(20000.0)


def test_data_to_frequency_validates_bytes():
    with raises(ValueError):
        data_to_frequency((0, 0, 128))
    with raises(ValueError):
        data_to_frequency((0, 0, -1))
    with raises(ValueError):
        data_to_frequency((0, 0))
    with raises(TypeError):
        data_to_frequency((0, 0, '1'))


def test_reference_single_note_message_bytes():
    # tune crate reference vector: retune A4 (key 69) to 445 Hz.
    msg = single_note_tuning_change([(69, 445.0)])
    assert msg.hex() == 'F0 7F 7F 08 02 00 01 45 45 19 05 F7'


def test_default_header_and_options():
    msg = single_note_tuning_change([(60, 261.63)])
    assert msg.type == 'sysex'
    # 7F = realtime, broadcast device, MTS 08 02, program 0, count 1.
    assert tuple(msg.data[:6]) == (0x7F, 0x7F, 0x08, 0x02, 0x00, 0x01)


def test_custom_device_program_and_non_realtime():
    msg = single_note_tuning_change(
        [(69, 445.0)], device_id=55, program=66, realtime=False)
    assert msg.hex() == 'F0 7E 37 08 02 42 01 45 45 19 05 F7'
    parsed = parse_single_note_tuning_change(msg)
    assert parsed.realtime is False
    assert parsed.device_id == 55
    assert parsed.program == 66


def test_multiple_notes_in_one_message():
    changes = [(60, 261.63), (69, 440.0), (72, 500.0)]
    msg = single_note_tuning_change(changes)
    parsed = parse_single_note_tuning_change(msg)
    assert [change.note for change in parsed.changes] == [60, 69, 72]
    assert len(msg.bytes()) == 8 + 3 * 4


def test_no_change_entries():
    msg = single_note_tuning_change([(60, None), (69, 440.0)])
    assert tuple(msg.data[6:10]) == (60, 0x7F, 0x7F, 0x7F)
    parsed = parse_single_note_tuning_change(msg)
    assert parsed.changes[0] == NoteTuning(60, None)
    assert parsed.changes[1].note == 69

    # The no_change keyword produces the same reserved word.
    explicit = single_note_tuning_change([(60, 440.0)], no_change=[60])
    assert tuple(explicit.data[6:10]) == (60, 0x7F, 0x7F, 0x7F)

    note_tuning = single_note_tuning_change([NoteTuning(60, None)])
    assert tuple(note_tuning.data[6:10]) == (60, 0x7F, 0x7F, 0x7F)


def test_time_is_carried_by_the_message():
    msg = single_note_tuning_change([(69, 445.0)], time=42)
    assert msg.time == 42
    parsed = parse_single_note_tuning_change(msg)
    assert parsed.time == 42


def test_round_trip_preserves_bytes():
    changes = [(12, 16.35), (60, 261.6256), (69, 445.0), (127, 12000.0)]
    msg = single_note_tuning_change(changes, device_id=3, program=7, time=5)
    parsed = parse_single_note_tuning_change(msg)
    rebuilt = single_note_tuning_change(
        [(c.note, c.frequency) for c in parsed.changes],
        device_id=parsed.device_id,
        program=parsed.program,
        realtime=parsed.realtime,
        time=parsed.time)
    assert rebuilt.bytes() == msg.bytes()


def test_parse_accepts_raw_bytes_with_and_without_framing():
    msg = single_note_tuning_change([(69, 445.0)], device_id=1)
    framed = msg.bytes()
    unframed = msg.data
    assert parse_single_note_tuning_change(framed).changes == \
        parse_single_note_tuning_change(unframed).changes


def test_parse_preserves_frequency_accuracy():
    msg = single_note_tuning_change([(69, 445.0)])
    parsed = parse_single_note_tuning_change(msg)
    # The word 45 19 05 decodes to within one 0.0061-cent step of 445 Hz.
    error_cents = 1200 * math.log2(parsed.changes[0].frequency / 445.0)
    assert abs(error_cents) < 0.007


def test_maximum_number_of_changes():
    changes = [(note, 440.0) for note in range(MAX_NOTE_CHANGES)]
    msg = single_note_tuning_change(changes)
    assert len(msg.bytes()) == 8 + MAX_NOTE_CHANGES * 4
    assert len(parse_single_note_tuning_change(msg).changes) == MAX_NOTE_CHANGES


def test_too_many_changes_rejected():
    changes = [(note % 128, 440.0) for note in range(MAX_NOTE_CHANGES + 1)]
    with raises(ValueError):
        single_note_tuning_change(changes)


def test_empty_change_list_rejected():
    with raises(ValueError):
        single_note_tuning_change([])


def test_duplicate_note_rejected_on_build():
    with raises(ValueError):
        single_note_tuning_change([(60, 200.0), (60, 300.0)])


def test_duplicate_note_rejected_on_parse():
    raw = [0xF0, 0x7F, 0x7F, 0x08, 0x02, 0x00, 0x02,
           60, 0x3C, 0x00, 0x00,
           60, 0x3C, 0x00, 0x00, 0xF7]
    with raises(ValueError):
        parse_single_note_tuning_change(raw)


def test_invalid_note_numbers():
    with raises(ValueError):
        single_note_tuning_change([(128, 440.0)])
    with raises(ValueError):
        single_note_tuning_change([(-1, 440.0)])
    with raises(TypeError):
        single_note_tuning_change([(60.5, 440.0)])
    with raises(TypeError):
        single_note_tuning_change([(True, 440.0)])


def test_invalid_device_and_program():
    with raises(ValueError):
        single_note_tuning_change([(60, 440.0)], device_id=128)
    with raises(ValueError):
        single_note_tuning_change([(60, 440.0)], program=-1)
    with raises(TypeError):
        single_note_tuning_change([(60, 440.0)], device_id=1.5)


def test_invalid_change_shape():
    with raises(TypeError):
        single_note_tuning_change([440.0])
    with raises(TypeError):
        single_note_tuning_change([(60, 440.0, 1)])


def test_parse_rejects_wrong_message_types():
    with raises(ValueError):
        parse_single_note_tuning_change(Message('note_on'))
    # Universal sysex but not MTS.
    not_mts = Message('sysex', data=[0x7F, 0x7F, 0x09, 0x02, 0x00, 0x01])
    with raises(ValueError):
        parse_single_note_tuning_change(not_mts)
    # MTS but wrong sub-ID#2.
    not_note_change = Message('sysex',
                              data=[0x7F, 0x7F, 0x08, 0x01, 0x00, 0x01])
    with raises(ValueError):
        parse_single_note_tuning_change(not_note_change)
    # Not a universal sysex header.
    with raises(ValueError):
        parse_single_note_tuning_change(
            [0xF0, 0x00, 0x7F, 0x08, 0x02, 0, 1, 69, 0x45, 0, 0, 0xF7])


def test_parse_rejects_truncated_payloads():
    good = [0xF0, 0x7F, 0x7F, 0x08, 0x02, 0x00, 0x01,
            69, 0x45, 0x19, 0x05, 0xF7]
    for length in range(4, len(good)):
        with raises(ValueError):
            parse_single_note_tuning_change(bytes(good[:length]))


def test_parse_rejects_count_mismatch_and_trailing_bytes():
    base = [0xF0, 0x7F, 0x7F, 0x08, 0x02, 0x00]
    # Declares two changes, contains one.
    with raises(ValueError):
        parse_single_note_tuning_change(
            base + [0x02, 69, 0x45, 0x19, 0x05, 0xF7])
    # Declares one change, contains two.
    with raises(ValueError):
        parse_single_note_tuning_change(
            base + [0x01,
                    69, 0x45, 0x19, 0x05,
                    70, 0x46, 0x19, 0x05, 0xF7])


def test_parse_rejects_non_seven_bit_data():
    bad = [0xF0, 0x7F, 0x7F, 0x08, 0x02, 0x00, 0x01,
           69, 0xFF, 0x19, 0x05, 0xF7]
    with raises(ValueError):
        parse_single_note_tuning_change(bad)


def test_parse_rejects_zero_change_count():
    with raises(ValueError):
        parse_single_note_tuning_change(
            [0xF0, 0x7F, 0x7F, 0x08, 0x02, 0x00, 0x00, 0xF7])


def test_midifile_round_trip_preserves_time_and_events():
    mid = MidiFile(type=1, ticks_per_beat=480)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(Message('program_change', program=5, time=0))
    track.append(Message('note_on', note=60, velocity=64, time=0))
    mts = single_note_tuning_change(
        [(69, 445.0), (70, 470.25)], device_id=0, program=2, time=123)
    track.append(mts)
    track.append(Message('note_off', note=60, velocity=64, time=240))

    import io
    buffer = io.BytesIO()
    mid.save(file=buffer)
    serialized = buffer.getvalue()

    buffer.seek(0)
    reloaded = MidiFile(file=buffer)
    events = reloaded.tracks[0]

    assert events[0] == Message('program_change', program=5, time=0)
    assert events[1] == Message('note_on', note=60, velocity=64, time=0)
    assert events[2].time == 123
    decoded = parse_single_note_tuning_change(events[2])
    assert decoded.time == 123
    assert decoded.device_id == 0
    assert decoded.program == 2
    assert [c.note for c in decoded.changes] == [69, 70]
    assert events[3] == Message('note_off', note=60, velocity=64, time=240)

    buffer2 = io.BytesIO()
    reloaded.save(file=buffer2)
    assert buffer2.getvalue() == serialized


def test_parser_feeds_realtime_message():
    msg = single_note_tuning_change([(69, 445.0)])
    parser = __import__('mido').Parser()
    parser.feed(msg.bytes())
    parsed_messages = list(parser)
    assert len(parsed_messages) == 1
    assert parse_single_note_tuning_change(
        parsed_messages[0]).changes[0].note == 69


def test_syx_file_round_trip(tmpdir):
    from mido import read_syx_file, write_syx_file
    messages = [
        single_note_tuning_change([(60, None), (69, 445.0)]),
        single_note_tuning_change([(60, 261.63)], realtime=False),
    ]
    path = tmpdir.join('tuning.syx').strpath
    write_syx_file(path, messages)
    assert read_syx_file(path) == messages


def test_note_tuning_equality_and_repr():
    assert NoteTuning(60, 440.0) == NoteTuning(60, 440.0)
    assert NoteTuning(60, None) == NoteTuning(60, None)
    assert NoteTuning(60, 440.0) != NoteTuning(60, 441.0)
    assert 'note=60' in repr(NoteTuning(60, 440.0))


def test_decoded_message_equality():
    first = parse_single_note_tuning_change(
        single_note_tuning_change([(60, None), (69, 445.0)]))
    second = parse_single_note_tuning_change(
        single_note_tuning_change([(60, None), (69, 445.0)]))
    assert first == second
    other = parse_single_note_tuning_change(
        single_note_tuning_change([(69, 445.0)]))
    assert first != other
    assert 'SingleNoteTuningChange' in repr(first)
