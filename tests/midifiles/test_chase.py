# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

import io

import pytest

import mido
from mido import Message, MetaMessage, MidiFile, MidiTrack
from mido.midifiles.chase import (
    BANK_SELECT_LSB,
    BANK_SELECT_MSB,
    ChannelChaseState,
    ChaseState,
    chase_events,
    chase_state,
)


def cc(control, value, channel=0, time=0):
    return Message('control_change', channel=channel, control=control,
                   value=value, time=time)


def make_file(tracks, type=1, ticks_per_beat=480):
    return MidiFile(type=type, ticks_per_beat=ticks_per_beat,
                    tracks=[MidiTrack(track) for track in tracks])


def as_tuples(messages):
    """Convert messages to simple comparable tuples."""
    result = []
    for msg in messages:
        if msg.type == 'control_change':
            result.append(('cc', msg.channel, msg.control, msg.value))
        elif msg.type == 'program_change':
            result.append(('program', msg.channel, msg.program))
        elif msg.type == 'pitchwheel':
            result.append(('pitchwheel', msg.channel, msg.pitch))
        elif msg.type == 'aftertouch':
            result.append(('aftertouch', msg.channel, msg.value))
        else:
            result.append((msg.type, msg.channel))
    return result


# ---------------------------------------------------------------------
# Basic reconstruction
# ---------------------------------------------------------------------

def test_chase_returns_midi_track_with_zero_times():
    mid = make_file([[
        cc(BANK_SELECT_MSB, 1, time=0),
        cc(BANK_SELECT_LSB, 2, time=0),
        Message('program_change', program=42, time=0),
        cc(7, 100, time=0),
        Message('pitchwheel', pitch=-1000, time=0),
        Message('aftertouch', value=64, time=0),
        cc(7, 20, time=480),
    ]])

    track = mid.chase(480)

    assert isinstance(track, MidiTrack)
    assert all(msg.time == 0 for msg in track)


def test_chase_rebuilds_all_tracked_state():
    mid = make_file([[
        cc(BANK_SELECT_MSB, 1),
        cc(BANK_SELECT_LSB, 2),
        Message('program_change', program=42),
        cc(7, 100),
        cc(10, 64),
        Message('pitchwheel', pitch=-1000),
        Message('aftertouch', value=64),
    ]])

    assert as_tuples(mid.chase(1)) == [
        ('cc', 0, BANK_SELECT_MSB, 1),
        ('cc', 0, BANK_SELECT_LSB, 2),
        ('program', 0, 42),
        ('cc', 0, 7, 100),
        ('cc', 0, 10, 64),
        ('pitchwheel', 0, -1000),
        ('aftertouch', 0, 64),
    ]


def test_chase_uses_latest_value_of_each_controller():
    mid = make_file([[
        cc(7, 10),
        cc(7, 20),
        cc(7, 30),
        cc(10, 40),
    ]])

    assert as_tuples(mid.chase(1)) == [
        ('cc', 0, 7, 30),
        ('cc', 0, 10, 40),
    ]


def test_chase_bank_select_precedes_program_change():
    # Bank select after a program change still has to be sent first so
    # that the program change is interpreted with the right bank.
    mid = make_file([[
        Message('program_change', program=5),
        cc(BANK_SELECT_MSB, 3),
        cc(BANK_SELECT_LSB, 4),
    ]])

    messages = list(mid.chase(1))
    assert messages.index(
        cc(BANK_SELECT_MSB, 3)) < messages.index(
        cc(BANK_SELECT_LSB, 4)) < messages.index(
        Message('program_change', program=5))


def test_chase_does_not_retrigger_notes():
    mid = make_file([[
        Message('note_on', note=60, velocity=100),
        Message('note_off', note=60, time=100),
        Message('note_on', note=62, velocity=90, time=100),
    ]])

    messages = mid.chase(200)

    assert messages == []
    assert all(msg.type not in ('note_on', 'note_off')
               for msg in messages)


def test_chase_note_on_is_not_chased_even_with_state_changes():
    mid = make_file([[
        cc(7, 100),
        Message('note_on', note=60, velocity=100),
        Message('pitchwheel', pitch=200),
    ]])

    assert as_tuples(mid.chase(3)) == [
        ('cc', 0, 7, 100),
        ('pitchwheel', 0, 200),
    ]


def test_chase_ignores_meta_messages():
    mid = make_file([[
        MetaMessage('track_name', name='Piano'),
        MetaMessage('set_tempo', tempo=250000),
        cc(7, 90),
    ]])

    assert as_tuples(mid.chase(1)) == [('cc', 0, 7, 90)]


# ---------------------------------------------------------------------
# Tick semantics
# ---------------------------------------------------------------------

def test_message_exactly_at_target_is_not_included():
    mid = make_file([[
        cc(7, 10),
        cc(7, 20, time=480),
    ]])

    assert as_tuples(mid.chase(480)) == [('cc', 0, 7, 10)]
    assert as_tuples(mid.chase(481)) == [('cc', 0, 7, 20)]


def test_chase_at_tick_zero_is_empty():
    mid = make_file([[
        cc(7, 10),
        MetaMessage('set_tempo', tempo=1000000),
    ]])

    assert mid.chase(0) == []
    assert mid.chase_state(0).channels == {}


def test_chasing_past_end_returns_final_state():
    mid = make_file([[
        cc(7, 10),
        Message('program_change', program=3, time=480),
    ]])

    assert as_tuples(mid.chase(10_000)) == [
        ('program', 0, 3),
        ('cc', 0, 7, 10),
    ]


def test_chase_does_not_need_tempo():
    # No set_tempo messages at all: the default tempo applies, and the
    # chase is purely tick based.
    mid = make_file([[
        cc(7, 11, time=0),
        cc(11, 22, time=999),
    ]])

    assert as_tuples(mid.chase(999)) == [('cc', 0, 7, 11)]
    assert as_tuples(mid.chase(1000)) == [
        ('cc', 0, 7, 11),
        ('cc', 0, 11, 22),
    ]


def test_chase_empty_file():
    mid = MidiFile(type=0)

    assert mid.chase(0) == []
    assert mid.chase(12345) == []
    assert mid.chase_state(0).channels == {}


def test_chase_empty_tracks():
    mid = make_file([[], []])

    assert mid.chase(0) == []
    assert mid.chase(999) == []


def test_chase_negative_tick_raises():
    mid = make_file([[]])

    with pytest.raises(ValueError):
        mid.chase(-1)
    with pytest.raises(ValueError):
        chase_state(mid, -1)


def test_chase_non_integer_tick_raises():
    mid = make_file([[]])

    with pytest.raises(TypeError):
        mid.chase(1.5)
    with pytest.raises(TypeError):
        mid.chase('10')
    with pytest.raises(TypeError):
        mid.chase(True)


def test_chase_negative_absolute_tick_raises():
    # A track whose deltas add up to a negative absolute time is
    # malformed; it is rejected instead of producing nonsense ticks.
    mid = make_file([[
        cc(7, 10, time=-5),
    ]])

    with pytest.raises(ValueError):
        mid.chase(0)


def test_chase_type_2_file_raises():
    mid = make_file([[], []], type=2)

    with pytest.raises(TypeError):
        mid.chase(0)
    with pytest.raises(TypeError):
        chase_events(mid, 0)


# ---------------------------------------------------------------------
# Cross-track ordering and conflicts
# ---------------------------------------------------------------------

def test_same_tick_order_across_tracks_is_by_track_index():
    # Both tracks set the same controller on the same channel at the
    # same absolute tick.  The later track wins.
    mid = make_file([
        [cc(7, 10, time=100)],
        [cc(7, 20, time=100)],
    ])

    assert as_tuples(mid.chase(101)) == [('cc', 0, 7, 20)]


def test_same_tick_order_is_stable_regardless_of_track_length():
    # Track 1 reaches the conflicting tick after fewer messages; the
    # tie must still be broken by track index, not by message count.
    mid = make_file([
        [cc(7, 10, time=100), cc(11, 1, time=0)],
        [cc(7, 20, time=100)],
    ])

    assert as_tuples(mid.chase(101)) == [
        ('cc', 0, 7, 20),
        ('cc', 0, 11, 1),
    ]


def test_same_tick_same_track_last_message_wins():
    mid = make_file([[
        cc(7, 10),
        cc(7, 20),
        cc(7, 30),
    ]])

    assert as_tuples(mid.chase(1)) == [('cc', 0, 7, 30)]


def test_cross_track_events_before_target_are_merged_in_tick_order():
    # Track 0 sets CC 7 at tick 200.  Track 1 sets it at tick 100 and
    # again at tick 200 (which is at index 1 of track 1, after track 0
    # at the same tick).
    mid = make_file([
        [cc(7, 10, time=200)],
        [cc(7, 20, time=100), cc(7, 30, time=100)],
    ])

    assert as_tuples(mid.chase(200)) == [('cc', 0, 7, 20)]
    assert as_tuples(mid.chase(201)) == [('cc', 0, 7, 30)]


def test_same_tick_order_matches_merged_track_order():
    # The chase and merge_tracks() must agree on which of several
    # same-tick messages comes last (the winner).
    mid = make_file([
        [cc(7, 10, time=100), cc(7, 11, time=0)],
        [cc(7, 20, time=0), cc(7, 21, time=100)],
        [cc(7, 30, time=50), cc(7, 31, time=50)],
    ])

    accumulated = 0
    winner = None
    for msg in mido.merge_tracks(mid.tracks):
        accumulated += msg.time
        if accumulated == 100 and msg.type == 'control_change':
            winner = msg.value

    assert winner is not None
    assert as_tuples(mid.chase(101)) == [('cc', 0, 7, winner)]


def test_chase_is_repeatable_and_does_not_modify_file():
    mid = make_file([[
        cc(7, 10),
        Message('program_change', program=5, time=100),
    ]])
    original = [(msg.type, msg.time) for msg in mid.tracks[0]]

    first = mid.chase(101)
    second = mid.chase(101)

    assert first == second
    assert [(msg.type, msg.time) for msg in mid.tracks[0]] == original


# ---------------------------------------------------------------------
# RPN and NRPN
# ---------------------------------------------------------------------

def rpn_selection(number, channel=0, time=0):
    return [
        cc(101, (number >> 7) & 0x7F, channel, time),
        cc(100, number & 0x7F, channel, time),
    ]


def nrpn_selection(number, channel=0, time=0):
    return [
        cc(99, (number >> 7) & 0x7F, channel, time),
        cc(98, number & 0x7F, channel, time),
    ]


def test_rpn_selection_and_14_bit_value_are_chased():
    mid = make_file([[
        *rpn_selection(0),  # pitch bend range
        cc(6, 2),
        cc(38, 100),
    ]])

    assert as_tuples(mid.chase(1)) == [
        ('cc', 0, 101, 0),
        ('cc', 0, 100, 0),
        ('cc', 0, 6, 2),
        ('cc', 0, 38, 100),
    ]


def test_rpn_selection_without_data_entry_is_chased():
    mid = make_file([[
        *rpn_selection(1),
    ]])

    assert as_tuples(mid.chase(1)) == [
        ('cc', 0, 101, 0),
        ('cc', 0, 100, 1),
    ]


def test_rpn_7_bit_value_uses_data_entry_msb_only():
    mid = make_file([[
        *rpn_selection(0),
        cc(6, 7),
    ]])

    assert as_tuples(mid.chase(1)) == [
        ('cc', 0, 101, 0),
        ('cc', 0, 100, 0),
        ('cc', 0, 6, 7),
    ]


def test_nrpn_selection_and_value_are_chased():
    mid = make_file([[
        *nrpn_selection(0x1234),
        cc(6, 0x10),
        cc(38, 0x20),
    ]])

    assert as_tuples(mid.chase(1)) == [
        ('cc', 0, 99, 0x24),
        ('cc', 0, 98, 0x34),
        ('cc', 0, 6, 0x10),
        ('cc', 0, 38, 0x20),
    ]


def test_rpn_null_reset_is_chased_and_drops_value():
    mid = make_file([[
        *rpn_selection(0),
        cc(6, 2),
        cc(101, 127),
        cc(100, 127),
    ]])

    assert as_tuples(mid.chase(1)) == [
        ('cc', 0, 101, 127),
        ('cc', 0, 100, 127),
    ]


def test_nrpn_127_127_is_not_a_null_reset():
    mid = make_file([[
        *nrpn_selection(0x3FFF),
        cc(6, 5),
    ]])

    assert as_tuples(mid.chase(1)) == [
        ('cc', 0, 99, 127),
        ('cc', 0, 98, 127),
        ('cc', 0, 6, 5),
    ]


def test_rpn_selection_does_not_leak_between_channels():
    mid = make_file([[
        *rpn_selection(0, channel=0),
        cc(6, 2, channel=0),
        cc(6, 9, channel=1),  # orphan on channel 1
        *rpn_selection(0, channel=2),
        cc(38, 50, channel=2),
    ]])

    assert as_tuples(mid.chase(1)) == [
        ('cc', 0, 101, 0),
        ('cc', 0, 100, 0),
        ('cc', 0, 6, 2),
        ('cc', 2, 101, 0),
        ('cc', 2, 100, 0),
    ]


def test_rpn_selector_halves_do_not_mix_across_channels():
    mid = make_file([[
        cc(101, 0, channel=0),
        cc(100, 1, channel=1),
        cc(6, 9, channel=1),
    ]])

    # Channel 0 only saw the MSB half and channel 1 only the LSB half,
    # so neither has a selection and the data entry is an orphan.
    assert mid.chase(1) == []


def test_rpn_increment_and_decrement():
    mid = make_file([[
        *rpn_selection(0),
        cc(6, 10),
        cc(96, 3),
        cc(97, 1),
    ]])

    assert as_tuples(mid.chase(1)) == [
        ('cc', 0, 101, 0),
        ('cc', 0, 100, 0),
        ('cc', 0, 6, 12),
    ]


def test_rpn_increment_clamps_to_encoding_width():
    mid = make_file([[
        *rpn_selection(0),
        cc(6, 120),
        cc(96, 20),
    ]])

    # 7-bit parameters cannot exceed 127; the width is not promoted.
    assert as_tuples(mid.chase(1)) == [
        ('cc', 0, 101, 0),
        ('cc', 0, 100, 0),
        ('cc', 0, 6, 127),
    ]


def test_rpn_14_bit_increment_clamps_to_14_bits():
    mid = make_file([[
        *rpn_selection(0),
        cc(6, 127),
        cc(38, 127),
        cc(96, 100),
    ]])

    assert as_tuples(mid.chase(1)) == [
        ('cc', 0, 101, 0),
        ('cc', 0, 100, 0),
        ('cc', 0, 6, 127),
        ('cc', 0, 38, 127),
    ]


def test_rpn_increment_without_absolute_value_is_ignored():
    # The increment is relative to a value that was never sent, so the
    # resulting value cannot be known.
    mid = make_file([[
        *rpn_selection(0),
        cc(96, 3),
    ]])

    assert as_tuples(mid.chase(1)) == [
        ('cc', 0, 101, 0),
        ('cc', 0, 100, 0),
    ]


def test_rpn_value_survives_away_from_selection():
    # The value only applies to the selected RPN.  Selecting another
    # parameter must not inherit it.
    mid = make_file([[
        *rpn_selection(0),
        cc(6, 2),
        *rpn_selection(1),
    ]])

    assert as_tuples(mid.chase(1)) == [
        ('cc', 0, 101, 0),
        ('cc', 0, 100, 1),
    ]


def test_orphan_data_entry_messages_are_ignored():
    mid = make_file([[
        cc(6, 10),
        cc(38, 20),
        cc(96, 1),
        cc(97, 1),
        cc(7, 100),
    ]])

    assert as_tuples(mid.chase(1)) == [('cc', 0, 7, 100)]


def test_orphan_data_entry_lsb_after_selection_is_ignored():
    mid = make_file([[
        *rpn_selection(0),
        cc(38, 20),
    ]])

    assert as_tuples(mid.chase(1)) == [
        ('cc', 0, 101, 0),
        ('cc', 0, 100, 0),
    ]


def test_selection_change_resets_pending_data_entry():
    mid = make_file([[
        *rpn_selection(0),
        cc(6, 10),
        *rpn_selection(1),
        cc(38, 20),  # LSB without a new MSB for RPN 1
    ]])

    assert as_tuples(mid.chase(1)) == [
        ('cc', 0, 101, 0),
        ('cc', 0, 100, 1),
    ]


def test_rpn_and_nrpn_are_independent():
    mid = make_file([[
        *rpn_selection(0),
        cc(6, 2),
        *nrpn_selection(5),
        cc(6, 9),
    ]])

    # The most recent complete selection wins, and it is the NRPN.
    assert as_tuples(mid.chase(1)) == [
        ('cc', 0, 99, 0),
        ('cc', 0, 98, 5),
        ('cc', 0, 6, 9),
    ]


# ---------------------------------------------------------------------
# Other controller messages
# ---------------------------------------------------------------------

def test_reset_all_controllers_applies_standard_reset_values():
    mid = make_file([[
        cc(BANK_SELECT_MSB, 1),
        Message('program_change', program=5),
        cc(7, 100),
        cc(1, 30),
        cc(11, 40),
        cc(64, 127),
        cc(20, 99),  # no standard reset value
        Message('pitchwheel', pitch=1000),
        Message('aftertouch', value=50),
        *rpn_selection(0),
        cc(6, 2),
        cc(121, 0),
    ]])

    # Standard reset values are applied, pitch bend is centered and
    # channel pressure zeroed.  Controllers whose reset value is not
    # specified (here 20) keep their values, and so do bank select,
    # program and the RPN selection.
    assert as_tuples(mid.chase(1)) == [
        ('cc', 0, BANK_SELECT_MSB, 1),
        ('program', 0, 5),
        ('cc', 0, 101, 0),
        ('cc', 0, 100, 0),
        ('cc', 0, 6, 2),
        ('cc', 0, 1, 0),
        ('cc', 0, 7, 100),
        ('cc', 0, 11, 127),
        ('cc', 0, 20, 99),
        ('cc', 0, 64, 0),
        ('pitchwheel', 0, 0),
        ('aftertouch', 0, 0),
    ]


def test_reset_all_controllers_does_not_deselect_rpn():
    mid = make_file([[
        *rpn_selection(0),
        cc(6, 2),
        cc(121, 0),
    ]])

    state = mid.chase_state(1)
    assert state[0].selection == ('rpn', 0)
    assert state[0].parameters == {('rpn', 0): (2, 7)}


def test_channel_mode_messages_are_not_chased():
    mid = make_file([[
        cc(120, 0),  # all sound off
        cc(123, 0),  # all notes off
        cc(125, 0),  # omni off
    ]])

    assert mid.chase(1) == []


def test_all_notes_off_is_not_chased_but_controllers_are():
    mid = make_file([[
        cc(7, 90),
        cc(123, 0),
    ]])

    assert as_tuples(mid.chase(1)) == [('cc', 0, 7, 90)]


def test_channels_are_emitted_in_ascending_order():
    mid = make_file([[
        cc(7, 1, channel=5),
        cc(7, 2, channel=0),
        cc(7, 3, channel=9),
    ]])

    assert as_tuples(mid.chase(1)) == [
        ('cc', 0, 7, 2),
        ('cc', 5, 7, 1),
        ('cc', 9, 7, 3),
    ]


def test_pitchwheel_and_aftertouch_are_per_channel():
    mid = make_file([[
        Message('pitchwheel', channel=0, pitch=-8192),
        Message('pitchwheel', channel=1, pitch=8191),
        Message('aftertouch', channel=0, value=0),
        Message('aftertouch', channel=1, value=127),
    ]])

    assert as_tuples(mid.chase(1)) == [
        ('pitchwheel', 0, -8192),
        ('aftertouch', 0, 0),
        ('pitchwheel', 1, 8191),
        ('aftertouch', 1, 127),
    ]


def test_pitchwheel_centered_by_reset():
    mid = make_file([[
        Message('pitchwheel', pitch=500),
        cc(121, 0),
    ]])

    assert as_tuples(mid.chase(1)) == [
        ('cc', 0, 1, 0),
        ('cc', 0, 11, 127),
        ('cc', 0, 64, 0),
        ('pitchwheel', 0, 0),
        ('aftertouch', 0, 0),
    ]


# ---------------------------------------------------------------------
# State object
# ---------------------------------------------------------------------

def test_chase_state_contains_per_channel_state():
    mid = make_file([[
        cc(BANK_SELECT_MSB, 1),
        Message('program_change', program=4),
        cc(7, 100),
    ]])

    state = mid.chase_state(1)

    assert isinstance(state, ChaseState)
    assert state.tick == 1
    assert list(state) == [0]
    assert 0 in state
    assert 1 not in state
    assert len(state) == 1

    channel = state[0]
    assert isinstance(channel, ChannelChaseState)
    assert channel.channel == 0
    assert channel.bank_msb == 1
    assert channel.bank_lsb is None
    assert channel.program == 4
    assert channel.controllers == {7: 100}
    assert channel.pitchwheel is None
    assert channel.channel_pressure is None


def test_chase_state_messages_match_chase():
    mid = make_file([
        [cc(BANK_SELECT_MSB, 1), Message('program_change', program=4)],
        [Message('pitchwheel', channel=1, pitch=-4)],
    ])

    state = mid.chase_state(100)
    assert state.messages() == mid.chase(100)


def test_chase_state_copy_is_independent():
    mid = make_file([[
        cc(7, 100),
    ]])

    state = mid.chase_state(1)
    copy = state.copy()

    assert copy == state
    copy[0].controllers[7] = 1
    assert state[0].controllers == {7: 100}
    assert copy != state


def test_chase_state_unknown_channel_raises_key_error():
    mid = make_file([[
        cc(7, 100),
    ]])

    with pytest.raises(KeyError):
        mid.chase_state(1)[1]


def test_chase_state_repr_mentions_tick():
    mid = make_file([[
        cc(7, 100),
    ]])

    assert 'tick=1' in repr(mid.chase_state(1))


# ---------------------------------------------------------------------
# Two tracks with bank, pitch bend and RPN
# ---------------------------------------------------------------------

def make_two_track_song():
    """A two-track file with a tempo map and a mid-song state.

    Track 0 (conductor/controller track):

      tick 0    set_tempo 500000
      tick 0    channel 0 bank select MSB 0, LSB 1
      tick 0    channel 0 program 10
      tick 0    channel 0 volume 100
      tick 480  channel 0 RPN 0 (pitch bend range) = 12 semitones
      tick 960  channel 0 pitch bend +2000
      tick 1200 channel 0 modulation 64

    Track 1 (notes):

      tick 0    note on 60
      tick 240  note off 60
      tick 480  channel 1 bank select MSB 2, LSB 3
      tick 480  channel 1 program 40
      tick 960  channel 1 RPN 1 (fine tuning) = 14-bit 8192
      tick 960  channel 1 pitch bend -1500
      tick 1440 channel 1 aftertouch 77
      tick 1920 note on 64
    """
    track0 = MidiTrack([
        MetaMessage('set_tempo', tempo=500000, time=0),
        cc(BANK_SELECT_MSB, 0, channel=0, time=0),
        cc(BANK_SELECT_LSB, 1, channel=0, time=0),
        Message('program_change', channel=0, program=10, time=0),
        cc(7, 100, channel=0, time=0),
        cc(101, 0, channel=0, time=480),
        cc(100, 0, channel=0, time=0),
        cc(6, 12, channel=0, time=0),
        Message('pitchwheel', channel=0, pitch=2000, time=960),
        cc(1, 64, channel=0, time=240),
    ])
    track1 = MidiTrack([
        Message('note_on', channel=1, note=60, velocity=100, time=0),
        Message('note_off', channel=1, note=60, time=240),
        cc(BANK_SELECT_MSB, 2, channel=1, time=240),
        cc(BANK_SELECT_LSB, 3, channel=1, time=0),
        Message('program_change', channel=1, program=40, time=0),
        cc(101, 0, channel=1, time=480),
        cc(100, 1, channel=1, time=0),
        cc(6, 64, channel=1, time=0),
        cc(38, 0, channel=1, time=0),
        Message('pitchwheel', channel=1, pitch=-1500, time=0),
        Message('aftertouch', channel=1, value=77, time=480),
        Message('note_on', channel=1, note=64, velocity=80, time=480),
    ])
    return MidiFile(type=1, ticks_per_beat=480,
                    tracks=[track0, track1])


def test_two_track_song_state_before_second_note():
    mid = make_two_track_song()

    state = mid.chase_state(1920)

    assert sorted(state.channels) == [0, 1]

    assert as_tuples(state[0].messages()) == [
        ('cc', 0, BANK_SELECT_MSB, 0),
        ('cc', 0, BANK_SELECT_LSB, 1),
        ('program', 0, 10),
        ('cc', 0, 101, 0),
        ('cc', 0, 100, 0),
        ('cc', 0, 6, 12),
        ('cc', 0, 1, 64),
        ('cc', 0, 7, 100),
        ('pitchwheel', 0, 2000),
    ]
    assert as_tuples(state[1].messages()) == [
        ('cc', 1, BANK_SELECT_MSB, 2),
        ('cc', 1, BANK_SELECT_LSB, 3),
        ('program', 1, 40),
        ('cc', 1, 101, 0),
        ('cc', 1, 100, 1),
        ('cc', 1, 6, 64),
        ('cc', 1, 38, 0),
        ('pitchwheel', 1, -1500),
        ('aftertouch', 1, 77),
    ]

    assert as_tuples(mid.chase(1920)) == (
        as_tuples(state[0].messages()) + as_tuples(state[1].messages()))


def test_two_track_song_before_channel_1_bank_select():
    mid = make_two_track_song()

    # At tick 480 channel 1 has not been set up yet, and channel 0's
    # RPN selection and value are exactly at the target so they are
    # not included.
    state = mid.chase_state(480)

    assert sorted(state.channels) == [0]
    assert state[0].bank_msb == 0
    assert state[0].bank_lsb == 1
    assert state[0].program == 10
    assert state[0].controllers == {7: 100}
    assert state[0].selection is None
    assert state[0].pitchwheel is None

    assert as_tuples(mid.chase(480)) == [
        ('cc', 0, BANK_SELECT_MSB, 0),
        ('cc', 0, BANK_SELECT_LSB, 1),
        ('program', 0, 10),
        ('cc', 0, 7, 100),
    ]


def test_two_track_song_just_after_rpn_and_pitch_bend():
    mid = make_two_track_song()

    state = mid.chase_state(1441)

    assert state[0].selection == ('rpn', 0)
    assert state[0].parameters == {('rpn', 0): (12, 7)}
    assert state[0].pitchwheel == 2000
    assert state[0].controllers == {7: 100}
    assert state[1].selection == ('rpn', 1)
    assert state[1].parameters == {('rpn', 1): (8192, 14)}
    assert state[1].pitchwheel == -1500
    assert state[1].channel_pressure == 77


def test_two_track_song_note_not_retriggered_at_target():
    mid = make_two_track_song()

    messages = mid.chase(1920)

    assert all(msg.type not in ('note_on', 'note_off') for msg in messages)
    assert as_tuples(messages) == as_tuples(
        mid.chase_state(1920)[0].messages()
        + mid.chase_state(1920)[1].messages())


def test_two_track_song_matches_original_notes_and_meta():
    mid = make_two_track_song()

    # The chase must not alter or consume the original file.
    assert mid.tracks[0][0] == MetaMessage('set_tempo', tempo=500000)
    assert mid.tracks[1][0].type == 'note_on'
    assert mid.length == pytest.approx(sum(msg.time for msg in mid))


def test_two_track_song_roundtrips_through_save_and_load():
    mid = make_two_track_song()
    buffer = io.BytesIO()
    mid.save(file=buffer)
    buffer.seek(0)
    reloaded = MidiFile(file=buffer)

    assert reloaded.chase(1920) == mid.chase(1920)


# ---------------------------------------------------------------------
# A two-track file read from raw bytes
# ---------------------------------------------------------------------

# ticks_per_beat = 480, one tempo message at the start.
#
# Track 0:
#   tick 0    set_tempo 500000
#             channel 0 bank select 0=1, 32=2, program 10
#             channel 0 volume 100
#   tick 480  channel 0 RPN 0 = 12 (7-bit data entry)
#   tick 960  channel 0 pitch bend +2000
#   tick 1200 channel 0 modulation 64
#
# Track 1:
#   tick 0    channel 1 note on 60
#   tick 240  channel 1 note off 60
#             channel 1 bank select 0=2, 32=3, program 40
#   tick 720  channel 1 RPN 1 = 14-bit 8192
#             channel 1 pitch bend -1500
#   tick 1200 channel 1 aftertouch 77
#   tick 1680 channel 1 note on 64
TWO_TRACK_HEXDUMP = """
4d 54 68 64 00 00 00 06 00 01 00 02 01 e0
4d 54 72 6b 00 00 00 31
00 ff 51 03 07 a1 20
00 b0 00 01
00 b0 20 02
00 c0 0a
00 b0 07 64
83 60 b0 65 00
00 b0 64 00
00 b0 06 0c
83 60 e0 50 4f
81 70 b0 01 40
00 ff 2f 00
4d 54 72 6b 00 00 00 36
00 91 3c 64
81 70 81 3c 00
00 b1 00 02
00 b1 20 03
00 c1 28
83 60 b1 65 00
00 b1 64 01
00 b1 06 40
00 b1 26 00
00 e1 24 34
83 60 d1 4d
83 60 91 40 50
00 ff 2f 00
"""


def read_raw_file(hexdump):
    data = bytearray()
    for line in hexdump.splitlines():
        data += bytearray.fromhex(line)
    return MidiFile(file=io.BytesIO(data))


def test_raw_two_track_file_parses_and_mid_song_state():
    mid = read_raw_file(TWO_TRACK_HEXDUMP)

    assert mid.type == 1
    assert mid.ticks_per_beat == 480
    assert len(mid.tracks) == 2
    # The note that starts at tick 1680 must not be retriggered when
    # chasing tick 1680.
    assert mid.tracks[1][-2] == Message(
        'note_on', channel=1, note=64, velocity=80, time=480)

    state = mid.chase_state(1680)

    assert sorted(state.channels) == [0, 1]

    assert as_tuples(state[0].messages()) == [
        ('cc', 0, BANK_SELECT_MSB, 1),
        ('cc', 0, BANK_SELECT_LSB, 2),
        ('program', 0, 10),
        ('cc', 0, 101, 0),
        ('cc', 0, 100, 0),
        ('cc', 0, 6, 12),
        ('cc', 0, 1, 64),
        ('cc', 0, 7, 100),
        ('pitchwheel', 0, 2000),
    ]
    assert as_tuples(state[1].messages()) == [
        ('cc', 1, BANK_SELECT_MSB, 2),
        ('cc', 1, BANK_SELECT_LSB, 3),
        ('program', 1, 40),
        ('cc', 1, 101, 0),
        ('cc', 1, 100, 1),
        ('cc', 1, 6, 64),
        ('cc', 1, 38, 0),
        ('pitchwheel', 1, -1500),
        ('aftertouch', 1, 77),
    ]

    assert as_tuples(mid.chase(1680)) == (
        as_tuples(state[0].messages()) + as_tuples(state[1].messages()))


def test_raw_two_track_file_before_notes_are_chased():
    mid = read_raw_file(TWO_TRACK_HEXDUMP)

    # At tick 720 channel 1's RPN selection and value are exactly at
    # the target, so they are left for playback; channel 1 still has
    # its bank/program from tick 240 and channel 0 has everything from
    # before tick 720.
    state = mid.chase_state(720)

    assert state[0].bank_msb == 1
    assert state[0].bank_lsb == 2
    assert state[0].program == 10
    assert state[0].selection == ('rpn', 0)
    assert state[0].parameters == {('rpn', 0): (12, 7)}
    assert state[0].pitchwheel is None

    assert state[1].bank_msb == 2
    assert state[1].bank_lsb == 3
    assert state[1].program == 40
    assert state[1].selection is None
    assert state[1].parameters == {}
    assert state[1].pitchwheel is None
    assert state[1].channel_pressure is None


def test_raw_two_track_file_after_all_state():
    mid = read_raw_file(TWO_TRACK_HEXDUMP)

    state = mid.chase_state(1680)

    assert state[0].selection == ('rpn', 0)
    assert state[0].parameters == {('rpn', 0): (12, 7)}
    assert state[0].controllers == {1: 64, 7: 100}
    assert state[0].pitchwheel == 2000

    assert state[1].selection == ('rpn', 1)
    assert state[1].parameters == {('rpn', 1): (8192, 14)}
    assert state[1].pitchwheel == -1500
    assert state[1].channel_pressure == 77


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def test_chase_is_exposed_at_package_level():
    assert mido.chase_events is chase_events
    assert mido.chase_state is chase_state
    assert mido.ChaseState is ChaseState
    assert mido.ChannelChaseState is ChannelChaseState


def test_module_level_chase_matches_method():
    mid = make_two_track_song()

    assert chase_events(mid, 1920) == mid.chase(1920)


def test_chase_ignores_sysex():
    mid = make_file([[
        Message('sysex', data=[1, 2, 3]),
        cc(7, 10),
    ]])

    assert as_tuples(mid.chase(1)) == [('cc', 0, 7, 10)]
