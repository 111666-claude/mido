.. SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
..
.. SPDX-License-Identifier: CC-BY-4.0

Routing
-------


About
^^^^^

A :class:`~mido.routing.Router` is an output :term:`port` that forwards
each message to zero or more other output ports according to a list of
rules. This is useful for a stage setup where a drum machine and a
synthesizer share one MIDI input: note messages go to both devices,
each on a different channel, while SysEx and clock are broadcast.

The router never touches the message that was passed to ``send()``, so
it is safe to feed the same message to other code after routing it.
It is an ordinary output port and can be mixed freely with
``MultiPort`` and the rest of Mido.


A Basic Router
^^^^^^^^^^^^^^

::

    from mido.routing import Router, RoutingRule

    router = Router([
        # Channel 9 is the drum machine. It gets its own copy.
        RoutingRule([drums], channels=9),

        # Notes on channels 0 and 1 are moved to channel 2 for the synth.
        RoutingRule([synth], types=['note_on', 'note_off'],
                    channels=[0, 1], remap=2),
    ])

    for message in inport:
        router.send(message)

A rule is a ``RoutingRule`` with these arguments:

``outputs``
    One output port, or an iterable of output ports. All of them
    receive the message, in the order given.

``types``
    A message type such as ``'note_on'``, a group name, or an iterable
    of either. The groups are ``'channel'``, ``'common'``,
    ``'realtime'``, ``'sysex'``, ``'system'`` and ``'all'``.
    **If omitted, the rule matches channel messages only.**

``channels``
    An integer or an iterable of integers in ``0..15``.

``remap``
    A channel number, or a callable that receives the current channel
    and returns the new one (or ``None`` to leave it unchanged).

``priority``
    A number. The default is ``0``.

``stop``
    Stop evaluating rules after this one has run.

``match``
    An optional predicate taking a message and returning a boolean.


SysEx and System Messages
^^^^^^^^^^^^^^^^^^^^^^^^^

Because a rule without ``types`` matches channel messages only, SysEx
and system messages never fall into a channel rule by accident. They
take an *explicit* default path instead::

    router = Router(
        [RoutingRule([drums], channels=9),
         RoutingRule([synth], channels=[0, 1])],
        default=[monitor],
        sysex=[patch_editor],
        realtime=[clock_light],
    )

The lookup order for a message is:

1. Every matching rule, in evaluation order.
2. If no rule matched: the ``sysex`` or ``realtime`` path for SysEx
   and system realtime messages, falling back to ``default``.
3. Otherwise the ``default`` path.

A rule can still match SySex or system messages directly by naming the
type or group in ``types``; such a rule runs before any default path.

Setting ``sysex=[]`` (or ``default=[]``) is not the same as leaving it
as ``None``: the empty path is an explicit "route nowhere" and
suppresses the fallback. A message that no rule and no path matched is
silently dropped by default; pass ``on_unmatched='raise'`` to raise
``UnmatchedMessageError`` instead.


Ordering
^^^^^^^^

Rules run in descending ``priority`` order. Rules with equal priority
run in the order they were declared, so the default is plain
declaration order. The ``rules`` attribute always returns the rules in
declaration order; ``ordered_rules`` returns the effective order.

Every matching rule is applied, unless a rule has ``stop=True``. A
rule that matched but has no outputs still counts as a match and
prevents the default path from being used.


Replacing Rules at Run Time
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``set_rules()`` (and the ``rules`` property) replaces the whole rule
list atomically. A message that is being routed while the rules change
is processed completely with either the old or the new set, never with
a mixture. The same holds for concurrent calls to ``send()`` from
several threads: each message sees one consistent snapshot.

    router.set_rules([
        RoutingRule([synth_a], channels=[0]),
        RoutingRule([synth_b], channels=[1]),
    ])


Closed Ports, Failures and Loops
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* A **closed** output port is silently skipped by default. Pass
  ``on_closed='raise'`` to get ``ClosedOutputError`` instead.
* If an output port raises while sending, the router raises
  ``RoutingSendError`` (the original exception is available as
  ``.original``). Ports that had already received the message keep it;
  ports later in the list do not get it. Pass ``on_error='skip'`` to
  log and continue instead. A router that has itself been closed
  raises ``ValueError`` from ``send()``, like every output port.
* A **cycle** in the router graph raises ``RoutingLoopError``. Direct
  router-to-router cycles are rejected when the rules are installed and
  leave the old rules in place. Cycles hidden behind custom wrapper
  ports are detected while routing; in either case the message is not
  delivered.
* Closing a router does **not** close the ports it forwards to, unless
  it was created with ``close_outputs=True``.


API
^^^

.. autoclass:: mido.routing.Router
   :members:

.. autoclass:: mido.routing.RoutingRule
   :members:

.. autoexception:: mido.routing.RoutingError

.. autoexception:: mido.routing.RoutingLoopError

.. autoexception:: mido.routing.RoutingSendError

.. autoexception:: mido.routing.ClosedOutputError

.. autoexception:: mido.routing.UnmatchedMessageError
