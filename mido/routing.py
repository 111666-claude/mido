# SPDX-FileCopyrightText: 2026 Ole Martin Bjorndalen <ombdalen@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Rule based routing between Mido output ports.

A :class:`Router` is an ordinary output port. Messages sent to it are
forwarded to zero or more other output ports according to a list of
:class:`RoutingRule` objects. This makes it possible to split one
incoming stream between several devices::

    from mido.routing import Router, RoutingRule

    router = Router([
        RoutingRule([drums], channels=9),
        RoutingRule([synth_a], channels=[0, 1], remap=2),
        RoutingRule([synth_b], types='note_on', channels=[2, 3]),
    ], default=[monitor])

    for message in inport:
        router.send(message)

Rules are evaluated in declaration order, unless an explicit
``priority`` moves a rule ahead of the rules declared before it. Every
matching rule is applied, so a message can fan out to several
destinations. A rule with ``stop=True`` ends rule processing after it
has run.

Rules without a ``types`` argument match *channel* messages only. This
means that SysEx and system messages are never swept into a channel
rule by accident. They take the explicit default path configured with
the ``sysex``, ``realtime`` and ``default`` arguments instead, unless a
rule names their type explicitly.

The router never modifies the message passed to :meth:`Router.send`.
A rule that remaps a channel sends a modified copy; the original
object is left untouched.

Replacing the rules with :meth:`Router.set_rules` (or the ``rules``
property) is atomic. Every message is processed completely with either
the old or the new set of rules, never with a mixture of the two.
"""

import math
import threading
from collections import namedtuple
from numbers import Integral, Real

from .messages.specs import SPEC_BY_TYPE
from .ports import BaseOutput

__all__ = [
    "ALL_TYPES",
    "CHANNEL_TYPES",
    "ClosedOutputError",
    "REALTIME_TYPES",
    "RoutingError",
    "RoutingLoopError",
    "RoutingRule",
    "RoutingSendError",
    "Router",
    "SYSTEM_TYPES",
    "UnmatchedMessageError",
]


# ---------------------------------------------------------------
# Message type groups.
#
# Routing classifies messages by their MIDI status byte rather than by
# Message.is_realtime. (The latter has a few historical quirks, for
# example tune_request is a system common message but is listed as
# realtime.)
# ---------------------------------------------------------------

ALL_TYPES = frozenset(SPEC_BY_TYPE)
CHANNEL_TYPES = frozenset(
    type_ for type_, spec in SPEC_BY_TYPE.items()
    if 0x80 <= spec['status_byte'] <= 0xef)
SYSTEM_TYPES = frozenset(
    type_ for type_, spec in SPEC_BY_TYPE.items()
    if spec['status_byte'] >= 0xf0)
REALTIME_TYPES = frozenset(
    type_ for type_, spec in SPEC_BY_TYPE.items()
    if spec['status_byte'] >= 0xf8)
_COMMON_TYPES = SYSTEM_TYPES - REALTIME_TYPES
_SYSEX_TYPE = 'sysex'

_TYPE_ALIASES = {
    'all': ALL_TYPES,
    'channel': CHANNEL_TYPES,
    'common': _COMMON_TYPES,
    'realtime': REALTIME_TYPES,
    'system': SYSTEM_TYPES,
    'sysex': frozenset({_SYSEX_TYPE}),
}

# Give up on pathologically deep (but acyclic) router chains. A real
# cycle is detected by identity long before this limit is reached.
_MAX_HOPS = 128

_ON_UNMATCHED_VALUES = ('drop', 'raise')
_ON_ERROR_VALUES = ('raise', 'skip')
_ON_CLOSED_VALUES = ('skip', 'raise')

# Serializes graph validation and rule installation so two routers
# can not be wired into a cycle concurrently.
_graph_lock = threading.RLock()

_thread_local = threading.local()


class RoutingError(Exception):
    """Base class for routing errors."""


class RoutingLoopError(RoutingError):
    """Raised for a cyclic router graph.

    The message is not delivered at all when this is raised. Eager
    checks catch direct router-to-router cycles when the rules are
    installed. Cycles hidden behind wrappers are caught when the
    message reaches a router that is already busy routing it.
    """


class UnmatchedMessageError(RoutingError):
    """Raised when no rule and no default path matched a message.

    Only raised when the router was created with
    ``on_unmatched='raise'``.
    """


class ClosedOutputError(RoutingError):
    """Raised when a message targets a closed port.

    Only raised when the router was created with ``on_closed='raise'``.
    With the default (``'skip'``) closed ports are silently skipped.
    """


class RoutingSendError(RoutingError):
    """Raised when an output port rejected a message.

    Only raised when the router was created with ``on_error='raise'``
    (the default). ``port`` is the port that failed, ``message`` is the
    message that was being sent and ``original`` is the exception that
    the port raised.
    """

    def __init__(self, port, message, original):
        self.port = port
        self.message = message
        self.original = original
        super().__init__(
            f'failed to send {message!r} to {port!r}: {original!r}')


class _ThreadState:
    """Routers currently processing a message on this thread."""

    __slots__ = ('stack', 'active')

    def __init__(self):
        self.stack = []
        self.active = set()


def _thread_state():
    state = getattr(_thread_local, 'state', None)
    if state is None:
        state = _ThreadState()
        _thread_local.state = state
    return state


def _normalize_ports(value, argument):
    """Return an output path as a tuple, or None if not configured.

    A single port is accepted as a convenience. An empty iterable is
    kept as an empty tuple, which means "explicitly route nowhere" and
    is different from None ("not configured").
    """
    if value is None:
        return None

    if hasattr(value, 'send'):
        ports = (value,)
    else:
        try:
            ports = tuple(value)
        except TypeError:
            raise TypeError(
                f'{argument} must be a port or an iterable of ports')

    for port in ports:
        if not hasattr(port, 'send'):
            raise TypeError(f'{argument} contains a non-port object: '
                            f'{port!r}')

    return ports


def _normalize_types(types):
    """Turn types into a frozenset of message type names."""
    if types is None:
        return CHANNEL_TYPES

    if isinstance(types, str):
        values = (types,)
    else:
        try:
            values = tuple(types)
        except TypeError:
            raise TypeError(
                'types must be a message type, a group name '
                'or an iterable of them')

    if not values:
        raise ValueError('types must not be empty')

    result = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f'message type must be a string: {value!r}')

        if value in _TYPE_ALIASES:
            result.update(_TYPE_ALIASES[value])
        elif value in SPEC_BY_TYPE:
            result.add(value)
        else:
            raise ValueError(f'unknown message type or group {value!r}')

    return frozenset(result)


def _normalize_channels(channels):
    """Turn channels into a frozenset of integers in range 0..15."""
    if channels is None:
        return None

    if (isinstance(channels, Integral)
            and not isinstance(channels, bool)):
        values = (channels,)
    else:
        try:
            values = tuple(channels)
        except TypeError:
            raise TypeError(
                'channels must be an integer or an iterable of integers')

    if not values:
        raise ValueError('channels must not be empty')

    for channel in values:
        if (not isinstance(channel, Integral)
                or isinstance(channel, bool)
                or not 0 <= channel <= 15):
            raise ValueError(
                f'channel must be an integer in range 0..15: {channel!r}')

    return frozenset(values)


def _normalize_remap(remap):
    if remap is None:
        return None

    if callable(remap):
        return remap

    if (not isinstance(remap, Integral)
            or isinstance(remap, bool)
            or not 0 <= remap <= 15):
        raise ValueError(
            f'remap must be an integer in range 0..15 or a callable: '
            f'{remap!r}')

    return int(remap)


def _is_channel_type(type_):
    spec = SPEC_BY_TYPE.get(type_)
    return bool(spec and 0x80 <= spec['status_byte'] <= 0xef)


class RoutingRule(namedtuple(
        'RoutingRule',
        'outputs types channels remap priority stop match')):
    """One routing rule.

    outputs
      An output port, or an iterable of output ports. Ports are
      addressed in the given order.

    types
      A message type name, a group name, or an iterable of either.
      Group names are ``'channel'``, ``'common'``, ``'realtime'``,
      ``'sysex'``, ``'system'`` and ``'all'``. If omitted the rule
      matches channel messages only, so SysEx and system messages are
      not matched by accident.

    channels
      An integer or an iterable of integers in range 0..15. If
      omitted, messages on all channels match. A rule with ``channels``
      set can only match when at least one of its ``types`` is a
      channel message type.

    remap
      A channel number in range 0..15, or a callable that takes the
      current channel and returns a new one (or None to leave it
      unchanged). Remapping never changes the message that was passed
      to the router; the selected ports receive a copy. Non-channel
      messages are not modified.

    priority
      A finite number. Rules run in descending priority order. Rules
      with equal priority run in declaration order. The default is 0.

    stop
      If true, no later rule is evaluated after this one has run,
      whether or not it delivered anything.

    match
      An optional callable taking a message and returning a boolean.
      It is combined (AND) with ``types`` and ``channels``. Message
      objects are immutable, so the callable can not modify them.
    """

    __slots__ = ()

    def __new__(cls, outputs, types=None, channels=None, remap=None,
                priority=0, stop=False, match=None):
        outputs = _normalize_ports(outputs, 'outputs')
        types = _normalize_types(types)
        channels = _normalize_channels(channels)
        remap = _normalize_remap(remap)

        if channels is not None and not types & CHANNEL_TYPES:
            raise ValueError(
                'channels was given but types contains no '
                'channel message type')

        if remap is not None and not types & CHANNEL_TYPES:
            raise ValueError(
                'remap was given but types contains no '
                'channel message type')

        if (not isinstance(priority, Real)
                or isinstance(priority, bool)
                or not math.isfinite(priority)):
            raise ValueError(
                f'priority must be a finite number: {priority!r}')

        if not isinstance(stop, bool):
            raise TypeError(f'stop must be a boolean: {stop!r}')

        if match is not None and not callable(match):
            raise TypeError(f'match must be callable: {match!r}')

        return super().__new__(
            cls, outputs, types, channels, remap, priority, stop, match)

    def matches(self, message):
        """Return True if this rule applies to message."""
        if message.type not in self.types:
            return False

        if self.channels is not None:
            channel = getattr(message, 'channel', None)
            if channel is None or channel not in self.channels:
                return False

        if self.match is not None and not self.match(message):
            return False

        return True

    def remap_channel(self, message):
        """Return the channel message should use, or None for unchanged."""
        if self.remap is None:
            return None

        channel = getattr(message, 'channel', None)
        if channel is None:
            # Not a channel message. Remapping does not apply.
            return None

        if callable(self.remap):
            new_channel = self.remap(channel)
            if new_channel is None:
                return None
        else:
            new_channel = self.remap

        if (not isinstance(new_channel, Integral)
                or isinstance(new_channel, bool)
                or not 0 <= new_channel <= 15):
            raise RoutingError(
                f'remap returned an invalid channel {new_channel!r} '
                f'for {message!r}')

        return int(new_channel)

    def apply(self, message):
        """Return message, or a channel-remapped copy of it."""
        channel = self.remap_channel(message)
        if channel is None or channel == message.channel:
            return message
        return message.copy(channel=channel)

    def __repr__(self):
        args = [repr(self.outputs)]
        if self.types != CHANNEL_TYPES:
            args.append(f'types={self.types!r}')
        if self.channels is not None:
            args.append(f'channels={self.channels!r}')
        if self.remap is not None:
            args.append(f'remap={self.remap!r}')
        if self.priority != 0:
            args.append(f'priority={self.priority!r}')
        if self.stop:
            args.append('stop=True')
        if self.match is not None:
            args.append(f'match={self.match!r}')
        return 'RoutingRule({})'.format(', '.join(args))


# An immutable snapshot of everything a Router needs to route one
# message. Replacing this object with set_rules() is atomic.
_Plan = namedtuple('_Plan', 'declared ordered default sysex realtime')

_EMPTY_PLAN = _Plan((), (), None, None, None)


def _iter_plan_outputs(plan):
    for rule in plan.ordered:
        for port in rule.outputs:
            yield port
    for path in (plan.sysex, plan.realtime, plan.default):
        if path is not None:
            for port in path:
                yield port


def _plan_outputs(plan):
    """Return unique downstream ports in routing order."""
    seen = set()
    result = []
    for port in _iter_plan_outputs(plan):
        if id(port) not in seen:
            seen.add(id(port))
            result.append(port)
    return tuple(result)


def _check_cycles(root, plan):
    """Raise RoutingLoopError if plan would close a router cycle.

    Only direct Router-to-Router links can be checked here. Cycles
    hidden inside wrapper ports are caught at run time.
    """
    visiting = set()
    visited = set()
    path = []

    def visit(router, router_plan):
        router_id = id(router)
        if router_id in visiting:
            start = next(
                (index for index, item in enumerate(path)
                 if id(item) == router_id), 0)
            names = [item.name for item in path[start:]] + [router.name]
            raise RoutingLoopError(
                'router cycle detected: {}'.format(' -> '.join(names)))

        if router_id in visited:
            return

        if len(path) >= _MAX_HOPS:
            raise RoutingLoopError(
                f'router chain is longer than {_MAX_HOPS} hops; '
                'this usually means there is a cycle')

        visiting.add(router_id)
        path.append(router)
        try:
            for port in _plan_outputs(router_plan):
                if isinstance(port, Router):
                    if port is root:
                        port_plan = plan
                    else:
                        port_plan = port._plan_snapshot()
                    visit(port, port_plan)
        finally:
            path.pop()
            visiting.remove(router_id)
            visited.add(router_id)

    visit(root, plan)


class Router(BaseOutput):
    """An output port that forwards messages according to rules.

    Router(rules=(), default=None, sysex=None, realtime=None,
           on_unmatched='drop', on_error='raise', on_closed='skip',
           close_outputs=False, name=None)

    rules
      An iterable of :class:`RoutingRule` objects.

    default
      Output port(s) for messages that no rule matched. ``None`` means
      no default path was configured.

    sysex
      Output port(s) for SysEx messages that no rule matched. Falls
      back to ``default`` when not configured.

    realtime
      Output port(s) for system realtime messages that no rule
      matched. Falls back to ``default`` when not configured.

    on_unmatched
      ``'drop'`` (the default) silently discards a message that
      neither a rule nor a default path matched. ``'raise'`` raises
      :class:`UnmatchedMessageError`.

    on_error
      ``'raise'`` (the default) wraps the first failed delivery in
      :class:`RoutingSendError`. Ports that had already received the
      message keep it, while later ports do not get it. ``'skip'``
      ignores a failed delivery and continues with the remaining
      ports.

    on_closed
      ``'skip'`` (the default) silently ignores closed output ports.
      ``'raise'`` raises :class:`ClosedOutputError`. A closed router
      itself always raises ``ValueError`` from ``send()``, like any
      other Mido output port.

    close_outputs
      If true, closing the router also closes its downstream ports.
      The default is false: the router does not own the ports it
      forwards to.

    name
      Port name. Defaults to ``'router'``.

    Rules are ordered by descending ``priority`` and then by
    declaration order. The ``rules`` attribute returns them in
    declaration order, while ``ordered_rules`` returns the effective
    order.
    """

    _device_type = 'router'

    # Routing state is protected by the plan snapshot and the
    # thread-local cycle guard, not by a per-port lock. Taking the
    # BaseOutput lock here would deadlock a router cycle hidden behind
    # a wrapper port before the cycle guard could report it.
    _locking = False

    def __init__(self, rules=(), default=None, sysex=None, realtime=None,
                 on_unmatched='drop', on_error='raise', on_closed='skip',
                 close_outputs=False, name=None):
        # Initialize everything __del__()/close() may touch before any
        # argument validation can raise, so a rejected constructor
        # argument does not leave a half-initialized port behind.
        self._plan_lock = threading.Lock()
        self._plan = _EMPTY_PLAN
        self._close_outputs = False
        BaseOutput.__init__(self, name or 'router')

        if on_unmatched not in _ON_UNMATCHED_VALUES:
            raise ValueError(
                f'on_unmatched must be one of {_ON_UNMATCHED_VALUES}: '
                f'{on_unmatched!r}')
        if on_error not in _ON_ERROR_VALUES:
            raise ValueError(
                f'on_error must be one of {_ON_ERROR_VALUES}: {on_error!r}')
        if on_closed not in _ON_CLOSED_VALUES:
            raise ValueError(
                f'on_closed must be one of {_ON_CLOSED_VALUES}: '
                f'{on_closed!r}')

        self._default_ports = _normalize_ports(default, 'default')
        self._sysex_ports = _normalize_ports(sysex, 'sysex')
        self._realtime_ports = _normalize_ports(realtime, 'realtime')
        self._on_unmatched = on_unmatched
        self._on_error = on_error
        self._on_closed = on_closed
        self._close_outputs = bool(close_outputs)

        self.set_rules(rules)

    # -- Configuration -------------------------------------------------

    @property
    def on_unmatched(self):
        """What to do with a message that no rule or path matched."""
        return self._on_unmatched

    @property
    def on_error(self):
        """What to do when an output port raises."""
        return self._on_error

    @property
    def on_closed(self):
        """What to do when an output port is closed."""
        return self._on_closed

    @property
    def default(self):
        """Configured default path, or None."""
        return self._default_ports

    @property
    def sysex(self):
        """Configured SysEx path, or None."""
        return self._sysex_ports

    @property
    def realtime(self):
        """Configured realtime path, or None."""
        return self._realtime_ports

    # -- Rules ---------------------------------------------------------

    def _plan_snapshot(self):
        with self._plan_lock:
            return self._plan

    @property
    def rules(self):
        """The rules in declaration order."""
        return self._plan_snapshot().declared

    @rules.setter
    def rules(self, rules):
        self.set_rules(rules)

    @property
    def ordered_rules(self):
        """The rules in the order they are evaluated."""
        return self._plan_snapshot().ordered

    @property
    def outputs(self):
        """Unique downstream ports, in routing order."""
        return _plan_outputs(self._plan_snapshot())

    def set_rules(self, rules):
        """Replace all rules atomically.

        Messages that are being routed when this is called finish with
        the old set of rules. Messages routed afterwards use the new
        set. If the new rules would create a direct router cycle, the
        rules are rejected with :class:`RoutingLoopError` and the old
        rules stay in place.
        """
        if rules is None:
            rules = ()
        elif isinstance(rules, RoutingRule):
            rules = (rules,)

        try:
            declared = tuple(rules)
        except TypeError:
            raise TypeError('rules must be an iterable of RoutingRule')

        for rule in declared:
            if not isinstance(rule, RoutingRule):
                raise TypeError(
                    f'rules must be RoutingRule objects: {rule!r}')

        # Stable sort: equal priorities keep declaration order.
        ordered = tuple(sorted(declared, key=lambda rule: -rule.priority))

        plan = _Plan(declared, ordered,
                     self._default_ports,
                     self._sysex_ports,
                     self._realtime_ports)

        with _graph_lock:
            _check_cycles(self, plan)
            with self._plan_lock:
                self._plan = plan

    def add_rule(self, rule):
        """Add one rule, preserving the existing ones."""
        if not isinstance(rule, RoutingRule):
            raise TypeError(f'rule must be a RoutingRule: {rule!r}')
        self.set_rules(self.rules + (rule,))

    def clear_rules(self):
        """Remove all rules."""
        self.set_rules(())

    # -- Routing -------------------------------------------------------

    def _route(self, message):
        plan = self._plan_snapshot()

        matched = False
        for rule in plan.ordered:
            if rule.matches(message):
                matched = True
                self._deliver(rule.outputs, rule.apply(message))
                if rule.stop:
                    break

        if matched:
            return

        if message.type == _SYSEX_TYPE:
            path = plan.sysex
            if path is None:
                path = plan.default
        elif message.type in REALTIME_TYPES:
            path = plan.realtime
            if path is None:
                path = plan.default
        else:
            path = plan.default

        if path is not None:
            self._deliver(path, message)
        elif self._on_unmatched == 'raise':
            raise UnmatchedMessageError(
                f'no routing rule or default path matched {message!r}')

    def _deliver(self, ports, message):
        for port in ports:
            if getattr(port, 'closed', False):
                if self._on_closed == 'raise':
                    raise ClosedOutputError(
                        f'output port is closed: {port!r}')
                continue

            try:
                port.send(message)
            except (RoutingLoopError, UnmatchedMessageError,
                    ClosedOutputError):
                # Structural problems always propagate; on_error is
                # meant for ordinary send failures.
                raise
            except Exception as exc:
                if self._on_error == 'skip':
                    continue
                if isinstance(exc, RoutingSendError):
                    # A nested router already reported the failure.
                    raise
                raise RoutingSendError(port, message, exc) from exc

    def _send(self, message):
        state = _thread_state()

        if id(self) in state.active:
            names = [router.name for router in state.stack] + [self.name]
            raise RoutingLoopError(
                'router cycle detected while routing: {}'.format(
                    ' -> '.join(names)))

        if len(state.stack) >= _MAX_HOPS:
            raise RoutingLoopError(
                f'router chain is longer than {_MAX_HOPS} hops; '
                'this usually means there is a cycle')

        state.stack.append(self)
        state.active.add(id(self))
        try:
            self._route(message)
        finally:
            state.stack.pop()
            state.active.remove(id(self))

    # -- Port lifecycle ------------------------------------------------

    def _close(self):
        if not self._close_outputs:
            return

        # Best effort: try every port even if one of them fails.
        for port in self.outputs:
            try:
                port.close()
            except Exception:
                pass
