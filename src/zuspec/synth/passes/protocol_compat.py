"""Protocol compatibility checker for IfProtocol bind() points.

Validates that ``IfProtocolProperties`` are compatible across a connection
between a port (requester side) and an export (provider side), and validates
each set of properties in isolation.

Errors are raised as ``ProtocolCompatError``.
Warnings are issued via Python's ``warnings.warn`` mechanism.
"""
from __future__ import annotations

import logging
import warnings
from typing import Optional

_log = logging.getLogger(__name__)


class ProtocolCompatError(Exception):
    """Raised when two IfProtocol property sets are incompatible."""


def _props_from(obj) -> "IfProtocolProperties":  # type: ignore[name-defined]
    """Accept either an ``IfProtocolProperties`` instance or an ``IfProtocolType``."""
    from zuspec.ir.core.data_type import IfProtocolProperties, IfProtocolType
    if isinstance(obj, IfProtocolType):
        if obj.properties is None:
            return IfProtocolProperties()
        return obj.properties
    return obj


class ProtocolCompatChecker:
    """Checks ``IfProtocol`` properties are compatible at every bind() point.

    Usage::

        checker = ProtocolCompatChecker()
        checker.check(port_props, export_props, location="MyComp.icache")

    ``port_props`` and ``export_props`` may be ``IfProtocolProperties``
    instances or ``IfProtocolType`` instances (properties are extracted
    automatically).

    ``location`` is a human-readable string used in error / warning messages
    (e.g. the component and field name).
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, props, location: str) -> None:
        """Validate a single set of ``IfProtocolProperties`` in isolation.

        Raises ``ProtocolCompatError`` for hard errors; emits
        ``UserWarning`` for soft warnings.
        """
        p = _props_from(props)
        self._validate_standalone(p, location)

    def check(self, port_props, export_props, location: str) -> None:
        """Check compatibility between *port_props* (requester) and
        *export_props* (provider) at the named connection point.

        Validates each side independently first, then checks cross-side
        compatibility.
        """
        pp = _props_from(port_props)
        ep = _props_from(export_props)

        self._validate_standalone(pp, f"{location}[port]")
        self._validate_standalone(ep, f"{location}[export]")

        self._check_outstanding(pp, ep, location)
        self._check_ordering(pp, ep, location)
        self._check_request_channel(pp, ep, location)
        self._check_response_channel(pp, ep, location)

    # ------------------------------------------------------------------
    # Intra-protocol validation (§6.2)
    # ------------------------------------------------------------------

    def _validate_standalone(self, p, location: str) -> None:
        """Validate a single IfProtocolProperties object."""
        if p.max_outstanding < 1:
            raise ProtocolCompatError(
                f"{location}: max_outstanding must be >= 1, got {p.max_outstanding}"
            )
        if p.initiation_interval < 1:
            raise ProtocolCompatError(
                f"{location}: initiation_interval must be >= 1, got {p.initiation_interval}"
            )
        if p.resp_always_valid and p.fixed_latency is None:
            raise ProtocolCompatError(
                f"{location}: resp_always_valid=True requires fixed_latency to be set"
            )
        if p.fixed_latency is not None and p.resp_has_backpressure:
            raise ProtocolCompatError(
                f"{location}: fixed_latency and resp_has_backpressure=True are mutually exclusive"
            )
        if not p.in_order and p.max_outstanding == 1:
            warnings.warn(
                f"{location}: in_order=False with max_outstanding=1 is trivially in-order — "
                "consider setting in_order=True",
                UserWarning,
                stacklevel=3,
            )

    # ------------------------------------------------------------------
    # Cross-side compatibility checks (§6.1)
    # ------------------------------------------------------------------

    def _check_outstanding(self, pp, ep, location: str) -> None:
        """max_outstanding: export must be able to serve as many requests."""
        if ep.max_outstanding < pp.max_outstanding:
            raise ProtocolCompatError(
                f"{location}: export max_outstanding={ep.max_outstanding} cannot serve "
                f"port max_outstanding={pp.max_outstanding} simultaneous requests"
            )

    def _check_ordering(self, pp, ep, location: str) -> None:
        """in_order: port expects ordered; export may deliver out-of-order → warning."""
        if pp.in_order and not ep.in_order:
            warnings.warn(
                f"{location}: port expects in_order=True but export may deliver "
                "responses out of order (in_order=False) — a reorder buffer is required",
                UserWarning,
                stacklevel=3,
            )

    def _check_request_channel(self, pp, ep, location: str) -> None:
        """req_always_ready: no constraints prevent a valid connection."""
        # req_always_ready=True on port → export's ready line is ignored or tied high; always OK.
        # req_always_ready=False on port + req_always_ready=True on export → OK.
        # Any other combination is also OK at this level; signal routing handles it.
        pass  # no error / warning cases for the request channel

    def _check_response_channel(self, pp, ep, location: str) -> None:
        """resp_always_valid / resp_has_backpressure cross-checks."""
        if pp.resp_always_valid and not ep.resp_always_valid:
            raise ProtocolCompatError(
                f"{location}: port assumes fixed-latency response (resp_always_valid=True) "
                "but export has variable latency (resp_always_valid=False)"
            )
        if pp.resp_has_backpressure and not ep.resp_has_backpressure:
            warnings.warn(
                f"{location}: port drives resp_ready (resp_has_backpressure=True) "
                "but export does not honour it (resp_has_backpressure=False) — "
                "resp_ready will be synthesized but left unconnected on the export side",
                UserWarning,
                stacklevel=3,
            )
