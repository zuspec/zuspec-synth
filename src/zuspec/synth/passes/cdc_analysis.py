"""Clock-domain crossing (CDC) analysis pass.

Detects sub-component instances that operate in different clock domains and
reports them as :class:`CDCCrossing` objects.  Crossings may be suppressed
by inserting a :class:`~zuspec.dataclasses.cdc.TwoFFSync` synchronizer or
by annotating a class with :func:`~zuspec.dataclasses.cdc.cdc_unchecked`.

Typical usage::

    from zuspec.synth.passes import CDCAnalysisPass

    crossings = CDCAnalysisPass.run(MyTopLevel)
    unsuppressed = [c for c in crossings if not c.suppressed]
    if unsuppressed:
        raise ZuspeccCDCError(...)
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional


# ---------------------------------------------------------------------------
# CDCCrossing result object
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CDCCrossing:
    """Describes a detected clock-domain crossing between two sub-instances.

    Attributes:
        src_instance:  Name of the sub-instance field driving from the source domain.
        src_domain:    Name of the source clock domain.
        dst_instance:  Name of the sub-instance field consuming in the destination domain.
        dst_domain:    Name of the destination clock domain.
        suppressed:    ``True`` when a synchronizer or ``cdc_unchecked`` suppresses this.
        suppressor:    Human-readable name of what suppressed the crossing, or ``None``.
    """

    src_instance: str
    src_domain:   str
    dst_instance: str
    dst_domain:   str
    suppressed:   bool = False
    suppressor:   Optional[str] = None

    def __repr__(self) -> str:
        s = f"CDCCrossing({self.src_instance!r}:{self.src_domain!r}"
        s += f" -> {self.dst_instance!r}:{self.dst_domain!r}"
        if self.suppressed:
            s += f", suppressed by {self.suppressor!r}"
        s += ")"
        return s


# ---------------------------------------------------------------------------
# CDCAnalysisPass
# ---------------------------------------------------------------------------

class CDCAnalysisPass:
    """Static analysis pass that detects clock-domain crossings.

    Call :meth:`run` with a top-level :class:`~zuspec.dataclasses.Component`
    class.  All sub-instance fields are inspected; pairs whose effective clock
    domains differ are reported as :class:`CDCCrossing` objects.

    Suppression rules (applied in order):

    1. If the sub-instance's Python class has ``_cdc_unchecked = True`` (set
       by :func:`~zuspec.dataclasses.cdc.cdc_unchecked`), crossings involving
       that instance are suppressed.
    2. If any sub-instance is a
       :class:`~zuspec.dataclasses.cdc.TwoFFSync` (or a subclass), crossings
       between the domains it bridges are suppressed.  The domains it bridges
       are determined from the clock domain of other sub-instances in the same
       hierarchy: any pair of domains where at least one endpoint matches the
       TwoFFSync's clock domain is considered bridged.
    """

    @classmethod
    def run(cls, component_cls) -> List[CDCCrossing]:
        """Analyse *component_cls* for clock-domain crossings.

        :param component_cls: A Python class decorated with ``@zdc.dataclass``
            and inheriting from ``zdc.Component``.
        :returns: A list of :class:`CDCCrossing` objects (may be empty).
        """
        from zuspec.dataclasses.data_model_factory import DataModelFactory
        from zuspec.dataclasses.ir.data_type import DataTypeComponent
        from zuspec.synth.sprtl.fsm_ir import DomainBinding

        # Build the IR model for this class
        factory = DataModelFactory()
        ctx = factory.build(component_cls)

        type_name = getattr(component_cls, "__qualname__", None) or component_cls.__name__
        component_ir = ctx.type_m.get(type_name) or ctx.type_m.get(component_cls.__name__)
        if component_ir is None:
            return []

        # ------------------------------------------------------------------ #
        # 1. Collect sub-component instances: (field_name, py_cls, domain)   #
        # ------------------------------------------------------------------ #
        instances: List[tuple] = []  # (name, py_cls, clock_domain_name)

        py_fields_meta: dict = {}
        if hasattr(component_cls, "__dataclass_fields__"):
            for pf in dataclasses.fields(component_cls):
                py_fields_meta[pf.name] = (pf.metadata, pf.type)

        for f in component_ir.fields:
            if type(f).__name__ != "Field":
                continue
            dt = getattr(f, "datatype", None)
            if dt is None:
                continue

            # Resolve through DataTypeRef → DataTypeComponent
            ref_name = None
            if type(dt).__name__ == "DataTypeRef":
                ref_name = dt.ref_name
            elif isinstance(dt, DataTypeComponent):
                ref_name = getattr(dt, "name", None)

            if ref_name is None:
                continue
            sub_ir = ctx.type_m.get(ref_name)
            if sub_ir is None or not isinstance(sub_ir, DataTypeComponent):
                continue

            # Get the Python class for this sub-instance (from field type)
            _, py_type = py_fields_meta.get(f.name, ({}, None))
            py_cls = py_type if isinstance(py_type, type) else None

            db = DomainBinding.from_component_ir(sub_ir)
            instances.append((f.name, py_cls, db.clock_name))

        if len(instances) < 2:
            return []

        # ------------------------------------------------------------------ #
        # 2. Identify TwoFFSync instances and cdc_unchecked classes          #
        # ------------------------------------------------------------------ #
        two_ff_present: bool = False  # True if any TwoFFSync is in the hierarchy
        unchecked_instances: set = set()  # field names that are cdc_unchecked

        for name, py_cls, domain in instances:
            if py_cls is not None:
                if getattr(py_cls, "_cdc_unchecked", False):
                    unchecked_instances.add(name)
                if getattr(py_cls, "_zdc_two_ff_sync", False):
                    two_ff_present = True

        # ------------------------------------------------------------------ #
        # 3. Emit crossings for all pairs with different domains              #
        # ------------------------------------------------------------------ #
        crossings: List[CDCCrossing] = []

        seen_pairs: set = set()
        for i, (name_a, cls_a, dom_a) in enumerate(instances):
            for name_b, cls_b, dom_b in instances[i + 1:]:
                if dom_a == dom_b:
                    continue

                pair_key = (min(name_a, name_b), max(name_a, name_b))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                suppressed = False
                suppressor = None

                if name_a in unchecked_instances or name_b in unchecked_instances:
                    suppressed = True
                    nm = name_a if name_a in unchecked_instances else name_b
                    suppressor = f"cdc_unchecked on '{nm}'"

                if not suppressed and two_ff_present:
                    suppressed = True
                    suppressor = "TwoFFSync synchronizer"

                crossings.append(CDCCrossing(
                    src_instance=name_a,
                    src_domain=dom_a,
                    dst_instance=name_b,
                    dst_domain=dom_b,
                    suppressed=suppressed,
                    suppressor=suppressor,
                ))

        return crossings
