"""Walk a zdc Action activity sequence and extract FSM structure.

Given an ``activity()`` method of the form::

    async def activity(self):
        while True:
            fetch = await Fetch(self.comp)
            dec   = await Decode(self.comp, fetch_buf=fetch.out)
            exe   = await Execute(self.comp, dec_buf=dec.out)
            mem   = await Memory(self.comp, exe_buf=exe.out)
            await Writeback(self.comp, exe_buf=exe.out, mem_buf=mem.out)

The walker produces an ordered list of :class:`ActionStep` objects describing:

  - which action class is instantiated
  - its Buffer inputs (from earlier steps' outputs)
  - the binding name (variable name in the activity loop)

This information is used by:
  - ``BufferElaborationPass`` to declare wire/register sets
  - Constraint emitters to construct signal_maps
  - The synthesis entry point to orchestrate constraint compilation
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BufferEdge:
    """One ``Buffer[T]`` connection between two action steps.

    Attributes
    ----------
    src_step:   variable name of the producing step (e.g. 'fetch')
    src_field:  field name on the producing step (e.g. 'out')
    dst_step:   variable name of the consuming step (e.g. 'dec')
    dst_kwarg:  keyword arg name in the consumer ctor (e.g. 'fetch_buf')
    """
    src_step:  str
    src_field: str
    dst_step:  str
    dst_kwarg: str


@dataclass
class ActionStep:
    """One action instantiation inside the activity loop.

    Attributes
    ----------
    var_name:      Python variable name bound to this step (or '' if unused)
    action_name:   Class name of the action (e.g. 'Fetch', 'Decode')
    action_cls:    Resolved class object (None if not resolved at parse time)
    buffer_inputs: mapping from kwarg name to (src_step, src_field) tuple
    """
    var_name:      str
    action_name:   str
    action_cls:    Optional[type]
    buffer_inputs: Dict[str, Tuple[str, str]] = field(default_factory=dict)


class ActivityBodyWalker:
    """Parse a ``activity()`` coroutine and extract the action sequence.

    Parameters
    ----------
    activity_method:
        The unbound ``activity`` method from the action/scenario class.
    namespace:
        Optional dict to resolve action class names (e.g. the module's
        globals).  When ``None``, class objects are not resolved.
    """

    def __init__(self, activity_method, namespace: Optional[Dict[str, Any]] = None):
        self._method = activity_method
        self._namespace = namespace or {}
        self._steps: List[ActionStep] = []
        self._parse()

    @property
    def steps(self) -> List[ActionStep]:
        """Ordered list of ActionStep objects extracted from the activity."""
        return list(self._steps)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(self) -> None:
        src = inspect.getsource(self._method)
        src = textwrap.dedent(src)
        tree = ast.parse(src)
        func_def = tree.body[0]
        if not isinstance(func_def, (ast.AsyncFunctionDef, ast.FunctionDef)):
            raise ValueError(f'Expected (Async)FunctionDef, got {type(func_def).__name__}')

        # Flatten: look inside while-True loop if present
        body = self._unwrap_while_true(func_def.body)
        self._parse_body(body)

    @staticmethod
    def _unwrap_while_true(stmts: List[ast.stmt]) -> List[ast.stmt]:
        """If the body is a single ``while True:`` loop, return its inner body."""
        if (len(stmts) == 1
                and isinstance(stmts[0], ast.While)
                and isinstance(stmts[0].test, ast.Constant)
                and stmts[0].test.value is True):
            return stmts[0].body
        return stmts

    def _parse_body(self, stmts: List[ast.stmt]) -> None:
        for stmt in stmts:
            step = self._try_parse_step(stmt)
            if step is not None:
                self._steps.append(step)

    def _try_parse_step(self, stmt: ast.stmt) -> Optional[ActionStep]:
        """Try to parse one ``var = await ActionClass(...)`` statement."""
        # Pattern 1: ``var = await ActionClass(...)``
        if (isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.Await)):
            var_name = stmt.targets[0].id
            call = stmt.value.value
            return self._parse_action_call(var_name, call)

        # Pattern 2: ``await ActionClass(...)`` (result discarded)
        if (isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Await)):
            call = stmt.value.value
            return self._parse_action_call('', call)

        return None

    def _parse_action_call(
        self, var_name: str, call: ast.expr
    ) -> Optional[ActionStep]:
        """Parse an ``ActionClass(self.comp, kwarg=step.field, ...)`` call."""
        if not isinstance(call, ast.Call):
            return None

        # Class name
        if isinstance(call.func, ast.Name):
            action_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            action_name = call.func.attr
        else:
            return None

        # Resolve class
        action_cls = self._namespace.get(action_name)

        # Parse keyword args for buffer connections: ``fetch_buf=fetch.out``
        buffer_inputs: Dict[str, Tuple[str, str]] = {}
        for kw in call.keywords:
            kw_name = kw.arg  # e.g. 'fetch_buf'
            kw_val  = kw.value
            if (isinstance(kw_val, ast.Attribute)
                    and isinstance(kw_val.value, ast.Name)):
                src_step  = kw_val.value.id   # e.g. 'fetch'
                src_field = kw_val.attr        # e.g. 'out'
                if kw_name is not None:
                    buffer_inputs[kw_name] = (src_step, src_field)

        return ActionStep(
            var_name=var_name,
            action_name=action_name,
            action_cls=action_cls,
            buffer_inputs=buffer_inputs,
        )


def build_signal_map(
    step: ActionStep,
    step_prefix: str,
    buffer_field_widths: Dict[str, Dict[str, int]],
) -> Dict[str, str]:
    """Build a signal_map for FunctionalConstraintCompiler from one ActionStep.

    Parameters
    ----------
    step:
        The ActionStep whose action class is being compiled.
    step_prefix:
        Prefix for output signals of this step (e.g. 'exe' for Execute).
    buffer_field_widths:
        Nested dict ``{field_name: {subfield_name: bit_width}}``.
        Used to enumerate field names for each Buffer input/output.
        E.g. ``{'dec': {'kind': 4, 'alu_op': 4, ...}}``.

    Returns
    -------
    dict mapping dotted path (without 'self.') to SV signal name.
    """
    signal_map: Dict[str, str] = {}

    if step.action_cls is None:
        return signal_map

    import dataclasses as dc

    try:
        dc_fields = dc.fields(step.action_cls)
    except TypeError:
        return signal_map

    for f in dc_fields:
        fname = f.name
        if fname.startswith('_'):
            # Class-level rand field (e.g. _taken) — map directly
            sv_name = f'{step_prefix}_{fname.lstrip("_")}'
            signal_map[fname] = sv_name
            continue

        # Inspect metadata to determine field kind
        meta = f.metadata or {}
        kind = meta.get('zdc_kind', '')

        # Output buffer: self.out.t.X → {step_prefix}_{X}
        if kind == 'output':
            subfields = buffer_field_widths.get(fname, {})
            for sf in subfields:
                signal_map[f'{fname}.t.{sf}'] = f'{step_prefix}_{sf}'

        # Input buffer: self.dec.t.X → {src_prefix}_{X}
        elif kind == 'input':
            # Find the corresponding buffer's source step prefix
            src_prefix = fname  # default: use field name as prefix
            subfields = buffer_field_widths.get(fname, {})
            for sf in subfields:
                signal_map[f'{fname}.t.{sf}'] = f'{src_prefix}_{sf}'

        # Resource (lock/share): self.pc.t → {step_prefix}_{fname}
        elif kind in ('lock', 'share'):
            signal_map[f'{fname}.t'] = f'{step_prefix}_{fname}'
            signal_map[f'{fname}.id'] = f'{step_prefix}_{fname}_id'

    return signal_map
