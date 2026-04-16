"""Compile a functional unit execute()/access() method body to SystemVerilog RTL.

Handles the synthesizable Python subset present in ALUUnit.execute() and
LoadStoreUnit.access():
  - match op: case Enum.V: return expr   → SV case block (combinational)
  - arithmetic/logic expressions         → direct SV operators
  - signed comparisons via _s32()        → $signed()
  - int() wrapper                        → transparent
  - zdc.uN(const)                        → integer literal
  - await self.wait()                    → multi-cycle boundary (state machine)
  - IfExp (cond ? a : b)                 → ternary
"""
from __future__ import annotations

import ast
import inspect
import logging
import textwrap
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


def _mask_to_bits(mask: int) -> int:
    """Return number of bits needed to represent mask (e.g. 0x1F → 5)."""
    n = 0
    while mask >> n:
        n += 1
    return n


def _binop_sv(op: ast.operator) -> str:
    return {
        ast.Add:    '+',
        ast.Sub:    '-',
        ast.BitAnd: '&',
        ast.BitOr:  '|',
        ast.BitXor: '^',
        ast.LShift: '<<',
        ast.RShift: '>>',
        ast.Mult:   '*',
        ast.FloorDiv: '/',
        ast.Mod:    '%',
    }.get(type(op), '?')


def _cmpop_sv(op: ast.cmpop) -> str:
    return {
        ast.Lt:    '<',
        ast.Gt:    '>',
        ast.LtE:   '<=',
        ast.GtE:   '>=',
        ast.Eq:    '==',
        ast.NotEq: '!=',
    }.get(type(op), '?')


class _ExprTranslator(ast.NodeVisitor):
    """Translate a Python expression AST node to a SV expression string.

    Args:
        local_vars: dict of locally-assigned variable name → SV expression string
        xlen: data width (e.g. 32) for literal widths
        signal_map: optional dict mapping Python param names to SV signal names
    """

    def __init__(self, local_vars: Dict[str, str], xlen: int = 32,
                 signal_map: Optional[Dict[str, str]] = None):
        self._locals = local_vars
        self._xlen = xlen
        self._signal_map = signal_map or {}

    def translate(self, node: ast.expr) -> str:
        result = self.visit(node)
        if result is None:
            return f'/* unsupported:{ast.dump(node)} */'
        return result

    def visit_Name(self, node: ast.Name) -> str:
        name = node.id
        if name in self._locals:
            return self._locals[name]
        return self._signal_map.get(name, name)

    def visit_Constant(self, node: ast.Constant) -> str:
        v = node.value
        if isinstance(v, int):
            return str(v)
        return repr(v)

    def _const_value(self, node) -> Optional[int]:
        """Extract integer constant from ast.Constant or zdc.uN(const)."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'zdc'
                and node.func.attr.startswith('u')
                and node.args and isinstance(node.args[0], ast.Constant)):
            return int(node.args[0].value)
        return None

    def visit_BinOp(self, node: ast.BinOp) -> str:
        left  = self.translate(node.left)
        right = self.translate(node.right)
        op    = _binop_sv(node.op)

        # int(a) >> int(b) (arithmetic shift) in model → SV $signed(a) >>>
        if isinstance(node.op, ast.RShift):
            if self._is_int_call(node.left) and self._is_int_call(node.right):
                inner_l = self.translate(node.left.args[0])
                inner_r = self.translate(node.right.args[0])
                return f'$signed({inner_l}) >>> {inner_r}'

        # a & mask  (power-of-two-minus-1 constant) → bit-select
        if isinstance(node.op, ast.BitAnd):
            mask_val = self._const_value(node.right)
            if mask_val is not None:
                nbits = _mask_to_bits(mask_val)
                if (1 << nbits) - 1 == mask_val:
                    return f'{left}[{nbits - 1}:0]'

        return f'({left} {op} {right})'

    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
        operand = self.translate(node.operand)
        if isinstance(node.op, ast.USub):
            return f'(-{operand})'
        if isinstance(node.op, ast.Invert):
            return f'(~{operand})'
        if isinstance(node.op, ast.Not):
            return f'(!{operand})'
        return f'(?{operand})'

    def visit_IfExp(self, node: ast.IfExp) -> str:
        test  = self.translate(node.test)
        body  = self.translate(node.body)
        orelse = self.translate(node.orelse)
        return f'(({test}) ? {body} : {orelse})'

    def visit_Compare(self, node: ast.Compare) -> str:
        left  = self.translate(node.left)
        parts = []
        for op, comp in zip(node.ops, node.comparators):
            right = self.translate(comp)
            parts.append(f'{left} {_cmpop_sv(op)} {right}')
        return ' && '.join(parts)

    def visit_Call(self, node: ast.Call) -> str:
        func = node.func

        # int(x) → transparent
        if isinstance(func, ast.Name) and func.id == 'int':
            return self.translate(node.args[0])

        # _s32(x) → $signed(x)
        if isinstance(func, ast.Name) and func.id == '_s32':
            inner = self.translate(node.args[0])
            return f'$signed({inner})'

        # zdc.u32(const) / zdc.uN(const) → just the constant
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id == 'zdc' and func.attr.startswith('u'):
                if node.args and isinstance(node.args[0], ast.Constant):
                    return str(node.args[0].value)
                return self.translate(node.args[0]) if node.args else '0'

        # bool(x) → x
        if isinstance(func, ast.Name) and func.id == 'bool':
            return self.translate(node.args[0])

        # Fallback: render as a comment
        return f'/* call:{ast.dump(func)} */'

    def _is_int_call(self, node: ast.expr) -> bool:
        return (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == 'int')


class UnitBodyCompiler:
    """Compile ALUUnit.execute() or LoadStoreUnit.access() to SV RTL.

    Usage::

        from zuspec.synth.sprtl.unit_body_compiler import UnitBodyCompiler
        ubc = UnitBodyCompiler(ALUUnit)
        lines = ubc.emit_alu_case(xlen=32, pfx='e')
    """

    def __init__(self, unit_cls: type, method_name: str = 'execute'):
        self._cls         = unit_cls
        self._method_name = method_name
        self._fn          = self._unwrap(getattr(unit_cls, method_name))
        self._enum_cls    = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def emit_alu_case(self, xlen: int, pfx: str = 'e',
                      signal_map: Optional[Dict[str, str]] = None) -> List[str]:
        """Emit a combinational always block for an enum-dispatched execute method.

        Args:
            xlen: data width (32 or 64)
            pfx: pipeline stage prefix (e.g. 'e' → 'e_alu_result', 'e_alu_op')
            signal_map: dict mapping Python parameter names to SV signal names.
                e.g. {'rs1': 'e_rs1_val', 'rs2': 'e_rs2_val'}

        Returns a list of SV lines (no trailing newlines).
        """
        L: List[str] = []
        L.append(f'  always @(*) begin')
        L.extend(self.emit_alu_case_inner(xlen, pfx, signal_map, indent='    '))
        L.append(f'  end')
        return L

    def emit_alu_case_inner(self, xlen: int, pfx: str = 'e',
                             signal_map: Optional[Dict[str, str]] = None,
                             indent: str = '      ') -> List[str]:
        """Emit only the inner ``case`` block for the ALU (no ``always @(*)`` wrapper).

        Used when the case block must be embedded inside a larger conditional
        structure (e.g. LUI / AUIPC / MUL guards).  ``indent`` is prepended to
        every returned line.
        """
        fn_ast  = self._get_ast()
        match_node = self._find_match(fn_ast)
        if match_node is None:
            log.warning("[UnitBodyCompiler] No match statement found in %s.%s",
                        self._cls.__name__, self._method_name)
            return []

        result_sig = f'{pfx}_alu_result'
        op_sig     = f'{pfx}_alu_op'

        # Determine enum class from first case pattern
        self._enum_cls = self._infer_enum_cls(match_node)

        L: List[str] = []
        L.append(f'{indent}case ({op_sig})')
        for case in match_node.cases:
            enum_val, sv_expr = self._compile_case(case, xlen, pfx, signal_map)
            if enum_val is None:
                L.append(f'{indent}  default: {result_sig} = {sv_expr};')
            else:
                L.append(f"{indent}  {enum_val}: {result_sig} = {sv_expr};")
        L.append(f'{indent}endcase')
        return L

    def emit_muldiv_wires(self, xlen: int, pfx: str = 'e',
                          a: str = 'e_alu_a', b: str = 'e_alu_b') -> List[str]:
        """Emit pre-computed multiply-product wire declarations.

        Returns three wires used by :meth:`emit_muldiv_case_inner`:
        ``{pfx}_mul_uu``, ``{pfx}_mul_ss``, ``{pfx}_mul_su`` — each
        ``2*xlen`` bits wide.

        Args:
            xlen: data width (32 or 64)
            pfx:  pipeline stage prefix
            a, b: SV signal names for the two multiplicands
        """
        w2 = 2 * xlen
        return [
            f'wire [{w2-1}:0] {pfx}_mul_uu = {a} * {b};',
            f'wire [{w2-1}:0] {pfx}_mul_ss = $signed({a}) * $signed({b});',
            f'wire [{w2-1}:0] {pfx}_mul_su = $signed({a}) * $unsigned({b});',
        ]

    def emit_muldiv_case_inner(self, xlen: int, pfx: str = 'e',
                                a: str = 'e_alu_a', b: str = 'e_alu_b',
                                indent: str = '      ') -> List[str]:
        """Emit the inner ``case`` block for MUL/DIV operations.

        Parses the if/elif chain in ``MulDivUnit.execute()`` and produces:

            case ({pfx}_funct3)
              3'bNNN: {pfx}_alu_result = <expr>;
              ...
            endcase

        Recognises multiply patterns (signed×signed, signed×unsigned,
        unsigned×unsigned, upper/lower half) and division/remainder
        patterns with div-by-zero guards.

        Prerequisite: :meth:`emit_muldiv_wires` wires must be in scope.
        """
        fn_ast = self._get_ast()
        # Skip to the top-level function body and find the first If node
        if_node = self._find_if_chain(fn_ast)
        if if_node is None:
            log.warning("[UnitBodyCompiler] No if/elif chain found in %s.%s",
                        self._cls.__name__, self._method_name)
            return []

        result_sig = f'{pfx}_alu_result'
        funct3_sig = f'{pfx}_funct3'
        zero_xlen  = f"{{{xlen}{{1'b0}}}}"
        all_ones   = f"{{{xlen}{{1'b1}}}}"
        L: List[str] = []
        L.append(f'{indent}case ({funct3_sig})')

        arms = list(self._collect_if_arms(if_node))
        for key, body_stmts in arms:
            sv_expr = self._compile_muldiv_body(body_stmts, xlen, pfx, a, b,
                                                zero_xlen, all_ones)
            if key is None:
                L.append(f'{indent}  default: {result_sig} = {sv_expr};')
            else:
                L.append(f"{indent}  3'd{key}: {result_sig} = {sv_expr};")
        L.append(f'{indent}endcase')
        return L

    def emit_branch_cond(self, xlen: int, pfx: str = 'e') -> List[str]:
        """Emit a combinational case block for branch condition selection.

        Walks a method whose structure is ``match int(field): case N: cond_expr``.
        Used to compile the branch-condition sub-block from ExecuteInstruction.body().
        """
        fn_ast     = self._get_ast()
        match_node = self._find_match(fn_ast)
        if match_node is None:
            return []

        funct3_sig    = f'{pfx}_funct3'
        branch_sig    = f'{pfx}_branch_cond'
        rs1_sig       = f'{pfx}_rs1_val'
        rs2_sig       = f'{pfx}_rs2_val'

        L: List[str] = []
        L.append(f'  always @(*) begin')
        L.append(f'    case ({funct3_sig})')
        for case in match_node.cases:
            key = self._literal_value(case.pattern)
            body_expr = self._simple_compare(case.body, rs1_sig, rs2_sig)
            if key is None:
                L.append(f"      default: {branch_sig} = 1'b0;")
            else:
                L.append(f"      3'd{key}: {branch_sig} = {body_expr};")
        L.append(f'    endcase')
        L.append(f'  end')
        return L

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap(obj: Any):
        """Unwrap @zdc.proc / ExecProc to the underlying function."""
        if hasattr(obj, 'method'):
            return obj.method
        return obj

    def _get_ast(self) -> ast.Module:
        src = inspect.getsource(self._fn)
        src = textwrap.dedent(src)
        return ast.parse(src)

    @staticmethod
    def _find_match(tree: ast.Module) -> Optional[ast.Match]:
        for node in ast.walk(tree):
            if isinstance(node, ast.Match):
                return node
        return None

    def _infer_enum_cls(self, match_node: ast.Match) -> Optional[type]:
        """Try to find the enum class used in case patterns."""
        import sys
        for case in match_node.cases:
            if not isinstance(case.pattern, ast.MatchValue):
                continue
            val = case.pattern.value
            if not isinstance(val, ast.Attribute):
                continue
            cls_name = val.value.id if isinstance(val.value, ast.Name) else None
            if cls_name:
                for mod in list(sys.modules.values()):
                    candidate = getattr(mod, cls_name, None)
                    if candidate is not None and hasattr(candidate, '__members__'):
                        return candidate
        return None

    def _get_enum_int(self, attr_node: ast.Attribute) -> Optional[int]:
        """Resolve AluOp.ADD → integer value using the enum class."""
        if self._enum_cls is None:
            return None
        attr_name = attr_node.attr
        try:
            return int(self._enum_cls[attr_name])
        except (KeyError, TypeError):
            return None

    @staticmethod
    def _literal_value(pattern) -> Optional[int]:
        """Extract integer from MatchValue(Constant(N))."""
        if isinstance(pattern, ast.MatchValue) and isinstance(pattern.value, ast.Constant):
            return int(pattern.value.value)
        return None

    def _compile_case(self, case: ast.match_case,
                      xlen: int, pfx: str,
                      signal_map: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], str]:
        """Return (case_label_str, sv_expr_str) for one match_case."""
        # Extract enum value from pattern
        if isinstance(case.pattern, ast.MatchValue):
            val_node = case.pattern.value
            if isinstance(val_node, ast.Attribute):
                enum_int = self._get_enum_int(val_node)
                label = str(enum_int) if enum_int is not None else f'/* {val_node.attr} */'
            elif isinstance(val_node, ast.Constant):
                label = str(int(val_node.value))
            else:
                label = None
        elif isinstance(case.pattern, ast.MatchAs) and case.pattern.name is None:
            label = None  # default
        else:
            label = None

        sv_expr = self._compile_body_stmts(case.body, xlen, pfx, signal_map)
        return label, sv_expr

    def _compile_body_stmts(self, stmts: List[ast.stmt],
                             xlen: int, pfx: str,
                             signal_map: Optional[Dict[str, str]] = None) -> str:
        """Compile a list of statements into a single SV expression string."""
        local_vars: Dict[str, str] = {}
        tr = _ExprTranslator(local_vars, xlen, signal_map)
        result_expr = f"{xlen}'b0"

        for stmt in stmts:
            if isinstance(stmt, ast.Return):
                if stmt.value is not None:
                    result_expr = tr.translate(stmt.value)
            elif isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name):
                        local_vars[tgt.id] = tr.translate(stmt.value)
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                if isinstance(stmt.target, ast.Name):
                    local_vars[stmt.target.id] = tr.translate(stmt.value)
            elif isinstance(stmt, ast.Expr):
                # Could be `await self.wait()` — ignore silently
                pass

        return result_expr

    @staticmethod
    def _simple_compare(stmts: List[ast.stmt],
                        rs1_sig: str, rs2_sig: str) -> str:
        """Extract branch condition from ``taken = (cond)`` assignment."""
        local_map = {
            'rs1i': rs1_sig,
            'rs2i': rs2_sig,
            'rs1s': f'$signed({rs1_sig})',
            'rs2s': f'$signed({rs2_sig})',
        }
        tr = _ExprTranslator(local_map)
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == 'taken':
                        return tr.translate(stmt.value)
        return "1'b0"

    # ------------------------------------------------------------------
    # MulDiv helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_if_chain(tree: ast.Module) -> Optional[ast.If]:
        """Return the first top-level ``If`` node in the function body."""
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                return node
        return None

    @staticmethod
    def _collect_if_arms(node: ast.If):
        """Yield (int_key, body_stmts) pairs from an if/elif chain.

        ``int_key`` is the integer from ``op == zdc.u32(N)``; ``None`` for
        a bare ``else`` fall-through.
        """
        current = node
        while True:
            key = UnitBodyCompiler._extract_equality_key(current.test)
            yield (key, current.body)
            orelse = current.orelse
            if not orelse:
                break
            if len(orelse) == 1 and isinstance(orelse[0], ast.If):
                current = orelse[0]
            else:
                # bare else block: yield as default
                yield (None, orelse)
                break

    @staticmethod
    def _extract_equality_key(test_node: ast.expr) -> Optional[int]:
        """Extract N from ``op == zdc.u32(N)`` or ``op == N``."""
        if not isinstance(test_node, ast.Compare):
            return None
        if len(test_node.ops) != 1 or not isinstance(test_node.ops[0], ast.Eq):
            return None
        comp = test_node.comparators[0]
        # zdc.u32(N) form
        if (isinstance(comp, ast.Call)
                and isinstance(comp.func, ast.Attribute)
                and isinstance(comp.func.value, ast.Name)
                and comp.func.value.id == 'zdc'
                and comp.func.attr.startswith('u')
                and comp.args):
            arg = comp.args[0]
            if isinstance(arg, ast.Constant):
                return int(arg.value)
        # bare integer form
        if isinstance(comp, ast.Constant):
            return int(comp.value)
        return None

    @staticmethod
    def _compile_muldiv_body(stmts: List[ast.stmt],
                              xlen: int, pfx: str,
                              a: str, b: str,
                              zero_xlen: str,
                              all_ones: str) -> str:
        """Compile one arm of a MulDiv if/elif chain to a SV expression.

        Recognises:
          * Multiply patterns → wire slices of {pfx}_mul_{uu,ss,su}
          * Division/remainder with div-by-zero guard → ternary with $signed
        Falls back to the generic translator if the pattern is unrecognised.
        """
        w2   = 2 * xlen
        lo   = f'{pfx}_mul_ss[{xlen-1}:0]'
        hi   = f'[{w2-1}:{xlen}]'

        # Collect local variable assignments into a dict so we can resolve
        # references like `product = ...; return (product >> 32) ...`
        local_assigns: Dict[str, ast.expr] = {}
        return_expr: Optional[ast.expr] = None

        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    local_assigns[stmt.targets[0].id] = stmt.value
            elif isinstance(stmt, ast.Return) and stmt.value is not None:
                return_expr = stmt.value

        if return_expr is None:
            return zero_xlen

        # Unwrap zdc.u32(...) wrapper if present
        inner = UnitBodyCompiler._unwrap_zdc_u(return_expr)

        # --- Classify multiply patterns ---

        # Pattern: (X & 0xFFFF_FFFF) or plain X where X is a multiply-derived expr
        masked = UnitBodyCompiler._strip_mask(inner)

        # Resolve intermediate variable references
        resolved = UnitBodyCompiler._resolve_var(masked, local_assigns)

        mul_kind = UnitBodyCompiler._classify_mul(resolved, local_assigns)
        if mul_kind is not None:
            mtype, hi_lo = mul_kind
            wire = f'{pfx}_mul_{mtype}'
            if hi_lo == 'lo':
                return f'{wire}[{xlen-1}:0]'
            else:
                return f'{wire}[{w2-1}:{xlen}]'

        # --- Classify division/remainder patterns ---
        div_kind = UnitBodyCompiler._classify_divmod(stmts, xlen, pfx, a, b,
                                                       zero_xlen, all_ones)
        if div_kind is not None:
            return div_kind

        # --- Fallback: generic expression translator ---
        local_sv: Dict[str, str] = {}
        tr = _ExprTranslator(local_sv, xlen, {})
        for name, expr in local_assigns.items():
            local_sv[name] = tr.translate(expr)
        return tr.translate(return_expr)

    @staticmethod
    def _unwrap_zdc_u(node: ast.expr) -> ast.expr:
        """Strip ``zdc.u32(...)`` / ``zdc.uN(...)`` wrapper."""
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'zdc'
                and node.func.attr.startswith('u')
                and node.args):
            return node.args[0]
        return node

    @staticmethod
    def _strip_mask(node: ast.expr) -> ast.expr:
        """Strip ``& 0xFFFF_FFFF`` masking (irrelevant in SV with truncating regs)."""
        if (isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.BitAnd)
                and isinstance(node.right, ast.Constant)
                and (node.right.value & (node.right.value + 1)) == 0):  # power-of-2 minus 1
            return node.left
        return node

    @staticmethod
    def _resolve_var(node: ast.expr,
                     assigns: Dict[str, ast.expr]) -> ast.expr:
        """Replace a Name reference with its assigned expression if known."""
        if isinstance(node, ast.Name) and node.id in assigns:
            return assigns[node.id]
        return node

    @staticmethod
    def _classify_mul(node: ast.expr,
                      assigns: Dict[str, ast.expr]) -> Optional[Tuple[str, str]]:
        """Return (mul_type, 'lo'|'hi') if *node* is a recognised multiply expr.

        Patterns matched:
          * BinOp(*) directly                              → lo
          * BinOp(>>) where left is a multiply            → hi
          * BinOp(&) then any of the above (already stripped by caller)
        """
        # Upper half: X >> shift
        if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.RShift)):
            inner = UnitBodyCompiler._resolve_var(
                UnitBodyCompiler._strip_mask(node.left), assigns)
            kind = UnitBodyCompiler._mul_type(inner)
            if kind is not None:
                return (kind, 'hi')

        # Lower half: direct multiply
        kind = UnitBodyCompiler._mul_type(node)
        if kind is not None:
            return (kind, 'lo')

        return None

    @staticmethod
    def _mul_type(node: ast.expr) -> Optional[str]:
        """Return 'ss', 'su', or 'uu' if node is a binary multiplication."""
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)):
            return None

        def is_signed(n: ast.expr) -> bool:
            # _s32(x) or $signed(...)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                return n.func.id in ('_s32',)
            return False

        def is_unsigned(n: ast.expr) -> bool:
            # int(x) or (x & mask)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                return n.func.id == 'int'
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitAnd):
                return True
            return False

        l_sig = is_signed(node.left)
        r_sig = is_signed(node.right)
        l_uns = is_unsigned(node.left) or (not l_sig)
        r_uns = is_unsigned(node.right) or (not r_sig)

        if l_sig and r_sig:
            return 'ss'
        if l_sig and r_uns:
            return 'su'
        return 'uu'

    @staticmethod
    def _classify_divmod(stmts: List[ast.stmt],
                          xlen: int, pfx: str,
                          a: str, b: str,
                          zero_xlen: str,
                          all_ones: str) -> Optional[str]:
        """Return SV div/rem expression with div-by-zero guard, or None."""
        # Scan for: an If-guard (div-by-zero) and a return or BinOp result.
        has_guard = False
        guard_returns_dividend = False   # True → remainder operation
        guard_returns_allones  = False   # True → division operation
        local_names: set = set()

        for stmt in stmts:
            if isinstance(stmt, ast.If):
                # Detect div-by-zero guard: test is ``X == 0``
                test = stmt.test
                if (isinstance(test, ast.Compare)
                        and len(test.ops) == 1
                        and isinstance(test.ops[0], ast.Eq)
                        and isinstance(test.comparators[0], ast.Constant)
                        and test.comparators[0].value == 0):
                    has_guard = True
                    for s in stmt.body:
                        if not isinstance(s, ast.Return):
                            continue
                        val = s.value
                        # return all-ones constant → division
                        if (isinstance(val, ast.Call)
                                and isinstance(val.func, ast.Attribute)
                                and isinstance(val.func.value, ast.Name)
                                and val.func.value.id == 'zdc'
                                and val.func.attr.startswith('u')
                                and val.args
                                and isinstance(val.args[0], ast.Constant)):
                            guard_returns_allones = True
                        # return rs1 Name → remainder
                        elif isinstance(val, ast.Name):
                            guard_returns_dividend = True
            elif isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name):
                        local_names.add(tgt.id)

        if not has_guard:
            return None

        # Determine signed-ness from local variable names or return expression
        signed = any(n.endswith('_s') or '_s32' in n for n in local_names)
        # Also look at the return statement
        for stmt in reversed(stmts):
            if not isinstance(stmt, ast.Return):
                continue
            val = stmt.value
            inner = UnitBodyCompiler._strip_mask(UnitBodyCompiler._unwrap_zdc_u(val))
            if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Mod):
                # Direct modulo in return
                op_signed = isinstance(inner.left, ast.Call) and isinstance(
                    inner.left.func, ast.Name) and inner.left.func.id == '_s32'
                if guard_returns_dividend:
                    if op_signed:
                        return f'({b} == {zero_xlen}) ? {a} : ($signed({a}) % $signed({b}))'
                    else:
                        return f'({b} == {zero_xlen}) ? {a} : ({a} % {b})'
            elif (isinstance(inner, ast.BinOp)
                    and isinstance(inner.op, ast.Sub)):
                # rs1_s - rs2_s * q form (signed remainder)
                if guard_returns_dividend:
                    return f'({b} == {zero_xlen}) ? {a} : ($signed({a}) % $signed({b}))'
            break

        # Fallback: use guard context
        if guard_returns_dividend:
            if signed:
                return f'({b} == {zero_xlen}) ? {a} : ($signed({a}) % $signed({b}))'
            else:
                return f'({b} == {zero_xlen}) ? {a} : ({a} % {b})'
        if guard_returns_allones:
            if signed:
                return f'({b} == {zero_xlen}) ? {all_ones} : ($signed({a}) / $signed({b}))'
            else:
                return f'({b} == {zero_xlen}) ? {all_ones} : ({a} / {b})'
        return None
        """Extract branch condition from ``taken = (cond)`` assignment."""
        local_map = {
            'rs1i': rs1_sig,
            'rs2i': rs2_sig,
            'rs1s': f'$signed({rs1_sig})',
            'rs2s': f'$signed({rs2_sig})',
        }
        tr = _ExprTranslator(local_map)
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == 'taken':
                        return tr.translate(stmt.value)
        return "1'b0"
