"""Minimal component for testing SPRTL self-method inlining.

Exercises three synthesiser fixes:
1. ExprSubscript index rewriting  — ``gpr[rd]`` inside ``_alu`` where ``rd``
   is a *local* variable (not a passed argument) must be prefixed to
   ``gpr[_alu_rd]`` after inlining.
2. Tuple-temporary expansion — ``StmtAnnAssign`` with ``_zsp_tuple``.
3. Nested FSMCond register inference — signals assigned only inside
   ``if/else`` branches must be declared.
"""

import typing
import zuspec.dataclasses as zdc


class _MemPort(typing.Protocol):
    async def read_word(self, addr: zdc.u32) -> zdc.u32: ...


@zdc.dataclass
class InlineCore(zdc.Component):
    """Minimal core: fetch → decode (local) → execute (local) loop."""

    mem: _MemPort = zdc.port()

    # 8-entry GPR register file
    gpr: zdc.Array[zdc.u32] = zdc.array(8)

    @zdc.proc
    async def _run(self):
        pc: int = 0
        while True:
            instr = await self.mem.read_word(pc)
            # _decode is inlined at DMF level (module-level pure async fn)
            rs1   = (instr >> 15) & 0x1F
            imm   = zdc.sext((instr >> 20) & 0xFFF, 12)
            # _alu is a self._ private method — inlined by SPRTL transformer.
            # It has a LOCAL 'rd' (not passed as argument) to test ExprSubscript
            # index rewriting.
            next_pc = await self._alu(instr, rs1, imm, pc)
            pc = next_pc & 0xFFFFFFFF

    async def _alu(self, instr: int, rs1: int, imm: int, pc: int) -> int:
        # rd is a LOCAL variable inside _alu — after inlining it becomes _alu_rd.
        # The assignment gpr[rd] must use the prefixed _alu_rd index.
        rd      = (instr >> 7) & 0x1F
        a       = self.gpr[rs1] & 0xFFFFFFFF
        result  = (a + imm) & 0xFFFFFFFF
        taken   = False
        if rd != 0:
            self.gpr[rd] = result & 0xFFFFFFFF
            taken = True
        return (pc + 4) & 0xFFFFFFFF
