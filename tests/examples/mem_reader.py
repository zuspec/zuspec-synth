"""Minimal component with one awaited ProtocolPort method call.

Used by test_port_call_lowering.py to verify FSMPortCall lowering.
"""

import typing
import zuspec.dataclasses as zdc


class MemPort(typing.Protocol):
    async def read_word(self, addr: zdc.u32) -> zdc.u32: ...


@zdc.dataclass
class MemReader(zdc.Component):
    mem: MemPort = zdc.port()

    @zdc.proc
    async def _run(self):
        while True:
            data = await self.mem.read_word(0)
