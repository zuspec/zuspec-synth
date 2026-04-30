"""Minimal component with a non-awaited ProtocolPort method call.

Used by test_port_call_lowering.py to verify FSMPortOutput lowering.
"""

import typing
import zuspec.dataclasses as zdc


class MonitorPort(typing.Protocol):
    def on_event(self, value: zdc.u32) -> None: ...


@zdc.dataclass
class MonitorWriter(zdc.Component):
    monitor: MonitorPort = zdc.port()

    @zdc.proc
    async def _run(self):
        while True:
            self.monitor.on_event(42)
