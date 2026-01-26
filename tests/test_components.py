"""
Test components for SPRTL transformer tests.

These components must be defined in a separate file from the tests
so that inspect.getsource() can retrieve the source code.
"""

import zuspec.dataclasses as zdc


@zdc.dataclass
class SimpleCounter(zdc.Component):
    """A simple counter that increments on each cycle."""
    clock : zdc.bit = zdc.input()
    reset : zdc.bit = zdc.input()
    count : zdc.u32 = zdc.output(reset=0)
    
    @zdc.sync(clock=lambda s: s.clock, reset=lambda s: s.reset)
    async def run(self):
        while True:
            self.count += 1
            await zdc.cycles(1)


@zdc.dataclass
class UpDownCounter(zdc.Component):
    """A counter with increment and decrement enables."""
    clock : zdc.bit = zdc.input()
    reset : zdc.bit = zdc.input()
    inc_en : zdc.bit = zdc.input()
    dec_en : zdc.bit = zdc.input()
    count : zdc.u32 = zdc.output(reset=0)
    
    @zdc.sync(clock=lambda s: s.clock, reset=lambda s: s.reset)
    async def run(self):
        while True:
            if self.inc_en:
                self.count += 1
            elif self.dec_en:
                self.count -= 1
            await zdc.cycles(1)


@zdc.dataclass
class SequentialProcessor(zdc.Component):
    """A multi-state sequential processor."""
    clock : zdc.bit = zdc.input()
    reset : zdc.bit = zdc.input()
    start : zdc.bit = zdc.input()
    data_in : zdc.u32 = zdc.input()
    result : zdc.u32 = zdc.output(reset=0)
    busy : zdc.bit = zdc.output(reset=0)
    
    @zdc.sync(clock=lambda s: s.clock, reset=lambda s: s.reset)
    async def process(self):
        while True:
            # IDLE state: wait for start
            self.busy = 0
            await self.start == 1
            
            # LOAD state: capture input
            self.busy = 1
            temp = self.data_in
            await zdc.cycles(1)
            
            # COMPUTE state: process data
            temp = temp * 2 + 1
            await zdc.cycles(1)
            
            # STORE state: write result
            self.result = temp
            await zdc.cycles(1)
