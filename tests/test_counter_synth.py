import sys
import pytest

sys.path.insert(0, 'packages/zuspec-dataclasses/src')
sys.path.insert(0, 'packages/zuspec-synth/src')
sys.path.insert(0, 'examples/01_counter')


@pytest.fixture(scope="module")
def counter_sv():
    import zuspec.dataclasses as zdc
    from zuspec.synth import synthesize

    @zdc.dataclass
    class Counter(zdc.Component):
        count: zdc.u32 = zdc.field(default=0)

        @zdc.proc
        async def _count(self):
            while True:
                self.count = self.count + 1
                await zdc.tick()

    return synthesize(Counter)


def test_has_clock_reset_ports(counter_sv):
    assert "input clock" in counter_sv
    assert "input reset" in counter_sv


def test_has_count_output_reg(counter_sv):
    assert "output reg" in counter_sv
    assert "count" in counter_sv


def test_has_always_posedge(counter_sv):
    assert "always @(posedge clock or posedge reset)" in counter_sv


def test_reset_clears_count(counter_sv):
    assert "count <= 0" in counter_sv


def test_body_increments_count(counter_sv):
    # count <= count + 1 (possibly with spacing variations)
    assert "count <= " in counter_sv
    assert "count + 1" in counter_sv or "(count + 1)" in counter_sv
