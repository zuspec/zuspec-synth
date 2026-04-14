#!/usr/bin/env python3
"""
Tests for the SPRTL transformer.
"""

import pytest
import sys
import os

# Ensure paths are set correctly for development
_this_dir = os.path.dirname(__file__)
_synth_src = os.path.join(_this_dir, '..', 'src')
_dc_src = os.path.join(_this_dir, '..', '..', 'zuspec-dataclasses', 'src')

# Insert at front but after '' if present
if '' in sys.path:
    sys.path.insert(1, _synth_src)
    sys.path.insert(2, _dc_src)
else:
    sys.path.insert(0, _synth_src)
    sys.path.insert(1, _dc_src)

import zuspec.dataclasses as zdc
from zuspec.synth.sprtl import SPRTLTransformer, FSMModule, FSMStateKind


class TestFSMIR:
    """Test FSM IR data structures."""
    
    def test_fsm_module_creation(self):
        """Test basic FSM module creation."""
        module = FSMModule(name="test_fsm")
        assert module.name == "test_fsm"
        assert len(module.states) == 0
        assert len(module.ports) == 0
    
    def test_add_port(self):
        """Test adding ports to FSM module."""
        module = FSMModule(name="test")
        module.add_port("clk", "input", width=1)
        module.add_port("data", "output", width=32, reset_value=0)
        
        assert len(module.ports) == 2
        assert module.ports[0].name == "clk"
        assert module.ports[0].direction == "input"
        assert module.ports[1].name == "data"
        assert module.ports[1].reset_value == 0
    
    def test_add_state(self):
        """Test adding states to FSM module."""
        module = FSMModule(name="test")
        s0 = module.add_state("IDLE")
        s1 = module.add_state("RUN")
        
        assert len(module.states) == 2
        assert s0.id == 0
        assert s0.name == "IDLE"
        assert s1.id == 1
        assert s1.name == "RUN"
    
    def test_state_transitions(self):
        """Test state transitions."""
        module = FSMModule(name="test")
        s0 = module.add_state("IDLE")
        s1 = module.add_state("RUN")
        
        s0.add_transition(s1.id, condition="start")
        s0.add_transition(s0.id, priority=1)  # Stay in IDLE if no start
        
        assert len(s0.transitions) == 2
        assert s0.transitions[0].target_state == 1
        assert s0.transitions[0].condition == "start"
        assert s0.transitions[1].is_unconditional == False or s0.transitions[1].priority == 1
    
    def test_state_encoding(self):
        """Test automatic state encoding."""
        module = FSMModule(name="test")
        module.add_state("S0")
        module.add_state("S1")
        module.add_state("S2")
        module.add_state("S3")
        
        # 4 states need 2 bits
        assert module.state_width == 2
        assert len(module.state_encoding) == 4


class TestSPRTLTransformer:
    """Test the SPRTL to FSM transformer."""
    
    def test_transformer_creation(self):
        """Test transformer can be created."""
        transformer = SPRTLTransformer()
        assert transformer is not None
    
    def test_simple_counter_transform(self):
        """Test transforming a simple counter component."""
        from test_components import SimpleCounter
        
        # Get the IR for the component using build()
        factory = zdc.DataModelFactory()
        context = factory.build(SimpleCounter)
        
        # Get the component type from context
        component_ir = context.type_m.get('SimpleCounter')
        assert component_ir is not None, f"Should have SimpleCounter type, got keys: {list(context.type_m.keys())}"
        
        # Find the sync process
        sync_processes = getattr(component_ir, 'sync_processes', [])
        assert len(sync_processes) > 0, "Should have sync processes"
        
        process_ir = sync_processes[0]
        
        # Transform to FSM
        transformer = SPRTLTransformer()
        fsm = transformer.transform(component_ir, process_ir)
        
        assert fsm is not None
        assert isinstance(fsm, FSMModule)
        assert len(fsm.states) > 0, "Should have states"
        
        # Should have at least IDLE state
        idle_state = fsm.get_state_by_name("IDLE")
        assert idle_state is not None, "Should have IDLE state"


class TestCounterExample:
    """Test with the counter example from examples/sprtl."""
    
    def test_counter_ir_extraction(self):
        """Test that counter IR can be extracted."""
        from test_components import UpDownCounter
        
        # Build IR using build()
        factory = zdc.DataModelFactory()
        context = factory.build(UpDownCounter)
        
        # Get component type
        component_ir = context.type_m.get('UpDownCounter')
        assert component_ir is not None, f"Should have UpDownCounter, got keys: {list(context.type_m.keys())}"
        
        # Check component IR structure
        assert hasattr(component_ir, 'fields')
        assert hasattr(component_ir, 'sync_processes')
        
        # Check fields
        fields = component_ir.fields
        field_names = [f.name for f in fields]
        assert 'clock' in field_names
        assert 'reset' in field_names
        assert 'count' in field_names
        
        # Check sync process
        sync_processes = component_ir.sync_processes
        assert len(sync_processes) == 1
        
        process = sync_processes[0]
        assert process.name == 'run'
        assert len(process.body) > 0


# ---------------------------------------------------------------------------
# DMA-engine specific await patterns
# ---------------------------------------------------------------------------

# Path helpers so tests can find the DMA source even when run from the
# packages/zuspec-synth directory.
import os as _os
_repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..'))
_dma_src   = _os.path.join(_repo_root, 'src')
_dc_src    = _os.path.join(_repo_root, 'packages', 'zuspec-dataclasses', 'src')

if _dma_src not in sys.path:
    sys.path.insert(0, _dma_src)
if _dc_src not in sys.path:
    sys.path.insert(1, _dc_src)


def _build_dma_fsm():
    """Build the DmaEngine IR and return (comp_ir, run_proc, FSMModule)."""
    from dma.dma_engine import DmaEngine  # noqa: PLC0415
    factory = zdc.DataModelFactory()
    ctx = factory.build(DmaEngine)
    comp = ctx.type_m['DmaEngine']
    run_proc = next(f for f in comp.functions if f.name == 'run')
    transformer = SPRTLTransformer()
    fsm = transformer.transform(comp, run_proc)
    return comp, run_proc, fsm


class TestDmaAwaitPatterns:
    """Test each await pattern produced by the DMA engine transformer."""

    def test_dma_engine_transforms(self):
        """DmaEngine.run() must transform without exceptions."""
        _, _, fsm = _build_dma_fsm()
        assert fsm is not None
        assert isinstance(fsm, FSMModule)
        assert len(fsm.states) > 0

    def test_has_idle_state(self):
        """FSM must start with an IDLE state."""
        _, _, fsm = _build_dma_fsm()
        idle = fsm.get_state_by_name('IDLE')
        assert idle is not None, f"No IDLE state; got {[s.name for s in fsm.states]}"

    def test_reg_read_creates_normal_state(self):
        """await reg.read() must produce a NORMAL state."""
        from zuspec.synth.sprtl import FSMRegRead  # noqa: PLC0415
        _, _, fsm = _build_dma_fsm()
        reg_read_states = [
            s for s in fsm.states
            if s.kind == FSMStateKind.NORMAL
            and any(isinstance(op, FSMRegRead) for op in s.operations)
        ]
        assert len(reg_read_states) > 0, "No NORMAL states with FSMRegRead operations found"

    def test_mem_request_creates_wait_cond_state(self):
        """await mem.request(req) must produce a WAIT_COND state."""
        from zuspec.synth.sprtl import FSMMemRequest  # noqa: PLC0415
        _, _, fsm = _build_dma_fsm()
        req_states = [
            s for s in fsm.states
            if s.kind == FSMStateKind.WAIT_COND
            and any(isinstance(op, FSMMemRequest) for op in s.operations)
        ]
        assert len(req_states) > 0, "No WAIT_COND states with FSMMemRequest operations found"

    def test_mem_response_creates_wait_cond_state(self):
        """await mem.response() must produce a WAIT_COND state."""
        from zuspec.synth.sprtl import FSMMemResponse  # noqa: PLC0415
        _, _, fsm = _build_dma_fsm()
        rsp_states = [
            s for s in fsm.states
            if s.kind == FSMStateKind.WAIT_COND
            and any(isinstance(op, FSMMemResponse) for op in s.operations)
        ]
        assert len(rsp_states) > 0, "No WAIT_COND states with FSMMemResponse operations found"

    def test_for_loop_creates_chk_and_body_states(self):
        """for var in range(n) must create LOOP_VAR_CHK and LOOP_VAR_BODY states."""
        _, _, fsm = _build_dma_fsm()
        names = [s.name for s in fsm.states]
        # Channel selection uses next() not a for-loop, so no LOOP_I_* states.
        # The transfer loop (for word_idx in range(length)) must be present.
        assert any('LOOP_WORD_IDX' in n for n in names), \
            f"Expected LOOP_WORD_IDX_* states; got {names}"
        assert 'LOOP_WORD_IDX_CHK'  in names
        assert 'LOOP_WORD_IDX_BODY' in names
        assert 'LOOP_WORD_IDX_DONE' in names

    def test_next_generator_creates_priority_encoder(self):
        """next((i for i, v in enumerate(...) if pred), default) must produce a
        FSMAssign with a 'priority_encode' tuple in the current NORMAL state.
        This is a purely combinational operation — no extra FSM state is created.
        """
        from zuspec.synth.sprtl import FSMAssign  # noqa: PLC0415
        _, _, fsm = _build_dma_fsm()
        pe_ops = [
            op for s in fsm.states
            for op in s.operations
            if isinstance(op, FSMAssign)
            and isinstance(op.value, tuple)
            and op.value[0] == 'priority_encode'
        ]
        assert len(pe_ops) > 0, (
            "No FSMAssign(priority_encode) found; next() pattern not recognised"
        )
        # Result must be assigned to the channel-index variable
        assert any(op.target == 'active_idx' for op in pe_ops), \
            f"priority_encode target should be 'active_idx'; got {[op.target for op in pe_ops]}"

    def test_mem_req_rsp_state_wait_conditions(self):
        """MEM_REQ and MEM_RSP WAIT_COND states must reference correct signals."""
        _, _, fsm = _build_dma_fsm()
        mem_req_states = [s for s in fsm.states if s.name.endswith('_REQ')
                          and s.kind == FSMStateKind.WAIT_COND]
        mem_rsp_states = [s for s in fsm.states if s.name.endswith('_RSP')
                          and s.kind == FSMStateKind.WAIT_COND]
        assert len(mem_req_states) > 0
        assert len(mem_rsp_states) > 0
        for s in mem_req_states:
            assert 'ready' in str(s.wait_condition), \
                f"REQ state {s.name!r} has unexpected wait_condition {s.wait_condition!r}"
        for s in mem_rsp_states:
            assert 'valid' in str(s.wait_condition), \
                f"RSP state {s.name!r} has unexpected wait_condition {s.wait_condition!r}"

    def test_reg_write_creates_normal_state(self):
        """await reg.write(val) must produce a NORMAL state (DONE_NOTIFY)."""
        from zuspec.synth.sprtl import FSMRegWrite  # noqa: PLC0415
        _, _, fsm = _build_dma_fsm()
        write_states = [
            s for s in fsm.states
            if s.kind == FSMStateKind.NORMAL
            and any(isinstance(op, FSMRegWrite) for op in s.operations)
        ]
        assert len(write_states) > 0, "No NORMAL states with FSMRegWrite operations"
        # The last write state should be CTRL_WRITE (done-notify)
        assert any(s.name == 'CTRL_WRITE' for s in write_states), \
            f"Expected CTRL_WRITE state; found {[s.name for s in write_states]}"

    def test_fsm_loops_back_to_idle(self):
        """After DONE_NOTIFY the FSM must loop back to state 0 (IDLE)."""
        _, _, fsm = _build_dma_fsm()
        idle = fsm.get_state_by_name('IDLE')
        # CTRL_WRITE (last state) should have a transition → IDLE
        ctrl_write = fsm.get_state_by_name('CTRL_WRITE')
        assert ctrl_write is not None
        back_edge_ids = [t.target_state for t in ctrl_write.transitions]
        assert idle.id in back_edge_ids, \
            f"CTRL_WRITE does not transition back to IDLE; targets={back_edge_ids}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
