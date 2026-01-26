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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
