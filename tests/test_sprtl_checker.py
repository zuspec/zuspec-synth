#!/usr/bin/env python3
"""
Tests for the SPRTL Synthesizable profile checker.
"""

import pytest
import sys
import os

# Ensure paths are set correctly for development
_this_dir = os.path.dirname(__file__)
_synth_src = os.path.join(_this_dir, '..', 'src')
_dc_src = os.path.join(_this_dir, '..', '..', 'zuspec-dataclasses', 'src')

if '' in sys.path:
    sys.path.insert(1, _synth_src)
    sys.path.insert(2, _dc_src)
else:
    sys.path.insert(0, _synth_src)
    sys.path.insert(1, _dc_src)

import zuspec.dataclasses as zdc
from zuspec.dataclasses.ir_checker import (
    SPRTLSynthesizableChecker, CheckContext, CheckerRegistry
)


class TestSPRTLCheckerBasic:
    """Basic tests for SPRTL checker."""
    
    def test_checker_registered(self):
        """Test that SPRTL checker is registered."""
        checker_class = CheckerRegistry.get_checker('SPRTLSynthesizable')
        assert checker_class is not None
        assert checker_class.PROFILE_NAME == 'SPRTLSynthesizable'
    
    def test_checker_creation(self):
        """Test checker can be instantiated."""
        checker = SPRTLSynthesizableChecker()
        assert checker is not None


class TestSPRTLValidComponents:
    """Test that valid SPRTL components pass the checker."""
    
    def test_simple_counter_valid(self):
        """Test that a simple counter passes validation."""
        from test_components import SimpleCounter
        
        factory = zdc.DataModelFactory()
        context = factory.build(SimpleCounter)
        
        checker = SPRTLSynthesizableChecker()
        check_ctx = CheckContext()
        
        errors = checker.check_context(context, check_ctx)
        
        # Filter to only SPRTL-specific errors (ZDS prefix)
        sprtl_errors = [e for e in errors if e.code.startswith('ZDS')]
        
        # Should have no SPRTL-specific errors for valid component
        assert len(sprtl_errors) == 0, f"Unexpected errors: {sprtl_errors}"
    
    def test_updown_counter_valid(self):
        """Test that an up/down counter passes validation."""
        from test_components import UpDownCounter
        
        factory = zdc.DataModelFactory()
        context = factory.build(UpDownCounter)
        
        checker = SPRTLSynthesizableChecker()
        check_ctx = CheckContext()
        
        errors = checker.check_context(context, check_ctx)
        sprtl_errors = [e for e in errors if e.code.startswith('ZDS')]
        
        assert len(sprtl_errors) == 0, f"Unexpected errors: {sprtl_errors}"
    
    def test_sequential_processor_valid(self):
        """Test that a multi-state FSM passes validation."""
        from test_components import SequentialProcessor
        
        factory = zdc.DataModelFactory()
        context = factory.build(SequentialProcessor)
        
        checker = SPRTLSynthesizableChecker()
        check_ctx = CheckContext()
        
        errors = checker.check_context(context, check_ctx)
        sprtl_errors = [e for e in errors if e.code.startswith('ZDS')]
        
        assert len(sprtl_errors) == 0, f"Unexpected errors: {sprtl_errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
