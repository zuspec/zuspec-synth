#****************************************************************************
# Copyright 2019-2026 Matthew Ballance and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#****************************************************************************
"""Tests for memory optimization."""

import pytest
import sys
import os

_this_dir = os.path.dirname(__file__)
_synth_src = os.path.join(_this_dir, '..', 'src')
_dc_src = os.path.join(_this_dir, '..', '..', 'zuspec-dataclasses', 'src')

if '' in sys.path:
    sys.path.insert(1, _synth_src)
    sys.path.insert(2, _dc_src)
else:
    sys.path.insert(0, _synth_src)
    sys.path.insert(1, _dc_src)

from zuspec.synth.sprtl.memory import (
    PartitionType, ArrayInfo, PartitionConfig, MemoryBank,
    PartitionedArray, ArrayPartitioner, BufferConfig, MemoryBuffer,
    MemorySVGenerator, partition_array
)


class TestArrayInfo:
    """Tests for array information."""
    
    def test_1d_array(self):
        """Test 1D array info."""
        arr = ArrayInfo(name="data", element_width=32, dimensions=[8])
        
        assert arr.total_elements == 8
        assert arr.total_bits == 256
        assert arr.num_dimensions == 1
    
    def test_2d_array(self):
        """Test 2D array info."""
        arr = ArrayInfo(name="matrix", element_width=16, dimensions=[4, 4])
        
        assert arr.total_elements == 16
        assert arr.total_bits == 256
        assert arr.num_dimensions == 2
    
    def test_3d_array(self):
        """Test 3D array info."""
        arr = ArrayInfo(name="tensor", element_width=8, dimensions=[2, 3, 4])
        
        assert arr.total_elements == 24
        assert arr.num_dimensions == 3


class TestPartitionConfig:
    """Tests for partition configuration."""
    
    def test_default_config(self):
        """Test default partition config."""
        config = PartitionConfig()
        assert config.partition_type == PartitionType.NONE
        assert config.factor == 1
    
    def test_cyclic_config(self):
        """Test cyclic partition config."""
        config = PartitionConfig(
            partition_type=PartitionType.CYCLIC,
            dimension=0,
            factor=4
        )
        assert config.partition_type == PartitionType.CYCLIC
        assert config.factor == 4


class TestMemoryBank:
    """Tests for memory bank."""
    
    def test_bank_creation(self):
        """Test creating a memory bank."""
        bank = MemoryBank(
            name="data_bank0",
            array_name="data",
            bank_id=0,
            element_width=32,
            depth=16
        )
        
        assert bank.addr_width == 4  # log2(16)
    
    def test_small_bank(self):
        """Test small bank address width."""
        bank = MemoryBank(
            name="data_bank0",
            array_name="data",
            bank_id=0,
            element_width=32,
            depth=1
        )
        
        assert bank.addr_width == 1  # Minimum


class TestArrayPartitioner:
    """Tests for array partitioning."""
    
    def test_no_partition(self):
        """Test no partitioning."""
        arr = ArrayInfo(name="data", element_width=32, dimensions=[8])
        config = PartitionConfig(partition_type=PartitionType.NONE)
        
        partitioner = ArrayPartitioner()
        result = partitioner.partition(arr, config)
        
        assert result.num_banks == 1
        assert result.banks[0].depth == 8
    
    def test_complete_partition(self):
        """Test complete partitioning (to registers)."""
        arr = ArrayInfo(name="data", element_width=32, dimensions=[4])
        config = PartitionConfig(partition_type=PartitionType.COMPLETE)
        
        partitioner = ArrayPartitioner()
        result = partitioner.partition(arr, config)
        
        assert result.num_banks == 4
        for bank in result.banks:
            assert bank.depth == 1
    
    def test_cyclic_partition(self):
        """Test cyclic partitioning."""
        arr = ArrayInfo(name="data", element_width=32, dimensions=[8])
        config = PartitionConfig(
            partition_type=PartitionType.CYCLIC,
            dimension=0,
            factor=2
        )
        
        partitioner = ArrayPartitioner()
        result = partitioner.partition(arr, config)
        
        assert result.num_banks == 2
        assert result.banks[0].depth == 4
        assert result.banks[1].depth == 4
    
    def test_block_partition(self):
        """Test block partitioning."""
        arr = ArrayInfo(name="data", element_width=32, dimensions=[8])
        config = PartitionConfig(
            partition_type=PartitionType.BLOCK,
            dimension=0,
            factor=2
        )
        
        partitioner = ArrayPartitioner()
        result = partitioner.partition(arr, config)
        
        assert result.num_banks == 2


class TestPartitionedArrayMapping:
    """Tests for address mapping in partitioned arrays."""
    
    def test_no_partition_mapping(self):
        """Test address mapping with no partition."""
        arr = ArrayInfo(name="data", dimensions=[8])
        result = partition_array(arr, PartitionType.NONE)
        
        bank, addr = result.get_bank_and_addr([5])
        assert bank == 0
        assert addr == 5
    
    def test_complete_partition_mapping(self):
        """Test address mapping with complete partition."""
        arr = ArrayInfo(name="data", dimensions=[4])
        result = partition_array(arr, PartitionType.COMPLETE)
        
        bank, addr = result.get_bank_and_addr([2])
        assert bank == 2
        assert addr == 0
    
    def test_cyclic_partition_mapping(self):
        """Test address mapping with cyclic partition."""
        arr = ArrayInfo(name="data", dimensions=[8])
        result = partition_array(arr, PartitionType.CYCLIC, factor=2)
        
        # Element 0 -> bank 0, addr 0
        bank, addr = result.get_bank_and_addr([0])
        assert bank == 0
        
        # Element 1 -> bank 1, addr 0
        bank, addr = result.get_bank_and_addr([1])
        assert bank == 1
        
        # Element 2 -> bank 0, addr 1
        bank, addr = result.get_bank_and_addr([2])
        assert bank == 0


class TestMemorySVGenerator:
    """Tests for memory SystemVerilog generation."""
    
    def test_generate_no_partition(self):
        """Test SV generation for non-partitioned array."""
        arr = ArrayInfo(name="data", element_width=32, dimensions=[16])
        result = partition_array(arr, PartitionType.NONE)
        
        generator = MemorySVGenerator()
        code = generator.generate_partitioned_array(result)
        
        assert "data_bank0" in code
        assert "[15:0]" in code  # depth
    
    def test_generate_complete_partition(self):
        """Test SV generation for complete partition."""
        arr = ArrayInfo(name="data", element_width=8, dimensions=[4])
        result = partition_array(arr, PartitionType.COMPLETE)
        
        generator = MemorySVGenerator()
        code = generator.generate_partitioned_array(result)
        
        assert "data_r0" in code
        assert "data_r1" in code
        assert "data_r2" in code
        assert "data_r3" in code
    
    def test_generate_buffer(self):
        """Test SV generation for buffer."""
        buf_config = BufferConfig(depth=8, width=32, is_fifo=True)
        buffer = MemoryBuffer(
            name="input_buf",
            config=buf_config,
            input_signal="data_in",
            output_signal="data_out"
        )
        
        generator = MemorySVGenerator()
        code = generator.generate_buffer(buffer)
        
        assert "input_buf" in code
        assert "wr_ptr" in code
        assert "rd_ptr" in code


class TestPartitionArrayFunction:
    """Tests for convenience function."""
    
    def test_partition_array(self):
        """Test partition_array function."""
        arr = ArrayInfo(name="test", dimensions=[16])
        
        result = partition_array(arr, PartitionType.CYCLIC, factor=4)
        
        assert result.num_banks == 4
        assert result.config.partition_type == PartitionType.CYCLIC
