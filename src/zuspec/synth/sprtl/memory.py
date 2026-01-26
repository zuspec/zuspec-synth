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
"""
Memory Optimization: Array partitioning and memory banking for HLS.

This module implements memory optimizations:
- Array partitioning (complete, block, cyclic)
- Memory banking for parallel access
- Buffer insertion for burst access

These optimizations increase memory bandwidth and reduce access conflicts.

Algorithm references:
- Adapted from ScaleHLS patterns (Apache 2.0)
"""

from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from math import ceil, log2


class PartitionType(Enum):
    """Types of array partitioning."""
    NONE = auto()       # No partitioning
    COMPLETE = auto()   # Fully partition into registers
    BLOCK = auto()      # Divide into contiguous blocks
    CYCLIC = auto()     # Interleave elements across banks


@dataclass
class ArrayInfo:
    """Information about an array in the design.
    
    Attributes:
        name: Array name
        element_type: Element data type
        element_width: Bits per element
        dimensions: List of dimension sizes
        total_elements: Total number of elements
    """
    name: str
    element_type: str = "logic"
    element_width: int = 32
    dimensions: List[int] = field(default_factory=list)
    
    @property
    def total_elements(self) -> int:
        result = 1
        for dim in self.dimensions:
            result *= dim
        return result
    
    @property
    def total_bits(self) -> int:
        return self.total_elements * self.element_width
    
    @property
    def num_dimensions(self) -> int:
        return len(self.dimensions)


@dataclass
class PartitionConfig:
    """Configuration for array partitioning.
    
    Attributes:
        partition_type: Type of partitioning
        dimension: Which dimension to partition (0-indexed)
        factor: Partitioning factor (for block/cyclic)
    """
    partition_type: PartitionType = PartitionType.NONE
    dimension: int = 0
    factor: int = 1


@dataclass
class MemoryBank:
    """A memory bank after partitioning.
    
    Attributes:
        name: Bank name
        array_name: Original array name
        bank_id: Bank identifier
        element_width: Bits per element
        depth: Number of elements in this bank
        addr_width: Address width
    """
    name: str
    array_name: str
    bank_id: int
    element_width: int
    depth: int
    
    @property
    def addr_width(self) -> int:
        if self.depth <= 1:
            return 1
        return ceil(log2(self.depth))


@dataclass
class PartitionedArray:
    """Result of array partitioning.
    
    Attributes:
        original: Original array info
        config: Partition configuration
        banks: List of memory banks
        address_map: Function to map original address to (bank, addr)
    """
    original: ArrayInfo
    config: PartitionConfig
    banks: List[MemoryBank] = field(default_factory=list)
    
    @property
    def num_banks(self) -> int:
        return len(self.banks)
    
    def get_bank_and_addr(self, indices: List[int]) -> Tuple[int, int]:
        """Map original array indices to bank and address.
        
        Args:
            indices: Original array indices
            
        Returns:
            Tuple of (bank_id, bank_address)
        """
        if self.config.partition_type == PartitionType.NONE:
            # Linear address
            addr = 0
            stride = 1
            for i in reversed(range(len(indices))):
                addr += indices[i] * stride
                stride *= self.original.dimensions[i]
            return 0, addr
        
        elif self.config.partition_type == PartitionType.COMPLETE:
            # Each element is its own "bank"
            addr = 0
            stride = 1
            for i in reversed(range(len(indices))):
                addr += indices[i] * stride
                stride *= self.original.dimensions[i]
            return addr, 0
        
        elif self.config.partition_type == PartitionType.CYCLIC:
            # Interleave based on partition dimension
            dim = self.config.dimension
            factor = self.config.factor
            
            bank_id = indices[dim] % factor
            
            # Compute address within bank
            new_indices = indices.copy()
            new_indices[dim] = indices[dim] // factor
            
            addr = 0
            stride = 1
            for i in reversed(range(len(new_indices))):
                dim_size = self.original.dimensions[i]
                if i == dim:
                    dim_size = ceil(dim_size / factor)
                addr += new_indices[i] * stride
                stride *= dim_size
            
            return bank_id, addr
        
        elif self.config.partition_type == PartitionType.BLOCK:
            # Contiguous blocks
            dim = self.config.dimension
            factor = self.config.factor
            dim_size = self.original.dimensions[dim]
            block_size = ceil(dim_size / factor)
            
            bank_id = indices[dim] // block_size
            
            # Compute address within bank
            new_indices = indices.copy()
            new_indices[dim] = indices[dim] % block_size
            
            addr = 0
            stride = 1
            for i in reversed(range(len(new_indices))):
                if i == dim:
                    addr += new_indices[i] * stride
                    stride *= block_size
                else:
                    addr += new_indices[i] * stride
                    stride *= self.original.dimensions[i]
            
            return bank_id, addr
        
        return 0, 0


class ArrayPartitioner:
    """Partitions arrays for increased memory bandwidth."""
    
    def partition(self, array: ArrayInfo, 
                  config: PartitionConfig) -> PartitionedArray:
        """Partition an array according to configuration.
        
        Args:
            array: Array to partition
            config: Partitioning configuration
            
        Returns:
            Partitioned array with memory banks
        """
        result = PartitionedArray(original=array, config=config)
        
        if config.partition_type == PartitionType.NONE:
            # Single bank with all elements
            bank = MemoryBank(
                name=f"{array.name}_bank0",
                array_name=array.name,
                bank_id=0,
                element_width=array.element_width,
                depth=array.total_elements
            )
            result.banks.append(bank)
        
        elif config.partition_type == PartitionType.COMPLETE:
            # Each element becomes a register
            for i in range(array.total_elements):
                bank = MemoryBank(
                    name=f"{array.name}_r{i}",
                    array_name=array.name,
                    bank_id=i,
                    element_width=array.element_width,
                    depth=1
                )
                result.banks.append(bank)
        
        elif config.partition_type == PartitionType.CYCLIC:
            # Interleaved banks
            factor = min(config.factor, array.dimensions[config.dimension])
            dim_size = array.dimensions[config.dimension]
            bank_depth = ceil(dim_size / factor)
            
            # Adjust for other dimensions
            other_size = array.total_elements // dim_size
            bank_depth *= other_size
            
            for i in range(factor):
                bank = MemoryBank(
                    name=f"{array.name}_bank{i}",
                    array_name=array.name,
                    bank_id=i,
                    element_width=array.element_width,
                    depth=bank_depth
                )
                result.banks.append(bank)
        
        elif config.partition_type == PartitionType.BLOCK:
            # Contiguous block banks
            factor = min(config.factor, array.dimensions[config.dimension])
            dim_size = array.dimensions[config.dimension]
            block_size = ceil(dim_size / factor)
            
            # Adjust for other dimensions
            other_size = array.total_elements // dim_size
            
            for i in range(factor):
                # Last bank might be smaller
                actual_block = min(block_size, dim_size - i * block_size)
                bank = MemoryBank(
                    name=f"{array.name}_bank{i}",
                    array_name=array.name,
                    bank_id=i,
                    element_width=array.element_width,
                    depth=actual_block * other_size
                )
                result.banks.append(bank)
        
        return result


@dataclass
class BufferConfig:
    """Configuration for buffer insertion.
    
    Attributes:
        depth: Buffer depth (number of entries)
        width: Data width
        is_fifo: True for FIFO, False for simple buffer
    """
    depth: int = 4
    width: int = 32
    is_fifo: bool = True


@dataclass
class MemoryBuffer:
    """A buffer for data staging.
    
    Attributes:
        name: Buffer name
        config: Buffer configuration
        input_signal: Input data signal
        output_signal: Output data signal
    """
    name: str
    config: BufferConfig
    input_signal: str
    output_signal: str


class MemorySVGenerator:
    """Generates SystemVerilog for memory structures."""
    
    def generate_partitioned_array(self, pa: PartitionedArray) -> str:
        """Generate SV for partitioned array.
        
        Args:
            pa: Partitioned array
            
        Returns:
            SystemVerilog code
        """
        lines = []
        
        lines.append(f"// Partitioned array: {pa.original.name}")
        lines.append(f"// Partition type: {pa.config.partition_type.name}")
        lines.append(f"// Number of banks: {pa.num_banks}")
        lines.append("")
        
        for bank in pa.banks:
            if pa.config.partition_type == PartitionType.COMPLETE:
                # Register
                lines.append(f"logic [{bank.element_width-1}:0] {bank.name};")
            else:
                # Memory array
                lines.append(f"logic [{bank.element_width-1}:0] {bank.name} [{bank.depth-1}:0];")
        
        return "\n".join(lines)
    
    def generate_buffer(self, buffer: MemoryBuffer) -> str:
        """Generate SV for a buffer/FIFO.
        
        Args:
            buffer: Buffer specification
            
        Returns:
            SystemVerilog code
        """
        lines = []
        cfg = buffer.config
        
        lines.append(f"// Buffer: {buffer.name}")
        lines.append(f"// Depth: {cfg.depth}, Width: {cfg.width}")
        lines.append("")
        
        if cfg.is_fifo:
            # FIFO implementation
            addr_width = ceil(log2(cfg.depth)) if cfg.depth > 1 else 1
            
            lines.append(f"logic [{cfg.width-1}:0] {buffer.name}_mem [{cfg.depth-1}:0];")
            lines.append(f"logic [{addr_width-1}:0] {buffer.name}_wr_ptr;")
            lines.append(f"logic [{addr_width-1}:0] {buffer.name}_rd_ptr;")
            lines.append(f"logic [{addr_width}:0] {buffer.name}_count;")
            lines.append("")
            lines.append(f"wire {buffer.name}_full = ({buffer.name}_count == {cfg.depth});")
            lines.append(f"wire {buffer.name}_empty = ({buffer.name}_count == 0);")
        else:
            # Simple buffer register
            lines.append(f"logic [{cfg.width-1}:0] {buffer.name}_data;")
            lines.append(f"logic {buffer.name}_valid;")
        
        return "\n".join(lines)


def partition_array(array: ArrayInfo, 
                    partition_type: PartitionType,
                    dimension: int = 0,
                    factor: int = 2) -> PartitionedArray:
    """Convenience function to partition an array.
    
    Args:
        array: Array to partition
        partition_type: Type of partitioning
        dimension: Dimension to partition
        factor: Partitioning factor
        
    Returns:
        Partitioned array
    """
    config = PartitionConfig(
        partition_type=partition_type,
        dimension=dimension,
        factor=factor
    )
    partitioner = ArrayPartitioner()
    return partitioner.partition(array, config)
