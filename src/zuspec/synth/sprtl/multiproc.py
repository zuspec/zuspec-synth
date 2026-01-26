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
Multi-Process Communication: FIFO channels and handshaking for HLS.

This module implements communication between concurrent processes:
- FIFO channel generation
- Handshake protocol synthesis
- Backpressure and flow control

These components enable safe communication between pipelined modules.
"""

from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from math import ceil, log2


class ChannelType(Enum):
    """Types of communication channels."""
    FIFO = auto()       # Buffered FIFO channel
    HANDSHAKE = auto()  # Simple valid/ready handshake
    STREAM = auto()     # AXI-Stream style interface


class FlowControl(Enum):
    """Flow control mechanisms."""
    NONE = auto()       # No flow control (may drop data)
    BLOCKING = auto()   # Block on full/empty
    CREDIT = auto()     # Credit-based flow control


@dataclass
class ChannelConfig:
    """Configuration for a communication channel.
    
    Attributes:
        channel_type: Type of channel
        data_width: Data width in bits
        depth: Buffer depth (for FIFO)
        flow_control: Flow control mechanism
        has_valid: Include valid signal
        has_ready: Include ready signal (backpressure)
    """
    channel_type: ChannelType = ChannelType.FIFO
    data_width: int = 32
    depth: int = 4
    flow_control: FlowControl = FlowControl.BLOCKING
    has_valid: bool = True
    has_ready: bool = True


@dataclass
class ChannelPort:
    """A port on one side of a channel.
    
    Attributes:
        name: Port name
        direction: 'producer' or 'consumer'
        data_signal: Data signal name
        valid_signal: Valid signal name
        ready_signal: Ready signal name
    """
    name: str
    direction: str  # 'producer' or 'consumer'
    data_signal: str = ""
    valid_signal: str = ""
    ready_signal: str = ""
    
    def __post_init__(self):
        if not self.data_signal:
            self.data_signal = f"{self.name}_data"
        if not self.valid_signal:
            self.valid_signal = f"{self.name}_valid"
        if not self.ready_signal:
            self.ready_signal = f"{self.name}_ready"


@dataclass
class Channel:
    """A communication channel between two processes.
    
    Attributes:
        name: Channel name
        config: Channel configuration
        producer: Producer port
        consumer: Consumer port
    """
    name: str
    config: ChannelConfig
    producer: ChannelPort = None
    consumer: ChannelPort = None
    
    def __post_init__(self):
        if self.producer is None:
            self.producer = ChannelPort(
                name=f"{self.name}_prod",
                direction="producer"
            )
        if self.consumer is None:
            self.consumer = ChannelPort(
                name=f"{self.name}_cons",
                direction="consumer"
            )


@dataclass
class ProcessInterface:
    """Interface definition for a process.
    
    Attributes:
        name: Process name
        input_channels: Channels consumed by this process
        output_channels: Channels produced by this process
        clock: Clock signal name
        reset: Reset signal name
    """
    name: str
    input_channels: List[str] = field(default_factory=list)
    output_channels: List[str] = field(default_factory=list)
    clock: str = "clk"
    reset: str = "rst_n"


@dataclass
class ProcessNetwork:
    """A network of communicating processes.
    
    Attributes:
        name: Network name
        processes: List of process interfaces
        channels: Channels connecting processes
    """
    name: str
    processes: List[ProcessInterface] = field(default_factory=list)
    channels: Dict[str, Channel] = field(default_factory=dict)
    
    def add_process(self, name: str) -> ProcessInterface:
        """Add a process to the network."""
        proc = ProcessInterface(name=name)
        self.processes.append(proc)
        return proc
    
    def add_channel(self, name: str, 
                    producer: str, consumer: str,
                    config: Optional[ChannelConfig] = None) -> Channel:
        """Add a channel between two processes.
        
        Args:
            name: Channel name
            producer: Producer process name
            consumer: Consumer process name
            config: Channel configuration
            
        Returns:
            Created channel
        """
        config = config or ChannelConfig()
        channel = Channel(name=name, config=config)
        self.channels[name] = channel
        
        # Update process interfaces
        for proc in self.processes:
            if proc.name == producer:
                proc.output_channels.append(name)
            if proc.name == consumer:
                proc.input_channels.append(name)
        
        return channel


class ChannelSVGenerator:
    """Generates SystemVerilog for communication channels."""
    
    def generate_fifo(self, channel: Channel) -> str:
        """Generate FIFO channel implementation.
        
        Args:
            channel: Channel to generate
            
        Returns:
            SystemVerilog code
        """
        cfg = channel.config
        lines = []
        
        name = channel.name
        width = cfg.data_width
        depth = cfg.depth
        addr_width = ceil(log2(depth)) if depth > 1 else 1
        
        lines.append(f"// FIFO Channel: {name}")
        lines.append(f"// Width: {width}, Depth: {depth}")
        lines.append("")
        lines.append(f"module {name}_fifo (")
        lines.append("  input  logic clk,")
        lines.append("  input  logic rst_n,")
        lines.append("  // Producer interface")
        lines.append(f"  input  logic [{width-1}:0] wr_data,")
        lines.append("  input  logic wr_valid,")
        lines.append("  output logic wr_ready,")
        lines.append("  // Consumer interface")
        lines.append(f"  output logic [{width-1}:0] rd_data,")
        lines.append("  output logic rd_valid,")
        lines.append("  input  logic rd_ready")
        lines.append(");")
        lines.append("")
        
        # Storage
        lines.append(f"  logic [{width-1}:0] mem [{depth-1}:0];")
        lines.append(f"  logic [{addr_width-1}:0] wr_ptr;")
        lines.append(f"  logic [{addr_width-1}:0] rd_ptr;")
        lines.append(f"  logic [{addr_width}:0] count;")
        lines.append("")
        
        # Status signals
        lines.append(f"  wire full = (count == {depth});")
        lines.append("  wire empty = (count == 0);")
        lines.append("")
        lines.append("  assign wr_ready = !full;")
        lines.append("  assign rd_valid = !empty;")
        lines.append("  assign rd_data = mem[rd_ptr];")
        lines.append("")
        
        # Write logic
        lines.append("  wire do_write = wr_valid && wr_ready;")
        lines.append("  wire do_read = rd_valid && rd_ready;")
        lines.append("")
        lines.append("  always_ff @(posedge clk or negedge rst_n) begin")
        lines.append("    if (!rst_n) begin")
        lines.append(f"      wr_ptr <= {addr_width}'d0;")
        lines.append(f"      rd_ptr <= {addr_width}'d0;")
        lines.append(f"      count <= {addr_width+1}'d0;")
        lines.append("    end else begin")
        lines.append("      if (do_write) begin")
        lines.append("        mem[wr_ptr] <= wr_data;")
        lines.append(f"        wr_ptr <= (wr_ptr == {depth-1}) ? {addr_width}'d0 : wr_ptr + 1;")
        lines.append("      end")
        lines.append("      if (do_read) begin")
        lines.append(f"        rd_ptr <= (rd_ptr == {depth-1}) ? {addr_width}'d0 : rd_ptr + 1;")
        lines.append("      end")
        lines.append("      case ({do_write, do_read})")
        lines.append("        2'b10: count <= count + 1;")
        lines.append("        2'b01: count <= count - 1;")
        lines.append("        default: ;")
        lines.append("      endcase")
        lines.append("    end")
        lines.append("  end")
        lines.append("")
        lines.append("endmodule")
        
        return "\n".join(lines)
    
    def generate_handshake(self, channel: Channel) -> str:
        """Generate simple handshake interface (no buffering).
        
        Args:
            channel: Channel to generate
            
        Returns:
            SystemVerilog code
        """
        cfg = channel.config
        lines = []
        
        name = channel.name
        width = cfg.data_width
        
        lines.append(f"// Handshake Channel: {name}")
        lines.append(f"// Width: {width}")
        lines.append("")
        lines.append(f"// Interface signals (directly connected):")
        lines.append(f"// Producer -> Consumer: {name}_data[{width-1}:0], {name}_valid")
        lines.append(f"// Consumer -> Producer: {name}_ready")
        lines.append("")
        lines.append("// Transfer occurs when valid && ready")
        
        return "\n".join(lines)
    
    def generate_channel(self, channel: Channel) -> str:
        """Generate channel based on type.
        
        Args:
            channel: Channel to generate
            
        Returns:
            SystemVerilog code
        """
        if channel.config.channel_type == ChannelType.FIFO:
            return self.generate_fifo(channel)
        elif channel.config.channel_type == ChannelType.HANDSHAKE:
            return self.generate_handshake(channel)
        else:
            return self.generate_fifo(channel)  # Default to FIFO


class NetworkSVGenerator:
    """Generates SystemVerilog for process networks."""
    
    def generate(self, network: ProcessNetwork) -> str:
        """Generate top-level module for process network.
        
        Args:
            network: Process network
            
        Returns:
            SystemVerilog code
        """
        lines = []
        
        lines.append(f"// Process Network: {network.name}")
        lines.append(f"// Processes: {len(network.processes)}")
        lines.append(f"// Channels: {len(network.channels)}")
        lines.append("")
        
        lines.append(f"module {network.name} (")
        lines.append("  input  logic clk,")
        lines.append("  input  logic rst_n")
        lines.append("  // Add external interface ports here")
        lines.append(");")
        lines.append("")
        
        # Generate channel wires
        lines.append("  // Channel signals")
        for ch_name, channel in network.channels.items():
            cfg = channel.config
            lines.append(f"  logic [{cfg.data_width-1}:0] {ch_name}_data;")
            if cfg.has_valid:
                lines.append(f"  logic {ch_name}_valid;")
            if cfg.has_ready:
                lines.append(f"  logic {ch_name}_ready;")
        lines.append("")
        
        # Generate channel FIFOs
        ch_gen = ChannelSVGenerator()
        for ch_name, channel in network.channels.items():
            if channel.config.channel_type == ChannelType.FIFO:
                lines.append(f"  // FIFO: {ch_name}")
                lines.append(f"  {ch_name}_fifo u_{ch_name}_fifo (")
                lines.append("    .clk(clk),")
                lines.append("    .rst_n(rst_n),")
                lines.append(f"    .wr_data({ch_name}_wr_data),")
                lines.append(f"    .wr_valid({ch_name}_wr_valid),")
                lines.append(f"    .wr_ready({ch_name}_wr_ready),")
                lines.append(f"    .rd_data({ch_name}_data),")
                lines.append(f"    .rd_valid({ch_name}_valid),")
                lines.append(f"    .rd_ready({ch_name}_ready)")
                lines.append("  );")
                lines.append("")
        
        # Process instantiation placeholders
        lines.append("  // Process instantiations")
        for proc in network.processes:
            lines.append(f"  // {proc.name} u_{proc.name} (...);")
        lines.append("")
        
        lines.append("endmodule")
        
        return "\n".join(lines)


def create_fifo_channel(name: str, width: int = 32, depth: int = 4) -> Channel:
    """Convenience function to create a FIFO channel.
    
    Args:
        name: Channel name
        width: Data width
        depth: FIFO depth
        
    Returns:
        Channel configuration
    """
    config = ChannelConfig(
        channel_type=ChannelType.FIFO,
        data_width=width,
        depth=depth,
        flow_control=FlowControl.BLOCKING,
        has_valid=True,
        has_ready=True
    )
    return Channel(name=name, config=config)


def create_handshake_channel(name: str, width: int = 32) -> Channel:
    """Convenience function to create a handshake channel.
    
    Args:
        name: Channel name
        width: Data width
        
    Returns:
        Channel configuration
    """
    config = ChannelConfig(
        channel_type=ChannelType.HANDSHAKE,
        data_width=width,
        depth=0,
        flow_control=FlowControl.BLOCKING,
        has_valid=True,
        has_ready=True
    )
    return Channel(name=name, config=config)
