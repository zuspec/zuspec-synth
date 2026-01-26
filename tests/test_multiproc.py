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
"""Tests for multi-process communication."""

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

from zuspec.synth.sprtl.multiproc import (
    ChannelType, FlowControl, ChannelConfig, ChannelPort,
    Channel, ProcessInterface, ProcessNetwork,
    ChannelSVGenerator, NetworkSVGenerator,
    create_fifo_channel, create_handshake_channel
)


class TestChannelConfig:
    """Tests for channel configuration."""
    
    def test_default_config(self):
        """Test default channel config."""
        config = ChannelConfig()
        assert config.channel_type == ChannelType.FIFO
        assert config.data_width == 32
        assert config.depth == 4
    
    def test_custom_config(self):
        """Test custom channel config."""
        config = ChannelConfig(
            channel_type=ChannelType.HANDSHAKE,
            data_width=64,
            depth=0
        )
        assert config.channel_type == ChannelType.HANDSHAKE
        assert config.data_width == 64


class TestChannelPort:
    """Tests for channel ports."""
    
    def test_producer_port(self):
        """Test producer port creation."""
        port = ChannelPort(name="ch1", direction="producer")
        
        assert port.data_signal == "ch1_data"
        assert port.valid_signal == "ch1_valid"
        assert port.ready_signal == "ch1_ready"
    
    def test_consumer_port(self):
        """Test consumer port creation."""
        port = ChannelPort(name="ch1", direction="consumer")
        
        assert port.direction == "consumer"


class TestChannel:
    """Tests for communication channels."""
    
    def test_channel_creation(self):
        """Test creating a channel."""
        config = ChannelConfig(data_width=16, depth=8)
        channel = Channel(name="data_ch", config=config)
        
        assert channel.name == "data_ch"
        assert channel.producer is not None
        assert channel.consumer is not None
    
    def test_fifo_channel(self):
        """Test creating FIFO channel."""
        channel = create_fifo_channel("fifo1", width=32, depth=16)
        
        assert channel.config.channel_type == ChannelType.FIFO
        assert channel.config.data_width == 32
        assert channel.config.depth == 16
    
    def test_handshake_channel(self):
        """Test creating handshake channel."""
        channel = create_handshake_channel("hs1", width=64)
        
        assert channel.config.channel_type == ChannelType.HANDSHAKE
        assert channel.config.data_width == 64
        assert channel.config.depth == 0


class TestProcessInterface:
    """Tests for process interfaces."""
    
    def test_basic_interface(self):
        """Test basic process interface."""
        proc = ProcessInterface(name="producer")
        
        assert proc.name == "producer"
        assert proc.clock == "clk"
        assert proc.reset == "rst_n"
    
    def test_interface_with_channels(self):
        """Test process with channels."""
        proc = ProcessInterface(
            name="filter",
            input_channels=["in_ch"],
            output_channels=["out_ch"]
        )
        
        assert "in_ch" in proc.input_channels
        assert "out_ch" in proc.output_channels


class TestProcessNetwork:
    """Tests for process networks."""
    
    def test_empty_network(self):
        """Test creating empty network."""
        network = ProcessNetwork(name="test_net")
        
        assert network.name == "test_net"
        assert len(network.processes) == 0
        assert len(network.channels) == 0
    
    def test_add_process(self):
        """Test adding processes."""
        network = ProcessNetwork(name="test")
        
        p1 = network.add_process("producer")
        p2 = network.add_process("consumer")
        
        assert len(network.processes) == 2
        assert p1.name == "producer"
    
    def test_add_channel(self):
        """Test adding channel between processes."""
        network = ProcessNetwork(name="test")
        network.add_process("producer")
        network.add_process("consumer")
        
        ch = network.add_channel("data", "producer", "consumer")
        
        assert "data" in network.channels
        
        # Check process interfaces updated
        prod = next(p for p in network.processes if p.name == "producer")
        cons = next(p for p in network.processes if p.name == "consumer")
        assert "data" in prod.output_channels
        assert "data" in cons.input_channels


class TestChannelSVGenerator:
    """Tests for channel SystemVerilog generation."""
    
    def test_generate_fifo(self):
        """Test generating FIFO channel."""
        channel = create_fifo_channel("test_fifo", width=32, depth=8)
        
        generator = ChannelSVGenerator()
        code = generator.generate_fifo(channel)
        
        assert "module test_fifo_fifo" in code
        assert "wr_data" in code
        assert "rd_data" in code
        assert "wr_valid" in code
        assert "wr_ready" in code
        assert "full" in code
        assert "empty" in code
    
    def test_generate_handshake(self):
        """Test generating handshake channel."""
        channel = create_handshake_channel("test_hs", width=16)
        
        generator = ChannelSVGenerator()
        code = generator.generate_handshake(channel)
        
        assert "Handshake Channel: test_hs" in code
        assert "valid" in code.lower()
        assert "ready" in code.lower()
    
    def test_generate_channel_auto(self):
        """Test auto channel generation based on type."""
        fifo_ch = create_fifo_channel("ch1")
        hs_ch = create_handshake_channel("ch2")
        
        generator = ChannelSVGenerator()
        fifo_code = generator.generate_channel(fifo_ch)
        hs_code = generator.generate_channel(hs_ch)
        
        assert "module ch1_fifo" in fifo_code
        assert "Handshake" in hs_code


class TestNetworkSVGenerator:
    """Tests for network SystemVerilog generation."""
    
    def test_generate_simple_network(self):
        """Test generating simple process network."""
        network = ProcessNetwork(name="simple_net")
        network.add_process("producer")
        network.add_process("consumer")
        network.add_channel("data", "producer", "consumer")
        
        generator = NetworkSVGenerator()
        code = generator.generate(network)
        
        assert "module simple_net" in code
        assert "data_data" in code
        assert "data_valid" in code
        assert "data_ready" in code
    
    def test_generate_multi_channel_network(self):
        """Test generating network with multiple channels."""
        network = ProcessNetwork(name="multi_net")
        network.add_process("src")
        network.add_process("proc")
        network.add_process("sink")
        
        network.add_channel("ch1", "src", "proc")
        network.add_channel("ch2", "proc", "sink")
        
        generator = NetworkSVGenerator()
        code = generator.generate(network)
        
        assert "ch1_data" in code
        assert "ch2_data" in code
        assert "Processes: 3" in code
        assert "Channels: 2" in code


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_create_fifo_channel(self):
        """Test create_fifo_channel function."""
        ch = create_fifo_channel("my_fifo", 16, 32)
        
        assert ch.name == "my_fifo"
        assert ch.config.data_width == 16
        assert ch.config.depth == 32
        assert ch.config.has_valid
        assert ch.config.has_ready
    
    def test_create_handshake_channel(self):
        """Test create_handshake_channel function."""
        ch = create_handshake_channel("my_hs", 8)
        
        assert ch.name == "my_hs"
        assert ch.config.data_width == 8
        assert ch.config.has_valid
        assert ch.config.has_ready
