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
SPRTL (Synchronous Process RTL) transformation module.

This module provides transformations from high-level async processes
to FSM-based RTL representations.

Key components:
- FSMState: Represents a single FSM state with operations and transitions
- FSMTransition: Represents a state transition with condition
- FSMModule: Complete FSM representation for a sync process
- SPRTLTransformer: Main transformation pass from IR to FSM
"""

from .fsm_ir import (
    FSMState,
    FSMStateKind,
    FSMTransition,
    FSMModule,
    FSMOperation,
    FSMAssign,
    FSMCond,
    FSMPort,
    FSMRegister,
)

from .transformer import SPRTLTransformer

from .scheduler import (
    OperationType,
    ScheduledOperation,
    Schedule,
    DependencyGraph,
    ASAPScheduler,
    ALAPScheduler,
    MobilityAnalyzer,
    FSMToScheduleGraphBuilder,
    # Resource-constrained scheduling
    PriorityMetric,
    ResourceConstraints,
    ResourceUsage,
    ListScheduler,
    SDCScheduler,
)

from .fsm_generator import (
    StateEncoding,
    FSMGeneratorConfig,
    LiveRange,
    FSMGenerator,
    RegisterAllocator,
    ScheduleToFSMBuilder,
)

from .sv_codegen import (
    ResetStyle,
    FSMStyle,
    SVGenConfig,
    SVCodeGenerator,
    generate_sv,
)

from .tb_codegen import (
    TVVector,
    TVSequence,
    TBGenConfig,
    TBGenerator,
    generate_testbench,
)

from .sva_gen import (
    AssertionType,
    PropertyKind,
    SVAProperty,
    SVAGenConfig,
    SVAGenerator,
    generate_sva,
)

from .optimizer import (
    OptimizationStats,
    DeadStateEliminator,
    StateMinimizer,
    TransitionOptimizer,
    OperationMerger,
    FSMOptimizer,
    optimize_fsm,
)

from .pipeline import (
    HazardType,
    PipelineStage,
    PipelineRegister,
    Pipeline,
    PipelineConfig,
    HazardDetector,
    PipelineScheduler,
    PipelineGenerator,
    PipelineSVGenerator,
    generate_pipeline,
)

from .memory import (
    PartitionType,
    ArrayInfo,
    PartitionConfig,
    MemoryBank,
    PartitionedArray,
    ArrayPartitioner,
    BufferConfig,
    MemoryBuffer,
    MemorySVGenerator,
    partition_array,
)

from .accessor_lowering import AccessorLowering
from .multiproc import (
    ChannelType,
    FlowControl,
    ChannelConfig,
    ChannelPort,
    Channel,
    ProcessInterface,
    ProcessNetwork,
    ChannelSVGenerator,
    NetworkSVGenerator,
    create_fifo_channel,
    create_handshake_channel,
)

__all__ = [
    'FSMState',
    'FSMStateKind',
    'FSMTransition',
    'FSMModule',
    'FSMOperation',
    'FSMAssign',
    'FSMCond',
    'FSMPort',
    'FSMRegister',
    'SPRTLTransformer',
    # Scheduler
    'OperationType',
    'ScheduledOperation',
    'Schedule',
    'DependencyGraph',
    'ASAPScheduler',
    'ALAPScheduler',
    'MobilityAnalyzer',
    'FSMToScheduleGraphBuilder',
    # Resource-constrained scheduling
    'PriorityMetric',
    'ResourceConstraints',
    'ResourceUsage',
    'ListScheduler',
    'SDCScheduler',
    # FSM Generator
    'StateEncoding',
    'FSMGeneratorConfig',
    'LiveRange',
    'FSMGenerator',
    'RegisterAllocator',
    'ScheduleToFSMBuilder',
    # SystemVerilog Code Generator
    'ResetStyle',
    'FSMStyle',
    'SVGenConfig',
    'SVCodeGenerator',
    'generate_sv',
    # Testbench Generator
    'TVVector',
    'TVSequence',
    'TBGenConfig',
    'TBGenerator',
    'generate_testbench',
    # SVA Assertion Generator
    'AssertionType',
    'PropertyKind',
    'SVAProperty',
    'SVAGenConfig',
    'SVAGenerator',
    'generate_sva',
    # FSM Optimizer
    'OptimizationStats',
    'DeadStateEliminator',
    'StateMinimizer',
    'TransitionOptimizer',
    'OperationMerger',
    'FSMOptimizer',
    'optimize_fsm',
    # Pipeline
    'HazardType',
    'PipelineStage',
    'PipelineRegister',
    'Pipeline',
    'PipelineConfig',
    'HazardDetector',
    'PipelineScheduler',
    'PipelineGenerator',
    'PipelineSVGenerator',
    'generate_pipeline',
    # Memory
    'PartitionType',
    'ArrayInfo',
    'PartitionConfig',
    'MemoryBank',
    'PartitionedArray',
    'ArrayPartitioner',
    'BufferConfig',
    'MemoryBuffer',
    'MemorySVGenerator',
    'partition_array',
    'AccessorLowering',
    # Multi-Process Communication
    'ChannelType',
    'FlowControl',
    'ChannelConfig',
    'ChannelPort',
    'Channel',
    'ProcessInterface',
    'ProcessNetwork',
    'ChannelSVGenerator',
    'NetworkSVGenerator',
    'create_fifo_channel',
    'create_handshake_channel',
]
