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
Pipeline: Pipelining support for HLS.

This module implements pipelining transformations for high-level synthesis:
- Modulo scheduling for loops
- Pipeline register insertion
- Valid/enable signal generation
- Hazard detection and resolution

Key concepts:
- Initiation Interval (II): Number of cycles between accepting new inputs
- Pipeline depth: Number of stages in the pipeline
- Hazards: RAW (Read-After-Write), WAR, WAW dependencies

Algorithm references:
- Modulo scheduling (Rau, "Iterative Modulo Scheduling")
- Adapted from XLS pipelining (Apache 2.0)
"""

from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from copy import deepcopy

from .fsm_ir import (
    FSMModule, FSMState, FSMStateKind, FSMTransition,
    FSMOperation, FSMAssign, FSMCond, FSMPort, FSMRegister
)
from .scheduler import (
    Schedule, DependencyGraph, ScheduledOperation, OperationType,
    ASAPScheduler, ALAPScheduler
)


class HazardType(Enum):
    """Types of pipeline hazards."""
    RAW = auto()  # Read-After-Write (true dependency)
    WAR = auto()  # Write-After-Read (anti-dependency)
    WAW = auto()  # Write-After-Write (output dependency)


@dataclass
class PipelineStage:
    """Represents a single pipeline stage.
    
    Attributes:
        stage_id: Stage number (0 = first stage)
        operations: Operations executed in this stage
        inputs: Signals read in this stage
        outputs: Signals written in this stage
        valid_signal: Name of valid signal for this stage
    """
    stage_id: int
    operations: List[FSMOperation] = field(default_factory=list)
    inputs: Set[str] = field(default_factory=set)
    outputs: Set[str] = field(default_factory=set)
    valid_signal: str = ""
    
    def __post_init__(self):
        if not self.valid_signal:
            self.valid_signal = f"stage{self.stage_id}_valid"


@dataclass
class PipelineRegister:
    """A register inserted between pipeline stages.
    
    Attributes:
        name: Register name
        width: Bit width
        source_stage: Stage that writes this register
        dest_stage: Stage that reads this register
        source_signal: Original signal name
    """
    name: str
    width: int
    source_stage: int
    dest_stage: int
    source_signal: str


@dataclass
class Pipeline:
    """Complete pipeline representation.
    
    Attributes:
        name: Pipeline name
        stages: List of pipeline stages
        registers: Pipeline registers between stages
        initiation_interval: Cycles between new inputs (II)
        latency: Total cycles from input to output
        input_ports: Pipeline input ports
        output_ports: Pipeline output ports
    """
    name: str
    stages: List[PipelineStage] = field(default_factory=list)
    registers: List[PipelineRegister] = field(default_factory=list)
    initiation_interval: int = 1
    latency: int = 0
    input_ports: List[FSMPort] = field(default_factory=list)
    output_ports: List[FSMPort] = field(default_factory=list)
    
    @property
    def num_stages(self) -> int:
        return len(self.stages)
    
    def add_stage(self) -> PipelineStage:
        """Add a new pipeline stage."""
        stage = PipelineStage(stage_id=len(self.stages))
        self.stages.append(stage)
        return stage
    
    def add_register(self, name: str, width: int, 
                     source_stage: int, dest_stage: int,
                     source_signal: str) -> PipelineRegister:
        """Add a pipeline register."""
        reg = PipelineRegister(
            name=name,
            width=width,
            source_stage=source_stage,
            dest_stage=dest_stage,
            source_signal=source_signal
        )
        self.registers.append(reg)
        return reg


@dataclass
class PipelineConfig:
    """Configuration for pipeline generation.
    
    Attributes:
        target_ii: Target initiation interval (1 = fully pipelined)
        max_stages: Maximum pipeline depth (0 = unlimited)
        insert_valid_signals: Generate valid signals per stage
        insert_ready_signals: Generate backpressure ready signals
        balance_stages: Balance operations across stages
    """
    target_ii: int = 1
    max_stages: int = 0
    insert_valid_signals: bool = True
    insert_ready_signals: bool = False
    balance_stages: bool = True


class HazardDetector:
    """Detects pipeline hazards between operations.
    
    Analyzes data dependencies to find RAW, WAR, and WAW hazards
    that must be resolved for correct pipelined execution.
    """
    
    def detect_hazards(self, graph: DependencyGraph, 
                       schedule: Schedule) -> List[Tuple[int, int, HazardType]]:
        """Detect all hazards in a scheduled graph.
        
        Args:
            graph: Dependency graph with operations
            schedule: Schedule mapping operations to times
            
        Returns:
            List of (producer_id, consumer_id, hazard_type) tuples
        """
        hazards = []
        
        # Track writes and reads per signal per time
        writes: Dict[str, List[Tuple[int, int]]] = {}  # signal -> [(time, op_id)]
        reads: Dict[str, List[Tuple[int, int]]] = {}
        
        for op_id, op in graph.operations.items():
            time = schedule.get_time(op_id)
            
            if op.source_op and isinstance(op.source_op, FSMAssign):
                target = op.source_op.target
                
                # Record write
                if target not in writes:
                    writes[target] = []
                writes[target].append((time, op_id))
                
                # Record reads from value
                read_signals = self._get_reads(op.source_op)
                for sig in read_signals:
                    if sig not in reads:
                        reads[sig] = []
                    reads[sig].append((time, op_id))
        
        # Detect RAW: read happens before write completes
        for signal, write_list in writes.items():
            if signal in reads:
                for read_time, read_op in reads[signal]:
                    for write_time, write_op in write_list:
                        if write_time < read_time:
                            # Check if write hasn't completed by read time
                            write_latency = graph.operations[write_op].latency
                            if write_time + write_latency > read_time:
                                hazards.append((write_op, read_op, HazardType.RAW))
        
        # Detect WAW: two writes to same signal
        for signal, write_list in writes.items():
            for i, (t1, op1) in enumerate(write_list):
                for t2, op2 in write_list[i+1:]:
                    if t1 != t2:
                        hazards.append((op1, op2, HazardType.WAW))
        
        return hazards

    def detect_regfile_hazards(self, regfile_decl) -> list:
        """Return regfile hazard pairs for a given RegFileDeclIR.

        Delegates to ``RegFileHazardAnalyzer`` so that pipeline scheduling
        can account for RAW forwarding and WAW stall requirements that arise
        from ``IndexedRegFile`` accesses.

        Args:
            regfile_decl: A ``RegFileDeclIR`` instance.

        Returns:
            List of ``RegFileHazardPair`` instances.
        """
        from .regfile_synth import RegFileHazardAnalyzer
        return RegFileHazardAnalyzer().analyze(regfile_decl)
    
    def _get_reads(self, op: FSMAssign) -> Set[str]:
        """Extract signals read by an assignment."""
        reads = set()
        value = op.value
        
        if isinstance(value, tuple) and len(value) == 3:
            target, _, _ = value
            if hasattr(target, 'name'):
                reads.add(target.name)
            elif hasattr(target, 'attr'):
                reads.add(target.attr)
        elif isinstance(value, str):
            reads.add(value)
        
        return reads


class PipelineScheduler:
    """Schedules operations for pipelined execution.
    
    Uses modulo scheduling to achieve target initiation interval
    while respecting resource constraints.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
    
    def schedule(self, graph: DependencyGraph) -> Tuple[Schedule, int]:
        """Schedule operations for pipelining.
        
        Args:
            graph: Dependency graph
            
        Returns:
            Tuple of (schedule, achieved_II)
        """
        # Start with ASAP schedule
        asap = ASAPScheduler()
        schedule = asap.schedule(graph)
        
        # Calculate minimum II based on resource constraints
        min_ii = self._calculate_min_ii(graph)
        
        # Try to achieve target II
        target_ii = max(self.config.target_ii, min_ii)
        
        # For II=1, ASAP schedule works if no resource conflicts
        # For II>1, use modulo scheduling
        if target_ii > 1:
            schedule = self._modulo_schedule(graph, target_ii)
        
        return schedule, target_ii
    
    def _calculate_min_ii(self, graph: DependencyGraph) -> int:
        """Calculate minimum possible II based on resources."""
        # Count operations by type
        op_counts: Dict[OperationType, int] = {}
        for op in graph.operations.values():
            op_type = op.op_type
            op_counts[op_type] = op_counts.get(op_type, 0) + 1
        
        # Assume 1 of each resource type for now
        # min_ii = max operations of any type
        if op_counts:
            return max(op_counts.values())
        return 1
    
    def _modulo_schedule(self, graph: DependencyGraph, ii: int) -> Schedule:
        """Perform modulo scheduling for given II.
        
        Modulo scheduling assigns operations to time slots modulo II,
        ensuring that when the loop repeats every II cycles,
        there are no resource conflicts.
        """
        schedule = Schedule()
        
        # Simple modulo scheduling: spread operations across II slots
        ops_by_type: Dict[OperationType, List[ScheduledOperation]] = {}
        for op in graph.operations.values():
            if op.op_type not in ops_by_type:
                ops_by_type[op.op_type] = []
            ops_by_type[op.op_type].append(op)
        
        # Assign each operation type to different modulo slots
        for op_type, ops in ops_by_type.items():
            for i, op in enumerate(ops):
                time = i % ii
                schedule.schedule_operation(op.id, time)
        
        return schedule


class PipelineGenerator:
    """Generates pipelined hardware from scheduled operations.
    
    Creates pipeline stages, inserts registers, and generates
    control signals (valid/ready).
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
    
    def generate(self, schedule: Schedule, graph: DependencyGraph,
                 source_fsm: Optional[FSMModule] = None) -> Pipeline:
        """Generate pipeline from schedule.
        
        Args:
            schedule: Operation schedule
            graph: Dependency graph
            source_fsm: Optional source FSM for port info
            
        Returns:
            Pipeline representation
        """
        pipeline = Pipeline(name=source_fsm.name if source_fsm else "pipeline")
        
        # Copy ports from source FSM
        if source_fsm:
            for port in source_fsm.ports:
                if port.direction == "input":
                    pipeline.input_ports.append(port)
                else:
                    pipeline.output_ports.append(port)
        
        # Create stages based on schedule
        max_time = schedule.total_latency
        for t in range(max_time):
            stage = pipeline.add_stage()
            
            # Add operations scheduled at this time
            ops_at_time = schedule.get_operations_at_time(t)
            for op_id in ops_at_time:
                op = graph.operations[op_id]
                if op.source_op:
                    stage.operations.append(op.source_op)
                    
                    # Track inputs/outputs
                    if isinstance(op.source_op, FSMAssign):
                        stage.outputs.add(op.source_op.target)
                        reads = self._get_reads(op.source_op)
                        stage.inputs.update(reads)
        
        # Insert pipeline registers
        self._insert_registers(pipeline, graph, schedule)
        
        # Set latency
        pipeline.latency = max_time
        
        return pipeline
    
    def _get_reads(self, op: FSMAssign) -> Set[str]:
        """Extract signals read by an assignment."""
        reads = set()
        value = op.value
        
        if isinstance(value, tuple) and len(value) == 3:
            target, _, _ = value
            if hasattr(target, 'name'):
                reads.add(target.name)
        elif isinstance(value, str):
            reads.add(value)
        
        return reads
    
    def _insert_registers(self, pipeline: Pipeline, 
                          graph: DependencyGraph, schedule: Schedule):
        """Insert pipeline registers between stages."""
        # For each signal that crosses stage boundaries, insert a register
        live_signals: Dict[str, int] = {}  # signal -> stage where defined
        
        for stage in pipeline.stages:
            # Check if any input was defined in earlier stage
            for sig in stage.inputs:
                if sig in live_signals:
                    def_stage = live_signals[sig]
                    if def_stage < stage.stage_id:
                        # Need pipeline register(s) to carry signal forward
                        for s in range(def_stage, stage.stage_id):
                            reg_name = f"{sig}_p{s}"
                            pipeline.add_register(
                                name=reg_name,
                                width=32,  # Default width
                                source_stage=s,
                                dest_stage=s + 1,
                                source_signal=sig
                            )
            
            # Record where signals are defined
            for sig in stage.outputs:
                live_signals[sig] = stage.stage_id


class PipelineSVGenerator:
    """Generates SystemVerilog code for a pipeline."""
    
    def generate(self, pipeline: Pipeline) -> str:
        """Generate SystemVerilog for pipeline.
        
        Args:
            pipeline: Pipeline to generate
            
        Returns:
            SystemVerilog code
        """
        lines = []
        
        lines.append(f"// Pipeline: {pipeline.name}")
        lines.append(f"// Stages: {pipeline.num_stages}, II: {pipeline.initiation_interval}, Latency: {pipeline.latency}")
        lines.append("")
        lines.append(f"module {pipeline.name}_pipe (")
        lines.append("  input  logic clk,")
        lines.append("  input  logic rst_n,")
        lines.append("  input  logic valid_in,")
        lines.append("  output logic valid_out,")
        
        # Input ports
        for port in pipeline.input_ports:
            if port.name not in ("clk", "rst_n"):
                width = f"[{port.width-1}:0] " if port.width > 1 else ""
                lines.append(f"  input  logic {width}{port.name},")
        
        # Output ports
        for i, port in enumerate(pipeline.output_ports):
            width = f"[{port.width-1}:0] " if port.width > 1 else ""
            comma = "," if i < len(pipeline.output_ports) - 1 else ""
            lines.append(f"  output logic {width}{port.name}{comma}")
        
        lines.append(");")
        lines.append("")
        
        # Valid signal chain
        lines.append("// Valid signal pipeline")
        for i in range(pipeline.num_stages):
            lines.append(f"logic stage{i}_valid;")
        lines.append("")
        
        # Pipeline registers
        if pipeline.registers:
            lines.append("// Pipeline registers")
            for reg in pipeline.registers:
                lines.append(f"logic [{reg.width-1}:0] {reg.name};")
            lines.append("")
        
        # Valid signal shift register
        lines.append("// Valid signal propagation")
        lines.append("always_ff @(posedge clk or negedge rst_n) begin")
        lines.append("  if (!rst_n) begin")
        for i in range(pipeline.num_stages):
            lines.append(f"    stage{i}_valid <= 1'b0;")
        lines.append("  end else begin")
        lines.append("    stage0_valid <= valid_in;")
        for i in range(1, pipeline.num_stages):
            lines.append(f"    stage{i}_valid <= stage{i-1}_valid;")
        lines.append("  end")
        lines.append("end")
        lines.append("")
        
        # Output valid
        if pipeline.num_stages > 0:
            lines.append(f"assign valid_out = stage{pipeline.num_stages-1}_valid;")
        else:
            lines.append("assign valid_out = valid_in;")
        lines.append("")
        
        # Pipeline register updates
        if pipeline.registers:
            lines.append("// Pipeline register updates")
            lines.append("always_ff @(posedge clk) begin")
            for reg in pipeline.registers:
                if reg.source_stage == 0:
                    lines.append(f"  if (valid_in) {reg.name} <= {reg.source_signal};")
                else:
                    prev_reg = f"{reg.source_signal}_p{reg.source_stage-1}" if reg.source_stage > 1 else reg.source_signal
                    lines.append(f"  if (stage{reg.source_stage-1}_valid) {reg.name} <= {prev_reg};")
            lines.append("end")
            lines.append("")
        
        # Stage logic (placeholder - actual operations would go here)
        for stage in pipeline.stages:
            if stage.operations:
                lines.append(f"// Stage {stage.stage_id} logic")
                lines.append(f"always_ff @(posedge clk) begin")
                lines.append(f"  if (stage{stage.stage_id}_valid) begin")
                for op in stage.operations:
                    if isinstance(op, FSMAssign):
                        lines.append(f"    // {op.target} <= ...;")
                lines.append("  end")
                lines.append("end")
                lines.append("")
        
        lines.append("endmodule")
        
        return "\n".join(lines)


def generate_pipeline(fsm: FSMModule, config: Optional[PipelineConfig] = None) -> Tuple[Pipeline, str]:
    """Convenience function to generate a pipeline from FSM.
    
    Args:
        fsm: Source FSM module
        config: Pipeline configuration
        
    Returns:
        Tuple of (Pipeline, SystemVerilog code)
    """
    from .scheduler import FSMToScheduleGraphBuilder
    
    # Build dependency graph from FSM
    builder = FSMToScheduleGraphBuilder()
    graph = builder.build(fsm)
    
    # Schedule for pipelining
    scheduler = PipelineScheduler(config)
    schedule, ii = scheduler.schedule(graph)
    
    # Generate pipeline
    generator = PipelineGenerator(config)
    pipeline = generator.generate(schedule, graph, fsm)
    pipeline.initiation_interval = ii
    
    # Generate SystemVerilog
    sv_gen = PipelineSVGenerator()
    sv_code = sv_gen.generate(pipeline)
    
    return pipeline, sv_code
