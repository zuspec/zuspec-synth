# zuspec-synth

ISA-agnostic multi-level synthesis (MLS) core for the Zuspec framework.

## Overview

`zuspec-synth` provides the core synthesis passes that transform a high-level
dataclass-based processor description into synthesisable SystemVerilog.  The
library is **ISA-agnostic**: it contains no RISC-V–specific logic.  All
architecture-specific stage generation (fetch, decode, execute, …) lives in
plugin passes supplied by the consumer (e.g. `zuspec-example-mls-riscv`).

## Core passes

| Pass | Description |
|---|---|
| `ElaboratePass` | Instantiate and elaborate the processor component |
| `FSMExtractPass` | Extract FSM/activity logic from dataclass actions |
| `SchedulePass` | Schedule operations onto pipeline stages |
| `LowerPass` | Lower IR to a form suitable for SV emission |
| `SVEmitPass` | Emit SystemVerilog from the lowered IR |
| `CertEmitPass` | Write a JSON synthesis certificate (deadlock check; ISA compliance optional) |

## Plugin extension points

Consumer packages inject ISA-specific behaviour by adding passes to the
`PassManager` chain **after** `LowerPass` and **before** `SVEmitPass`.

### `SynthIR.stage_sv`

The key extension point for stage generation.  A plugin pass (e.g.
`RVStageGeneratePass`) populates:

```python
ir.stage_sv[stage_name] = ["module FetchStage (", "  ...", ");", ...]
```

`SVEmitPass` (via `mls.py`) checks `ir.stage_sv[stage.name]` first; if
present, those lines are used verbatim.  Otherwise, a generic structural
placeholder module is emitted (useful for ISA-agnostic smoke tests).

### `SynthIR.lowered_sv`

Populated by domain-node lowering passes (e.g. `RVFieldLowerPass`,
`RVConstraintDecodeLowerPass`, `RVBodySynthLowerPass`) with per-unit SV
snippets that the stage generators incorporate.

## Typical pass chain

```python
ir = PassManager([
    ElaboratePass(MyCore, cfg),
    MyISAIdentifyPass(...),     # plugin: tag IR nodes with ISA semantics
    FSMExtractPass(synth_cfg),
    SchedulePass(synth_cfg),
    LowerPass(synth_cfg),
    MyISAStageIntroducePass(),  # plugin: add domain nodes
    MyISAFieldLowerPass(),      # plugin: lower field-extraction nodes
    MyISAStageGeneratePass(),   # plugin: populate ir.stage_sv
    SVEmitPass(synth_cfg, output_path),
]).run(SynthIR())
```

## Running tests

```bash
cd packages/zuspec-synth
python3 -m pytest tests/ -q
```


## CLI Plugin

`zuspec-synth` registers itself automatically with `zuspec-cli` when both packages are installed.

### Registered components

| Name | Type | Description |
|------|------|-------------|
| `synth` | Command | `zuspec synth` — synthesize a constraint-action class to RTL |
| `compute-support` | Transform | Compute active bit-ranges for each constraint |
| `build-cubes` | Transform | Enumerate minterms and build per-output truth tables |
| `odc` | Transform | Compute observability don't-cares |
| `minimize` | Transform | SOP minimisation via Quine-McCluskey |
| `python` | Frontend | Load a `@zdc.dataclass` action class by `module:Class` reference |

### Python frontend usage

```bash
# Synthesize a Python @zdc.dataclass action to RTL
zuspec synth --fe python mymodule:MyDecodeClass -o out.sv
```

The `--top` argument must be in `module:ClassName` form.  The module must be
importable (i.e. on `sys.path` or `PYTHONPATH`).

### Optimization flags

```bash
zuspec synth --fe python mymod:MyClass --no-odc --no-minimize -o out.sv
```

`--no-odc` skips the observability don't-care pass; `--no-minimize` skips
Quine-McCluskey minimisation (useful for debugging).
