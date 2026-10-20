"""Process lowering: clocked processes over bit-vector state -> FSM netlist.

Language-neutral: a front end (fw-hdl's SV elaborate-then-lower path, or any
other) builds a :class:`~zuspec.synth.proc.ir.Design`; :func:`lower` turns each
process into a state machine and resolves writes per bit; :func:`emit_verilog`
writes the result.
"""
from . import term
from .ir import (Assign, Await, Break, Conn, Continue, Design, If, Instance, Loop, Port,
                 Proc, Var, dump)

__all__ = ["term", "Assign", "Await", "Break", "Conn", "Continue", "Design", "If",
           "Instance", "Loop", "Port", "Proc", "Var", "dump"]
