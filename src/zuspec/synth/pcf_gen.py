"""
zuspec.synth.pcf_gen — Generate a Lattice .pcf file from a zdc.Component class.

Any port (input / output / inout) that carries a ``bind=zdc.pcf.io(pin=N)``
annotation is emitted as::

    set_io -nowarn <SIGNAL_NAME>  <PIN>

Usage (library)::

    from zuspec.synth.pcf_gen import gen_pcf
    print(gen_pcf(Icebreaker))

Usage (CLI)::

    python -m zuspec.synth.pcf_gen <module_path> <ClassName> [--output foo.pcf]
"""

from __future__ import annotations

import dataclasses as _dc
import textwrap
from typing import Optional, Type

import zuspec.dataclasses as zdc
from zuspec.dataclasses.decorators import Input, Output, Inout
from zuspec.dataclasses.pcf import io as PcfIo


def gen_pcf(cls: Type, header: Optional[str] = None) -> str:
    """Return the .pcf text for *cls*.

    Parameters
    ----------
    cls:
        A ``@zdc.dataclass`` component class whose port fields carry
        ``bind=zdc.pcf.io(pin=N)`` annotations.
    header:
        Optional comment block prepended to the output.  If *None* a
        default banner is generated.
    """
    if not hasattr(cls, "__dataclass_fields__"):
        raise TypeError(f"{cls!r} is not a @zdc.dataclass class")

    lines: list[str] = []

    if header is None:
        lines.append(f"# PCF generated from {cls.__name__}")
        lines.append("#")
    else:
        for ln in header.splitlines():
            lines.append(f"# {ln}" if not ln.startswith("#") else ln)

    port_markers = (Input, Output, Inout)
    entries: list[tuple[str, int]] = []  # (signal_name, pin)

    for f in _dc.fields(cls):
        if f.default_factory not in port_markers:
            continue
        bind = f.metadata.get("bind") if f.metadata else None
        if not isinstance(bind, PcfIo):
            continue
        entries.append((f.name, bind.pin))

    if not entries:
        lines.append("# (no pcf.io-annotated ports found)")
    else:
        max_name = max(len(name) for name, _ in entries)
        for name, pin in entries:
            lines.append(f"set_io -nowarn {name:<{max_name}}  {pin}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse
    import importlib.util
    import sys

    parser = argparse.ArgumentParser(
        description="Generate a .pcf file from a zdc.Component class."
    )
    parser.add_argument(
        "module",
        help="Path to the Python source file containing the component class "
             "(e.g. design/blinky/icebreaker.py)",
    )
    parser.add_argument(
        "classname",
        help="Name of the @zdc.dataclass component class inside the module.",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        default=None,
        help="Write output to FILE instead of stdout.",
    )
    parser.add_argument(
        "--search-path", "-I",
        metavar="DIR",
        action="append",
        default=[],
        help="Extra directories to add to sys.path before importing.",
    )
    args = parser.parse_args()

    for d in args.search_path:
        sys.path.insert(0, d)

    import os
    src_dir = os.path.dirname(os.path.abspath(args.module))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    spec = importlib.util.spec_from_file_location("_pcf_gen_target", args.module)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cls = getattr(mod, args.classname, None)
    if cls is None:
        print(f"error: class '{args.classname}' not found in {args.module}", file=sys.stderr)
        sys.exit(1)

    text = gen_pcf(cls)

    if args.output:
        with open(args.output, "w") as fh:
            fh.write(text)
        print(f"Wrote {args.output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    _main()
