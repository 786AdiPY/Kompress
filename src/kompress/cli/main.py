"""Kompress CLI main entrypoint."""
from __future__ import annotations

import argparse
import sys

from kompress.cli.banner import TURQUOISE, get_plain_banner, render_rich_banner
from kompress.cli.ui import (
    VERSION,
    get_system_info,
    print_hardware_table,
    render_plain_ui,
    render_ui,
    run_diagnostics,
    run_smoke_test,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kompress",
        description="Kompress: Model Compression & Hardware Optimization Platform CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run 'kompress' without arguments to launch the interactive CLI UI.",
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"kompress {VERSION}",
    )
    parser.add_argument(
        "-n", "--non-interactive",
        action="store_true",
        help="Run without prompting for interactive input",
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # kompress ui
    ui_parser = subparsers.add_parser("ui", help="Launch interactive CLI UI")
    ui_parser.add_argument(
        "-n", "--non-interactive",
        action="store_true",
        help="Render UI without interactive menu prompt",
    )

    # kompress banner
    subparsers.add_parser("banner", help="Print Kompress ASCII banner")

    # kompress info
    subparsers.add_parser("info", help="Display platform and environment diagnostics")

    # kompress targets
    subparsers.add_parser("targets", help="List supported hardware deployment targets and compressors")

    # kompress check
    subparsers.add_parser("check", help="Run platform health and dependency diagnostics")

    # kompress test
    subparsers.add_parser("test", help="Run platform smoke test")

    # kompress run <manifest>
    run_parser = subparsers.add_parser("run", help="Run a Gate B model compression job manifest")
    run_parser.add_argument(
        "--job", "-j",
        required=True,
        help="Path to job manifest YAML (e.g. job.yaml)",
    )
    run_parser.add_argument(
        "--artifacts-dir", "-a",
        default="artifacts",
        help="Artifacts output directory (default: artifacts)",
    )

    return parser


def main(args: list[str] | None = None) -> int:
    """CLI entrypoint function."""
    if args is None:
        args = sys.argv[1:]

    parser = build_parser()
    parsed = parser.parse_args(args)

    cmd = parsed.command

    if cmd is None or cmd == "ui":
        non_interactive = getattr(parsed, "non_interactive", False)
        render_ui(interactive=False if non_interactive else None)
        return 0

    if cmd == "banner":
        try:
            from rich.console import Console
            console = Console()
            console.print(render_rich_banner(VERSION))
        except ImportError:
            print(get_plain_banner(VERSION))
        return 0

    if cmd == "info":
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.box import ROUNDED

            console = Console()
            console.print(render_rich_banner(VERSION))
            info = get_system_info()
            table = Table(title="System & Platform Environment", title_style=f"bold {TURQUOISE}", box=ROUNDED, border_style=TURQUOISE)
            table.add_column("Property", style=f"bold {TURQUOISE}")
            table.add_column("Value", style="white")
            for k, v in info.items():
                table.add_row(k.replace("_", " ").title(), str(v))
            console.print(table)
        except ImportError:
            info = get_system_info()
            for k, v in info.items():
                print(f"{k}: {v}")
        return 0

    if cmd == "targets":
        try:
            from rich.console import Console
            console = Console()
            console.print(render_rich_banner(VERSION))
            print_hardware_table(console)
        except ImportError:
            render_plain_ui()
        return 0

    if cmd == "check":
        try:
            from rich.console import Console
            console = Console()
            ok = run_diagnostics(console)
            return 0 if ok else 1
        except ImportError:
            print("Running basic check...")
            return 0

    if cmd == "test":
        try:
            from rich.console import Console
            console = Console()
            run_smoke_test(console)
            return 0
        except ImportError:
            import subprocess
            return subprocess.call([sys.executable, "tests/smoke_test.py"])

    if cmd == "run":
        from kompress.services.plugin import run_job
        job_args = ["--job", parsed.job, "--artifacts-dir", parsed.artifacts_dir]
        return run_job.main(job_args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
