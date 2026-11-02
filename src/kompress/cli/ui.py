"""Kompress CLI user interface rendering and interactive workflows."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Any

from kompress.cli.banner import (
    TAGLINE,
    TURQUOISE,
    get_plain_banner,
    render_rich_banner,
)

VERSION = "0.1.0"


def get_system_info() -> dict[str, str]:
    """Inspect environment, hardware accelerators, and storage configuration."""
    # PyTorch & Accelerators
    torch_info = "Not installed"
    accel_info = "CPU only"
    try:
        import torch
        torch_info = f"v{torch.__version__}"
        if torch.cuda.is_available():
            accel_info = f"NVIDIA CUDA ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            accel_info = "Apple MPS (Metal)"
    except ImportError:
        pass

    # ONNX Runtime
    ort_info = "Not installed"
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        prov_summary = ", ".join(p.replace("ExecutionProvider", "") for p in providers[:3])
        ort_info = f"v{ort.__version__} ({prov_summary})"
    except ImportError:
        pass

    # MLflow & Storage
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "local (./mlruns)")
    s3_url = os.environ.get("S3_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL") or "local filesystem"
    queue_url = os.environ.get("KOMPRESS_QUEUE", "filesystem (./queue)")

    return {
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": sys.version.split()[0],
        "torch": torch_info,
        "accel": accel_info,
        "onnxruntime": ort_info,
        "mlflow": mlflow_uri,
        "objstore": s3_url,
        "queue": queue_url,
    }


def render_ui(interactive: bool | None = None) -> None:
    """Render the full Kompress CLI UI."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.columns import Columns
        from rich.box import ROUNDED, SIMPLE_HEAVY

        console = Console()
        console.clear()
        console.print(render_rich_banner(VERSION))

        sys_info = get_system_info()

        # System & Platform Panel
        sys_table = Table(show_header=False, box=None, padding=(0, 1))
        sys_table.add_column("Key", style=f"bold {TURQUOISE}")
        sys_table.add_column("Value", style="white")

        sys_table.add_row("Operating System", sys_info["platform"])
        sys_table.add_row("Python Runtime", sys_info["python"])
        sys_table.add_row("Hardware Compute", sys_info["accel"])
        sys_table.add_row("PyTorch Engine", sys_info["torch"])
        sys_table.add_row("ONNX Runtime", sys_info["onnxruntime"])
        sys_table.add_row("MLflow Registry", sys_info["mlflow"])
        sys_table.add_row("Storage Backend", sys_info["objstore"])
        sys_table.add_row("Task Queue", sys_info["queue"])

        sys_panel = Panel(
            sys_table,
            title="[bold bright_green]Platform & Environment[/bold bright_green]",
            border_style="green",
            box=ROUNDED,
        )

        # Entry Gates Panel
        gates_table = Table(show_header=False, box=None, padding=(0, 1))
        gates_table.add_column("Gate", style="bold magenta")
        gates_table.add_column("Description", style="white")

        gates_table.add_row(
            "Gate A: Cloud & Self-Serve",
            "REST API (:8000) + React Dashboard (:5173)\n"
            "[dim]Review Plan deltas, approve & deploy on consent[/dim]",
        )
        gates_table.add_row(
            "Gate B: CLI & CI/CD Plugin",
            "Single-command manifest runner ('kompress-job')\n"
            "[dim]Automates compress → benchmark → gate in pipelines[/dim]",
        )
        gates_table.add_row(
            "Core Engine",
            "Adapters (XGBoost, sklearn, PyTorch) → ONNX\n"
            "[dim]Pluggable compressors, latency/size gating & device export[/dim]",
        )

        gates_panel = Panel(
            gates_table,
            title="[bold bright_magenta]Architecture & Entry Gates[/bold bright_magenta]",
            border_style="magenta",
            box=ROUNDED,
        )

        console.print(Columns([sys_panel, gates_panel], expand=True))

        # Hardware targets summary
        print_hardware_table(console)

        # Quick Actions or Menu
        is_tty = sys.stdin.isatty() if interactive is None else interactive
        if is_tty:
            run_interactive_loop(console)
        else:
            print_quick_commands(console)

    except ImportError:
        render_plain_ui()


def print_hardware_table(console: Any) -> None:
    """Print the table of hardware targets and mapped compression methods."""
    from rich.table import Table
    from rich.box import ROUNDED

    try:
        from kompress.engine.common.hardware import HARDWARE_COMPRESSORS
        from kompress.engine.export import HARDWARE_FORMAT
    except Exception:
        HARDWARE_COMPRESSORS = {
            "cpu-generic": ["onnx_fp32", "onnx_int8_dynamic", "onnx_int8_static"],
            "nvidia-gpu": ["onnx_fp32", "trt_int8"],
            "sony-imx500": ["onnx_fp32", "onnx_int8_static"],
            "arm-npu": ["onnx_fp32", "onnx_int8_static"],
        }
        HARDWARE_FORMAT = {
            "cpu-generic": "onnx",
            "arm-npu": "tflite",
            "sony-imx500": "onnx",
            "nvidia-gpu": "tensorrt",
        }

    table = Table(
        title="Hardware Deployment Targets & Automated Techniques",
        title_style=f"bold {TURQUOISE}",
        box=ROUNDED,
        border_style=TURQUOISE,
        header_style=f"bold white on dark_green",
    )
    table.add_column("Target ID", style=f"bold {TURQUOISE}", width=16)
    table.add_column("Applicable Compressors", style="white")
    table.add_column("Target Device Export", style="bright_yellow", width=18)
    table.add_column("Optimization Strategy", style="dim", width=28)

    descriptions = {
        "cpu-generic": "x86/ARM CPU without GPU; ONNX Runtime INT8 quantization",
        "nvidia-gpu": "NVIDIA TensorRT INT8 GPU engine with FP32 fallback",
        "sony-imx500": "Intelligent vision sensor NPU static quantization",
        "arm-npu": "Mobile & Edge accelerator static INT8 quantization",
    }

    for target, compressors in HARDWARE_COMPRESSORS.items():
        comp_str = " → ".join(f"[bold]{c}[/bold]" for c in compressors)
        export_fmt = HARDWARE_FORMAT.get(target, "onnx")
        desc = descriptions.get(target, "Automated hardware compression")
        table.add_row(target, comp_str, export_fmt.upper(), desc)

    console.print(table)


def print_quick_commands(console: Any) -> None:
    """Print non-interactive CLI usage help."""
    from rich.panel import Panel

    commands_md = (
        f"[bold {TURQUOISE}]Quick CLI Commands:[/bold {TURQUOISE}]\n"
        "  • [yellow]kompress[/yellow]                     Open this interactive CLI UI\n"
        "  • [yellow]kompress check[/yellow]               Run platform diagnostics & health check\n"
        "  • [yellow]kompress targets[/yellow]             List supported hardware targets & compressors\n"
        "  • [yellow]kompress test[/yellow]                Run platform smoke verification test\n"
        "  • [yellow]kompress run <job.yaml>[/yellow]     Execute a Gate B compression job manifest\n"
        "  • [yellow]kompress-api[/yellow]                 Start Gate A REST API backend (:8000)\n"
        "  • [yellow]kompress-worker[/yellow]              Start background queue worker\n"
    )
    console.print(Panel(commands_md, border_style="dim", title="CLI Quick Reference"))


def run_interactive_loop(console: Any) -> None:
    """Prompt user for interactive CLI menu actions."""
    from rich.prompt import Prompt
    from rich.panel import Panel

    while True:
        console.print("\n[bold bright_white]Select an Action:[/bold bright_white]")
        console.print(f"  [bold {TURQUOISE}]1[/bold {TURQUOISE}]) Run Platform Diagnostics & Health Check")
        console.print(f"  [bold {TURQUOISE}]2[/bold {TURQUOISE}]) View Hardware Targets & Device Exporters")
        console.print(f"  [bold {TURQUOISE}]3[/bold {TURQUOISE}]) Run Platform Smoke Test")
        console.print(f"  [bold {TURQUOISE}]4[/bold {TURQUOISE}]) Gate B Quickstart & Example Manifest")
        console.print(f"  [bold {TURQUOISE}]5[/bold {TURQUOISE}]) Gate A Cloud Stack & Service Launch Info")
        console.print(f"  [bold {TURQUOISE}]6[/bold {TURQUOISE}]) Refresh Dashboard View")
        console.print("  [bold red]q[/bold red]) Quit\n")

        choice = Prompt.ask(f"[bold {TURQUOISE}]Kompress[/bold {TURQUOISE}]", choices=["1", "2", "3", "4", "5", "6", "q"], default="q")

        if choice == "q":
            console.print("[dim]Exiting Kompress CLI. Happy compressing![/dim]\n")
            break
        elif choice == "1":
            run_diagnostics(console)
        elif choice == "2":
            print_hardware_table(console)
        elif choice == "3":
            run_smoke_test(console)
        elif choice == "4":
            show_gate_b_guide(console)
        elif choice == "5":
            show_gate_a_guide(console)
        elif choice == "6":
            console.clear()
            console.print(render_rich_banner(VERSION))
            sys_info = get_system_info()
            print_hardware_table(console)


def run_diagnostics(console: Any) -> bool:
    """Run diagnostics check on platform modules and schemas."""
    from rich.table import Table
    from rich.panel import Panel

    table = Table(title="Kompress Diagnostics & Health Check", title_style=f"bold {TURQUOISE}", border_style=TURQUOISE)
    table.add_column("Component", style="bold white")
    table.add_column("Status", style="bold")
    table.add_column("Details", style="dim")

    all_ok = True

    checks = [
        ("Engine Adapters", "kompress.engine.adapters", "Framework model adapters (sklearn, xgboost, pytorch)"),
        ("Engine Compressors", "kompress.engine.compressors", "Pluggable quantization & FP16 engines"),
        ("Export Targets", "kompress.engine.export", "Hardware deployable formats (ONNX, TRT, TFLite, CoreML)"),
        ("Registry & State", "kompress.engine.registry", "MLflow tag states & model version promotion"),
        ("Cloud Services", "kompress.services.cloud.api", "FastAPI Gate A REST API backend"),
    ]

    for name, module_path, desc in checks:
        try:
            __import__(module_path)
            table.add_row(name, "[green]✓ READY[/green]", desc)
        except Exception as e:
            all_ok = False
            table.add_row(name, "[red]✗ ERROR[/red]", f"{desc} ({e})")

    # Schemas check
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    job_schema = os.path.join(repo_root, "src/kompress/services/plugin/job.schema.json")
    report_schema = os.path.join(repo_root, "src/kompress/engine/pipeline/report.schema.json")

    if os.path.exists(job_schema) and os.path.exists(report_schema):
        table.add_row("JSON Schemas", "[green]✓ VALID[/green]", "job.schema.json & report.schema.json located")
    else:
        table.add_row("JSON Schemas", "[yellow]! WARNING[/yellow]", "One or more schema files missing")

    console.print(table)
    return all_ok


def run_smoke_test(console: Any) -> None:
    """Execute smoke_test.py directly and show results."""
    from rich.panel import Panel

    console.print("[bold yellow]Executing platform smoke test...[/bold yellow]")
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    smoke_script = os.path.join(repo_root, "tests", "smoke_test.py")

    if not os.path.exists(smoke_script):
        console.print(f"[red]Smoke test script not found at {smoke_script}[/red]")
        return

    res = subprocess.run([sys.executable, smoke_script], capture_output=True, text=True)
    if res.returncode == 0:
        console.print(Panel("[bold green]✓ PLATFORM SMOKE TEST PASSED[/bold green]\nEngine compression, benchmarking, accuracy gate, and report generation verified.", border_style="green"))
    else:
        console.print(Panel(f"[bold red]✗ SMOKE TEST FAILED (exit code {res.returncode})[/bold red]\n{res.stderr or res.stdout}", border_style="red"))


def show_gate_b_guide(console: Any) -> None:
    """Display Gate B manifest guide."""
    from rich.syntax import Syntax
    from rich.panel import Panel

    manifest_example = """# job.yaml — Kompress Gate B single-model manifest
model:
  name: "churn-classifier"
  ref: "artifacts/models/churn.pkl"      # local path or object store pointer
  framework: "xgboost"
  task: "binary_classification"
  target: "churn"

test_data:
  ref: "data/churn_test.csv"

target_hardware: "cpu-generic"             # cpu-generic | nvidia-gpu | arm-npu | sony-imx500

gate:
  metric: "f1"
  max_drop_pct: 2.0                       # fail gate if metric drops > 2%
"""
    syntax = Syntax(manifest_example, "yaml", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="Sample Gate B Manifest (job.yaml)", border_style="cyan"))
    console.print("[bold]Run with:[/bold] [yellow]kompress-job --job job.yaml --artifacts-dir artifacts[/yellow]\n")


def show_gate_a_guide(console: Any) -> None:
    """Display Gate A cloud platform guide."""
    from rich.panel import Panel

    msg = (
        "[bold cyan]Gate A: Cloud & Self-Serve Platform[/bold cyan]\n\n"
        "1. [bold]Start the Full Stack[/bold]:\n"
        "   [yellow]./run.sh[/yellow]                      # Starts MLflow + API + Worker + React Dashboard\n"
        "   [yellow]docker compose -f deploy/docker-compose.oss.yml up --build[/yellow] # Full OSS container stack\n\n"
        "2. [bold]Web Dashboard[/bold]:\n"
        "   Access the React UI at: [bold link=http://localhost:5173]http://localhost:5173[/bold link]\n\n"
        "3. [bold]REST API Docs[/bold]:\n"
        "   Interactive Swagger documentation: [bold link=http://localhost:8000/docs]http://localhost:8000/docs[/bold link]\n"
    )
    console.print(Panel(msg, title="Gate A Service Information", border_style="magenta"))


def render_plain_ui() -> None:
    """Plain-text fallback when rich is not available."""
    print(get_plain_banner(VERSION))
    info = get_system_info()
    print("\n--- Platform & Environment ---")
    for k, v in info.items():
        print(f"  {k:15}: {v}")
    print("\n--- Available Commands ---")
    print("  kompress                     Run CLI UI")
    print("  kompress check               Run platform health checks")
    print("  kompress targets             List hardware targets")
    print("  kompress-job                 Gate B compression job runner")
    print("  kompress-api                 Gate A REST API backend")
