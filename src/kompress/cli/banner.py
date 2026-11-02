"""Banner art and visual styling for Kompress CLI."""
from __future__ import annotations

import os

KOMPRESS_ASCII = r"""    __ __                                        
   / //_/___  ____ ___  ____  ________  __________
  / ,< / __ \/ __ `__ \/ __ \/ ___/ _ \/ ___/ ___/
 / /| / /_/ / / / / / / /_/ / /  /  __(__  |__  ) 
/_/ |_\____/_/ /_/ /_/ .___/_/   \___/____/____/  
                    /_/                           """

TITLE = "KOMPRESS"
TAGLINE = "Model Compression & Hardware Optimization Platform"
SUBTITLE = "Gate A (Cloud & Self-Serve) • Gate B (CLI & CI/CD Orchestration)"

# Turquoise color palette matching #43deca
TURQUOISE = "#43deca"
TURQUOISE_SHADES = ["#43deca", "#3fd4c2", "#48d1cc", "#4ee3d0", "#40e0d0", "#36ad99"]


def get_plain_banner(version: str = "0.1.0") -> str:
    """Return plain ANSI or text banner with turquoise coloring."""
    use_color = os.environ.get("NO_COLOR") is None
    if use_color:
        # 24-bit ANSI truecolor for #43deca (RGB: 67, 222, 202)
        turquoise = "\033[38;2;67;222;202m"
        bold = "\033[1m"
        dim = "\033[2m"
        reset = "\033[0m"
        line = "─" * 60
        return (
            f"{turquoise}{bold}{KOMPRESS_ASCII}{reset}\n"
            f"{dim}{line}{reset}\n"
            f"{bold}  KOMPRESS — {TAGLINE} (v{version}){reset}\n"
            f"{dim}  {SUBTITLE}{reset}\n"
            f"{dim}{line}{reset}"
        )
    line = "-" * 60
    return (
        f"{KOMPRESS_ASCII}\n"
        f"{line}\n"
        f"  KOMPRESS — {TAGLINE} (v{version})\n"
        f"  {SUBTITLE}\n"
        f"{line}"
    )


def render_rich_banner(version: str = "0.1.0"):
    """Return a Rich renderable (Panel) for the Kompress banner with turquoise logo styling."""
    try:
        from rich.align import Align
        from rich.box import ROUNDED
        from rich.panel import Panel
        from rich.text import Text

        styled_text = Text()
        lines = KOMPRESS_ASCII.split("\n")
        for i, line in enumerate(lines):
            color = TURQUOISE_SHADES[i % len(TURQUOISE_SHADES)]
            styled_text.append(line + "\n", style=f"bold {color}")

        styled_text.append(f"\n{TAGLINE}\n", style="bold white")
        styled_text.append(f"{SUBTITLE}", style="dim")

        panel = Panel(
            Align.center(styled_text),
            box=ROUNDED,
            border_style=TURQUOISE,
            title=f"[bold {TURQUOISE}]KOMPRESS CLI[/bold {TURQUOISE}]",
            subtitle=f"[dim]v{version} • Open Source Architecture[/dim]",
            padding=(1, 2),
        )
        return panel
    except ImportError:
        return get_plain_banner(version)
