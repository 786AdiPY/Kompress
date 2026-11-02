"""Tests for Kompress CLI UI, banner, diagnostics, and entrypoints."""
import os
import sys
from io import StringIO
from unittest.mock import patch
import pytest

from kompress.cli.banner import (
    KOMPRESS_ASCII,
    TAGLINE,
    TURQUOISE,
    TURQUOISE_SHADES,
    get_plain_banner,
    render_rich_banner,
)
from kompress.cli.main import build_parser, main
from kompress.cli.ui import (
    VERSION,
    get_system_info,
    render_plain_ui,
    render_ui,
    run_diagnostics,
)


def test_banner_plain():
    banner = get_plain_banner("0.1.0")
    assert "KOMPRESS" in banner or "kompress" in banner.lower()
    assert TAGLINE in banner
    assert "0.1.0" in banner


def test_banner_rich():
    panel = render_rich_banner("0.1.0")
    assert panel is not None


def test_banner_turquoise():
    assert TURQUOISE == "#43deca"
    assert "#43deca" in TURQUOISE_SHADES
    panel = render_rich_banner("0.1.0")
    # Border style should be turquoise
    assert panel.border_style == TURQUOISE


def test_system_info():
    info = get_system_info()
    expected_keys = [
        "platform",
        "python",
        "torch",
        "accel",
        "onnxruntime",
        "mlflow",
        "objstore",
        "queue",
    ]
    for key in expected_keys:
        assert key in info
        assert isinstance(info[key], str)


def test_parser_build():
    parser = build_parser()
    assert parser.prog == "kompress"
    # parse test
    parsed = parser.parse_args(["--non-interactive"])
    assert parsed.non_interactive is True


def test_cli_version():
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0


def test_cli_banner_command(capsys):
    ret = main(["banner"])
    assert ret == 0


def test_cli_info_command():
    ret = main(["info"])
    assert ret == 0


def test_cli_targets_command():
    ret = main(["targets"])
    assert ret == 0


def test_cli_check_command():
    ret = main(["check"])
    assert ret == 0


def test_cli_non_interactive_ui():
    ret = main(["--non-interactive"])
    assert ret == 0

    ret_sub = main(["ui", "--non-interactive"])
    assert ret_sub == 0


def test_render_plain_ui(capsys):
    render_plain_ui()
    captured = capsys.readouterr()
    assert "Platform & Environment" in captured.out
    assert "kompress" in captured.out.lower()


def test_diagnostics():
    from rich.console import Console
    console = Console()
    ok = run_diagnostics(console)
    assert ok is True
