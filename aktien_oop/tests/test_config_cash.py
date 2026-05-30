from __future__ import annotations

import sys

from aktien_oop.config import Config


def test_config_from_cli_reads_cash_from_regime_section(tmp_path, monkeypatch):
    config_path = tmp_path / "runner_config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[regime]",
                "require_above_sma = true",
                'regime_below_action = "SELL"',
                "include_cash = true",
                "cash_yield_annual = 0.03",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["prog", "--config", str(config_path)])

    cfg = Config.from_cli()

    assert cfg.require_above_sma is True
    assert cfg.regime_below_action == "SELL"
    assert cfg.include_cash is True
    assert cfg.cash_yield_annual == 0.03
