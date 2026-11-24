import json

import pytest

from rlvr.cli import METHODS, main, summarize, to_markdown


def test_cli_end_to_end_tiny(tmp_path):
    main(
        [
            "--methods",
            "grpo",
            "dpo",
            "sft_more",
            "--seeds",
            "0",
            "--digits",
            "1",
            "--sft-steps",
            "5",
            "--rl-steps",
            "2",
            "--n-layer",
            "1",
            "--d-model",
            "32",
            "--out",
            str(tmp_path),
        ]
    )
    data = json.loads((tmp_path / "addition1.json").read_text())
    methods = [r["method"] for r in data["summary"]]
    assert methods == ["sft_base", "grpo", "dpo", "sft_more"]
    assert (tmp_path / "addition1.md").read_text().startswith("| method |")


def test_summarize_and_markdown():
    runs = [
        {"method": "a", "greedy_acc": 0.5, "pass@1": 0.4, "pass@16": 0.9},
        {"method": "a", "greedy_acc": 0.7, "pass@1": 0.6, "pass@16": 1.0},
    ]
    rows = summarize(runs)
    assert rows[0]["greedy_acc"] == pytest.approx(0.6)
    assert "60.0 ± 14.1" in to_markdown(rows)


def test_unknown_method_rejected():
    with pytest.raises(SystemExit):
        main(["--methods", "nope"])


def test_all_methods_have_valid_kind():
    assert all(spec["kind"] in {"grpo", "dpo", "sft"} for spec in METHODS.values())
