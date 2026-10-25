"""Run-derived prose is a record of named runs, never an expectation (Kurt's ruling, 2026-10-26).

Runs: Colab T4 of blobs 1eaf44ab (2026-10-08), 59200cad and ecca54ad (2026-10-25).
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "tutorials" / "maxvit_classification_colab.ipynb"


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_timing_promises_are_gone():
    md = _markdown()
    for stale in ("the fine-tune about 33 s", "(about 33 s on a Colab T4)", "build took about 1 minute",
                  "in the recorded run, so the fine-tune"):
        assert stale not in md, stale


def test_confirming_run_of_ecca54ad_is_recorded():
    md = _markdown()
    assert "0.3344, 0.0239, 0.0046, 0.0018 and 0.0013" in md
    assert "45 s and 34.7 s in the run of blob `ecca54ad`" in md
    assert "(0.766, 0.837 and 0.758)" in md and "(0.509, 0.549 and 0.649)" in md
