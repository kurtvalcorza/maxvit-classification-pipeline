"""Final prose pass (Kurt's ruling, 2026-10-26): the flagged expectation sentences are removed or
rewritten as records of named runs. maxvit: Section 9 worked answer.
Fails on PR head 9259fc0.
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / 'tutorials/maxvit_classification_colab.ipynb'


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


GONE = [
    'Another run can differ by an image',
]
PRESENT = [
    'its two-image edge over the zero-shot mapping is not evidence of a gain.</details>',
]


def test_flagged_expectations_are_gone():
    md = _markdown()
    for text in GONE:
        assert text not in md, text


def test_replacement_records_are_present():
    md = _markdown()
    for text in PRESENT:
        assert text in md, text
