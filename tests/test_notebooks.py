"""Smoke checks that run in CI without data, GPUs or re-training.

* every notebook is valid nbformat v4;
* the headline numbers quoted in README.md are present in the saved
  notebook outputs (so the README cannot drift from the evidence);
* every image referenced from README.md exists in the repository.
"""
import pathlib
import re

import nbformat
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(
    p for p in ROOT.rglob("*.ipynb")
    if ".ipynb_checkpoints" not in p.parts and ".venv" not in p.parts
)

# (notebook path relative to the repo root, text that must appear in its saved outputs)
EXPECTED_OUTPUTS = [('NPS (1).ipynb', 'RangeIndex: 73398 entries, 0 to 73397'),
 ('NPS (1).ipynb',
  'Random Forest R2 Score: 0.7179366661953921 | Cross-Validation Mean R2: 0.7225291903138779'),
 ('NPS (1).ipynb',
  'Gradient Boosting R2 Score: 0.7250243443778491 | Cross-Validation Mean R2: 0.7270752068315474'),
 ('NPS (1).ipynb',
  'XGBoost R2 Score: 0.7251151783790073 | Cross-Validation Mean R2: 0.7270909705031887'),
 ('NPS (1).ipynb', 'Ridge Regression R2 Score: 0.6337185968581929'),
 ('NPS (1).ipynb', 'Lasso Regression R2 Score: 0.6337206309387745'),
 ('NPS (1).ipynb', 'R² Score: 0.6276\nRMSE: 0.2832'),
 ('NPS (1).ipynb', 'R² Score: 0.5297\nRMSE: 0.3183'),
 ('NPS (1).ipynb', '35 fits failed out of a total of 50.')]


def _outputs_text(path: pathlib.Path) -> str:
    nb = nbformat.read(path, as_version=4)
    chunks = []
    for cell in nb.cells:
        for out in cell.get("outputs", []):
            if "text" in out:
                chunks.append(out["text"])
            data = out.get("data", {})
            if "text/plain" in data:
                chunks.append(data["text/plain"])
    return "\n".join(chunks)


def test_notebooks_found():
    assert NOTEBOOKS, "no notebooks found"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: str(p.relative_to(ROOT)))
def test_notebook_is_valid_nbformat(path):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)


@pytest.mark.parametrize("notebook,snippet", EXPECTED_OUTPUTS)
def test_reported_result_is_in_notebook_output(notebook, snippet):
    assert snippet in _outputs_text(ROOT / notebook), f"{snippet!r} not found in outputs of {notebook}"


def test_readme_images_exist():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    refs = re.findall(r"\]\((docs/images/[^)\s]+)\)", readme) + re.findall(r'src="(docs/images/[^"]+)"', readme)
    assert refs, "README.md should show at least one image from docs/images/"
    missing = [r for r in refs if not (ROOT / r).is_file()]
    assert not missing, f"missing images: {missing}"
