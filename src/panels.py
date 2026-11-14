"""Gene panel membership, so an untested gene is never coded as wild type.

MSK-MET mixes three IMPACT panel versions:

    IMPACT468   17,602 samples   68.3%    469 genes
    IMPACT410    6,372 samples   24.7%    410 genes
    IMPACT341    1,801 samples    7.0%    341 genes

Served studies can use others. msk_chord_2024 adds IMPACT505 (505 genes) and
IMPACT-HEME-400 (400 genes), so those are fetched and stored in the schema too
even though no training patient was sequenced on them. A panel the schema does
not know means no gene can be marked untested, which quietly turns "never
looked at" into "looked at and found nothing" for every gene off that panel.

A gene that is not on the panel a given sample was sequenced with was never
looked at. Coding it 0 would assert "no mutation found" for roughly a third of
the cohort on genes nobody tested, so those cells become NaN instead and
XGBoost routes them down its missing branch.

The panel definitions are not in the study tarball. They come from the public
cBioPortal API and are cached next to the model artefacts so training is
reproducible without network access.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

PANEL_API = "https://www.cbioportal.org/api/gene-panels/{panel}"

DEFAULT_CACHE = Path("artifacts/gene_panels.json")


def fetch_panels(names: list[str]) -> dict[str, list[str]]:
    panels: dict[str, list[str]] = {}
    for name in names:
        with urllib.request.urlopen(PANEL_API.format(panel=name)) as response:
            payload = json.load(response)
        panels[name] = sorted(
            gene["hugoGeneSymbol"] for gene in payload.get("genes", [])
        )
    return panels


def load_panels(names: list[str], cache: Path = DEFAULT_CACHE) -> dict[str, list[str]]:
    """Return panel -> gene list, fetching and caching on first use."""
    if cache.exists():
        cached = json.loads(cache.read_text())
        if all(name in cached for name in names):
            return {name: cached[name] for name in names}

    panels = fetch_panels(names)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(panels, indent=2, sort_keys=True))
    return panels


def panel_sets(panels: dict[str, list[str]]) -> dict[str, frozenset[str]]:
    return {name: frozenset(genes) for name, genes in panels.items()}


def describe(panels: dict[str, list[str]]) -> str:
    """Human-readable nesting check, for the report."""
    names = sorted(panels, key=lambda n: len(panels[n]))
    lines = [f"{name}: {len(panels[name])} genes" for name in names]
    for smaller, larger in zip(names, names[1:]):
        missing = set(panels[smaller]) - set(panels[larger])
        relation = "subset of" if not missing else f"NOT a subset of ({len(missing)} extra)"
        lines.append(f"{smaller} is {relation} {larger}")
    return "\n".join(lines)


if __name__ == "__main__":
    loaded = load_panels(["IMPACT341", "IMPACT410", "IMPACT468"])
    print(describe(loaded))
