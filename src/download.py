"""Fetch and extract the MSK-MET study from the cBioPortal DataHub.

The S3 bucket URL that appears in some documentation
(cbioportal-datahub.s3.amazonaws.com) returns 403; the working host is
datahub.assets.cbioportal.org, which is also what the portal deployment uses.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

DATAHUB = "https://datahub.assets.cbioportal.org/{study}.tar.gz"

TRAINING_STUDY = "msk_met_2021"

# Files the pipeline actually reads. Anything else in the tarball is ignored,
# but we assert these exist so a changed upstream layout fails loudly here
# rather than three stages later.
REQUIRED = (
    "data_clinical_patient.txt",
    "data_clinical_sample.txt",
    "data_mutations.txt",
    "data_cna.txt",
    "data_gene_panel_matrix.txt",
)


def download(study: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    tarball = dest / f"{study}.tar.gz"

    if tarball.exists():
        print(f"{tarball} exists ({tarball.stat().st_size:,} bytes), skipping download")
    else:
        url = DATAHUB.format(study=study)
        print(f"downloading {url}")
        tmp = tarball.with_suffix(".part")
        with urllib.request.urlopen(url) as response, tmp.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        tmp.rename(tarball)
        print(f"wrote {tarball} ({tarball.stat().st_size:,} bytes)")

    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    print(f"sha256 {digest}")

    study_dir = dest / study
    if study_dir.exists():
        print(f"{study_dir} exists, skipping extract")
    else:
        print(f"extracting into {dest}")
        with tarfile.open(tarball) as archive:
            # filter="data" refuses absolute paths and traversal outside dest.
            archive.extractall(dest, filter="data")

    missing = [name for name in REQUIRED if not (study_dir / name).exists()]
    if missing:
        raise SystemExit(
            f"extracted {study} is missing expected files: {', '.join(missing)}"
        )

    print(f"ok, {len(REQUIRED)} required files present in {study_dir}")
    return study_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default=TRAINING_STUDY)
    parser.add_argument("--dest", default="data", type=Path)
    args = parser.parse_args()
    download(args.study, args.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
