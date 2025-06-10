#!/usr/bin/env python3
"""
make_longtail_noisy_mvtecad.py
==============================
Produce a **long‑tail noisy** variant of the MVTec‑AD dataset using **two
manifest files**:

* **`--noisy-manifest`** — relative paths of *defect* images that should be
  injected into each object class' `train/good` folder.
* **`--prune-manifest`** — relative paths of *normal* (original good) images
  that should be **removed** from `train/good` to yield an imbalanced, long‑tail
  distribution.

Inputs
------
```
--source-dir      pristine MVTec‑AD root (contains full `test` & `ground_truth`)
--dest-dir        destination root where the long‑tail noisy dataset is built
--noisy-manifest  text file listing defect images to inject (relative to dest)
--prune-manifest  text file listing good images to delete (relative to dest)
--symlink         (optional) use symbolic links for injected images instead of copying
```

Behaviour
---------
1. Ensures `train/good`, `test`, and `ground_truth` under *dest* mirror those
   in *source* (they are copied only if missing).
2. **Deletes** every path listed in *prune‑manifest* from the destination.
3. **Injects** every path in *noisy‑manifest* by copying/symlinking the
   corresponding file from
   `source/<object>/test/<defect>/<filename>` to
   `dest/<object>/train/good/<defect>_<filename>`.

Both manifests must list paths **relative to the destination root**, matching
what `make_noisy_mvtecad.py` produced (e.g.
`bottle/train/good/broken_small_001.png`).
"""
import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List

###############################################################################
# Helpers
###############################################################################

def copy_or_link(src: Path, dst: Path, symlink: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if symlink:
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)


def replicate_split(src_root: Path, dst_root: Path, split: str):
    """Copy entire `split` folder (test or ground_truth) object‑wise if missing."""
    for obj in sorted(p.name for p in src_root.iterdir() if p.is_dir()):
        src_split = src_root / obj / split
        if not src_split.is_dir():
            continue
        dst_split = dst_root / obj / split
        shutil.copytree(src_split, dst_split, dirs_exist_ok=True)


def gather_defect_dirs(src_root: Path) -> Dict[str, List[str]]:
    """Return {object → list of defect category names}, longest first"""
    mapping: Dict[str, List[str]] = {}
    for obj in sorted(p.name for p in src_root.iterdir() if p.is_dir()):
        test_dir = src_root / obj / "test"
        if not test_dir.is_dir():
            continue
        defects = sorted((d.name for d in test_dir.iterdir() if d.is_dir()), key=len, reverse=True)
        mapping[obj] = defects
    return mapping

###############################################################################
# Core steps
###############################################################################

def prune_good_samples(dest: Path, prune_manifest: Path):
    if not prune_manifest.is_file():
        raise FileNotFoundError(f"Prune manifest not found: {prune_manifest}")
    removed, missing = 0, 0
    with open(prune_manifest) as f:
        for line in f:
            rel = line.strip()
            if not rel:
                continue
            target = dest / rel
            if target.is_file():
                target.unlink()
                removed += 1
            else:
                missing += 1
    print(f"✔ Removed {removed} original good images (listed in prune manifest)")
    if missing:
        print(f"⚠ {missing} paths from prune manifest were not found (already absent)")


def inject_defect_samples(source: Path, dest: Path, noisy_manifest: Path, symlink: bool):
    if not noisy_manifest.is_file():
        raise FileNotFoundError(f"Noisy manifest not found: {noisy_manifest}")

    defect_map = gather_defect_dirs(source)
    injected, missing_sources = 0, []

    with open(noisy_manifest) as f:
        for rel_dst in (ln.strip() for ln in f if ln.strip()):
            parts = Path(rel_dst).parts
            if len(parts) < 4 or parts[1:3] != ("train", "good"):
                raise ValueError(f"Malformed noisy manifest entry: {rel_dst}")
            obj = parts[0]
            filename = parts[3]
            defects = defect_map.get(obj, [])
            defect = next((d for d in defects if filename.startswith(f"{d}_")), None)
            if defect is None:
                raise ValueError(f"Cannot determine defect for entry: {rel_dst}")
            original_name = filename[len(defect) + 1 :]
            src_img = source / obj / "test" / defect / original_name
            if not src_img.is_file():
                missing_sources.append(str(src_img))
                continue
            dst_img = dest / rel_dst
            copy_or_link(src_img, dst_img, symlink)
            injected += 1

    if missing_sources:
        preview = "\n".join(missing_sources[:10])
        more = "..." if len(missing_sources) > 10 else ""
        raise FileNotFoundError(
            f"{len(missing_sources)} source images missing while injecting:\n{preview}{more}"
        )
    print(f"✔ Injected {injected} defect images (listed in noisy manifest)")

###############################################################################
# CLI
###############################################################################

def main():
    p = argparse.ArgumentParser(description="Build a long‑tail noisy MVTec‑AD dataset from two manifests.")
    p.add_argument("--source-dir", required=True, type=Path, help="Pristine MVTec‑AD root directory")
    p.add_argument("--dest-dir", required=True, type=Path, help="Output root for the long‑tail noisy dataset")
    p.add_argument("--noisy-manifest", required=True, type=Path, help="Manifest of defect images to inject")
    p.add_argument("--prune-manifest", required=True, type=Path, help="Manifest of original good images to delete")
    p.add_argument("--symlink", action="store_true", help="Use symbolic links instead of copying when injecting defects")
    args = p.parse_args()

    # 1. Ensure pristine structure in dest
    for obj_path in sorted(
        [d for d in args.source_dir.iterdir() if (d / "train").is_dir()]):
        obj_name = obj_path.name                    # keep the name once, as str
        src_good = obj_path / "train" / "good"      # pure Path arithmetic
        dst_good = args.dest_dir / obj_name / "train" / "good"
        shutil.copytree(src_good, dst_good, dirs_exist_ok=True)
    replicate_split(args.source_dir, args.dest_dir, "test")
    replicate_split(args.source_dir, args.dest_dir, "ground_truth")

    # 2. Prune specified good samples
    prune_good_samples(args.dest_dir, args.prune_manifest)

    # 3. Inject defect samples
    inject_defect_samples(args.source_dir, args.dest_dir, args.noisy_manifest, args.symlink)

    print(f"✔ Long‑tail noisy dataset ready at {args.dest_dir}")


if __name__ == "__main__":
    main()
