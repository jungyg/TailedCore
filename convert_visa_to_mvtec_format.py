import os
import csv
import shutil
from tqdm import tqdm
import argparse


def restructure_visa(source_dir: str, target_dir: str, use_symlink: bool = True) -> None:
    """Restructure the VISA dataset into a MVTec‑style directory layout.

    Parameters
    ----------
    source_dir : str
        Path to the directory that contains the original ``visa`` folder.
    target_dir : str
        Path where the restructured ``visa_`` folder will be created.
    use_symlink : bool, optional
        If ``True`` (default) create symbolic links. If ``False`` copy the
        files instead. Copying is slower and uses disk space but works on
        platforms where symlinks are not permitted.
    """
    visa_src = source_dir
    visa_dst = target_dir

    os.makedirs(visa_dst, exist_ok=True)

    csv_path = os.path.join(visa_src, "split_csv", "1cls.csv")
    with open(csv_path, newline="") as file:
        reader = csv.reader(file)
        _ = next(reader)  # skip header line

        for _, row in tqdm(enumerate(reader), desc="Restructuring"):
            class_, split_, label, img_rel_path, mask_rel_path = row

            # File names
            img_name = os.path.basename(img_rel_path)
            mask_name = os.path.basename(mask_rel_path) if mask_rel_path else ""

            # Map VISA label → folder label
            label_dir = "good" if label == "normal" else "bad"

            # ---- image ----
            img_dst_dir = os.path.join(visa_dst, class_, split_, label_dir)
            os.makedirs(img_dst_dir, exist_ok=True)

            img_src_abs = os.path.join(visa_src, img_rel_path)
            img_dst_abs = os.path.join(img_dst_dir, img_name)

            # NOTE: We removed the rename from .jpg → .png to avoid corruption
            # if img_dst_abs.lower().endswith(".jpg"):
            #     img_dst_abs = img_dst_abs[:-4] + ".png"

            _link_or_copy(img_src_abs, img_dst_abs, use_symlink)

            # ---- mask (only for anomalous test samples) ----
            if not mask_rel_path:
                continue

            assert split_ == "test" and label_dir == "bad", (
                "Mask present on non‐anomalous or non‐test sample: "
                f"{class_}/{split_}/{label}")

            mask_dst_dir = os.path.join(visa_dst, class_, "ground_truth", label_dir)
            os.makedirs(mask_dst_dir, exist_ok=True)

            mask_src_abs = os.path.join(visa_src, mask_rel_path)
            mask_dst_abs = os.path.join(mask_dst_dir, mask_name)
            # We keep this rename so the mask has "_mask.png" suffix (MVTec style)
            if mask_dst_abs.lower().endswith(".png"):
                mask_dst_abs = mask_dst_abs[:-4] + "_mask.png"

            _link_or_copy(mask_src_abs, mask_dst_abs, use_symlink)


def _link_or_copy(src: str, dst: str, use_symlink: bool) -> None:
    """Create *dst* pointing to *src* via symlink or copy.

    If *dst* already exists, it is silently overwritten when copying, and left
    untouched when linking to avoid ``FileExistsError``.
    """
    if use_symlink:
        try:
            os.symlink(src, dst)
        except FileExistsError:
            # Keep the existing link to allow re‑running the script
            # without cleanup.
            pass
    else:
        shutil.copy2(src, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Restructure the VISA dataset into a MVTec‑style directory layout."
    )

    parser.add_argument(
        "-s", "--source_dir",
        default="./datasets/visa",
        help="Path to the original VISA dataset (default: ./visa)",
    )
    parser.add_argument(
        "-t", "--target_dir",
        default="./datasets/visa_",
        help="Destination directory for the restructured dataset (default: ./visa_)",
    )

    parser.add_argument(
        "--copy",
        default=True,
        help="Copy files instead of linking (default: use symlinks)",
    )

    args = parser.parse_args()
    # Decide based on user input
    args.use_symlink = not args.copy

    restructure_visa(args.source_dir, args.target_dir, use_symlink=args.use_symlink)