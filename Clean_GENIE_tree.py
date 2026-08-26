"""
Walks a source directory tree and builds a new mirrored tree made of
symlinks, cleaned according to the rules below. Nothing in the source
tree is touched, moved, or deleted.

Expected structure at each "branch" level (the level this script targets):

    .../some/branch/
        2024NoGSF/
            something.root
            something.log
        branch.log
        branch.root

After cleaning, the corresponding location in the destination tree will
contain only:

    .../some/branch/
        something.root   (symlink)

Usage:
    python Clean_GENIE_tree.py /path/to/source /path/to/destination
    python Clean_GENIE_tree.py /path/to/source /path/to/destination --dry-run
"""

import argparse
import os
import sys
from pathlib import Path

# Pattern used to find the root file to keep inside 2024NoGSF.
ROOT_FILE_GLOB = "*.root"

def find_branch_dirs(src_root: Path):
    """Yield every directory in src_root that looks like a 'branch':
    i.e. it directly contains a 2024NoGSF subdirectory.
    """
    for dirpath, dirnames, _filenames in os.walk(src_root):
        if "2024NoGSF" in dirnames:
            yield Path(dirpath)
            # nothing below a branch needs separate handling.
            dirnames[:] = []


def make_symlink(link_path: Path, target_path: Path, dry_run: bool):
    if dry_run:
        print(f"[dry-run] symlink: {link_path}  ->  {target_path}")
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        print(f"  skipping (already exists): {link_path}")
        return
    link_path.symlink_to(target_path)


def process_branch(branch_src: Path, src_root: Path, dst_root: Path, dry_run: bool):
    rel = branch_src.relative_to(src_root)
    branch_dst = dst_root / rel

    pre_src = branch_src / "2024NoGSF"
    root_files = sorted(pre_src.glob(ROOT_FILE_GLOB))

    if not root_files:
        print(f"  WARNING: no root file found in {pre_src}, skipping this branch")
        return
    if len(root_files) > 1:
        print(f"  WARNING: multiple root files found in {pre_src}, keeping all of them")

    for root_file in root_files:
        # Use resolve() so the symlink target is an absolute, real path,
        # robust to the destination tree being moved elsewhere later.
        target = root_file.resolve()
        link_path = branch_dst / root_file.name

        print(f"  keep: {root_file}")
        make_symlink(link_path, target, dry_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to the existing directory tree")
    parser.add_argument("destination", type=Path, help="Path to the new (cleaned) tree to create")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without creating any files or directories",
    )
    args = parser.parse_args()

    src_root = args.source.resolve()
    dst_root = args.destination.resolve()

    if not src_root.is_dir():
        sys.exit(f"Source directory does not exist: {src_root}")

    if dst_root.exists() and any(dst_root.iterdir()):
        sys.exit(
            f"Destination directory already exists and is not empty: {dst_root}\n"
            "Refusing to write into it -- choose an empty/new path."
        )

    if not args.dry_run:
        dst_root.mkdir(parents=True, exist_ok=True)

    branch_count = 0
    for branch_src in find_branch_dirs(src_root):
        print(f"Branch: {branch_src}")
        process_branch(branch_src, src_root, dst_root, args.dry_run)
        branch_count += 1

    print(f"\nDone. Processed {branch_count} branch director{'y' if branch_count == 1 else 'ies'}.")
    if args.dry_run:
        print("(dry run -- nothing was actually created)")


if __name__ == "__main__":
    main()
