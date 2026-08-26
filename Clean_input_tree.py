"""
Walks a source directory tree and builds a new mirrored tree made of
symlinks, cleaned according to the rules below. Nothing in the source
tree is touched, moved, or deleted.

A "branch" is a directory holding the sub-directory the ROOT file to keep
lives in (given by --keep-dir), and every sub-directory given by
--require-dir. For example, with '--keep-dir BANFF_PRE --require-dir BANFF_POST':

    .../some/branch/
        BANFF_PRE/
            something.root
            something.log
        BANFF_POST/
            something.root
            something.log
        branch.log
        branch.root

After cleaning, the corresponding location in the destination tree will
contain only:

    .../some/branch/
        something.root   (symlink)

Usage:
    python Clean_input_tree.py /path/to/source /path/to/destination --keep-dir 2024NoGSF
    python Clean_input_tree.py /path/to/source /path/to/destination --keep-dir BANFF_PRE --require-dir BANFF_POST
    python Clean_input_tree.py /path/to/source /path/to/destination --keep-dir 2024NoGSF --dry-run
"""

import argparse
import os
import sys
from pathlib import Path

# Pattern used to find the root file to keep inside the kept sub-directory.
ROOT_FILE_GLOB = "*.root"

def find_branch_dirs(src_root: Path, keep_dir: str, require_dirs):
    """Yield every directory in src_root that looks like a 'branch':
    i.e. it directly contains the kept sub-directory and every required one.
    """
    needed = [keep_dir] + list(require_dirs)
    for dirpath, dirnames, _filenames in os.walk(src_root):
        if all(name in dirnames for name in needed):
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


def process_branch(branch_src: Path, src_root: Path, dst_root: Path, keep_dir: str, dry_run: bool):
    rel = branch_src.relative_to(src_root)
    branch_dst = dst_root / rel

    keep_src = branch_src / keep_dir
    root_files = sorted(keep_src.glob(ROOT_FILE_GLOB))

    if not root_files:
        print(f"  WARNING: no root file found in {keep_src}, skipping this branch")
        return
    if len(root_files) > 1:
        print(f"  WARNING: multiple root files found in {keep_src}, keeping all of them")

    for root_file in root_files:
        # Use resolve() so the symlink target is an absolute, real path,
        # robust to the destination tree being moved elsewhere later.
        target = root_file.resolve()
        link_path = branch_dst / root_file.name

        print(f"  keep: {root_file}")
        make_symlink(link_path, target, dry_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="Path to the existing directory tree")
    parser.add_argument("destination", type=Path, help="Path to the new (cleaned) tree to create")
    parser.add_argument(
        "--keep-dir",
        required=True,
        help="Name of the sub-directory of a branch holding the ROOT file to keep "
             "(ex: '2024NoGSF' for the GENIE tree, 'BANFF_PRE' for the NEUT one)",
    )
    parser.add_argument(
        "--require-dir",
        action="append",
        default=[],
        help="Name of a sub-directory a branch must also hold to be cleaned (ex: 'BANFF_POST'). "
             "Can be given several times. The ROOT files it holds are not kept.",
    )
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
    for branch_src in find_branch_dirs(src_root, args.keep_dir, args.require_dir):
        print(f"Branch: {branch_src}")
        process_branch(branch_src, src_root, dst_root, args.keep_dir, args.dry_run)
        branch_count += 1

    print(f"\nDone. Processed {branch_count} branch director{'y' if branch_count == 1 else 'ies'}.")
    if args.dry_run:
        print("(dry run -- nothing was actually created)")


if __name__ == "__main__":
    main()
