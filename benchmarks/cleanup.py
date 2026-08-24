#!/usr/bin/env python
"""Clean up downloaded and generated benchmark files.

This script removes all downloaded datasets and generated output files
to keep the repository clean.

Usage:
    python cleanup.py           # Clean all generated files
    python cleanup.py --dry-run # Show what would be deleted
"""

import argparse
from pathlib import Path

# Script directory
SCRIPT_DIR = Path(__file__).parent.resolve()
TESTSETS_DIR = SCRIPT_DIR / "testsets"
OUTFILES_DIR = SCRIPT_DIR / "outfiles"


def cleanup(dry_run: bool = False) -> None:
    """Remove all generated benchmark files.

    Args:
        dry_run: If True, only print what would be deleted
    """
    files_to_delete: list[Path] = []
    dirs_to_delete: list[Path] = []

    # Collect testset files (.gold, .all, .none, .mixed)
    if TESTSETS_DIR.exists():
        for pattern in ["*.gold", "*.all", "*.none", "*.mixed"]:
            files_to_delete.extend(TESTSETS_DIR.glob(pattern))

        # Check if directory will be empty after deletion
        remaining = set(TESTSETS_DIR.iterdir()) - set(files_to_delete)
        if not remaining:
            dirs_to_delete.append(TESTSETS_DIR)

    # Collect output files
    if OUTFILES_DIR.exists():
        files_to_delete.extend(OUTFILES_DIR.glob("*.out"))
        files_to_delete.extend(OUTFILES_DIR.glob("*.txt"))  # Error files

        # Check if directory will be empty after deletion
        remaining = set(OUTFILES_DIR.iterdir()) - set(files_to_delete)
        if not remaining:
            dirs_to_delete.append(OUTFILES_DIR)

    # Collect __pycache__ directories
    pycache_dir = SCRIPT_DIR / "__pycache__"
    if pycache_dir.exists():
        files_to_delete.extend(pycache_dir.glob("*"))
        dirs_to_delete.append(pycache_dir)

    if not files_to_delete and not dirs_to_delete:
        print("Nothing to clean up.")
        return

    # Print summary
    action = "Would delete" if dry_run else "Deleting"

    if files_to_delete:
        print(f"{action} {len(files_to_delete)} files:")
        for f in sorted(files_to_delete):
            print(f"  {f.relative_to(SCRIPT_DIR)}")

            if not dry_run:
                f.unlink()

    if dirs_to_delete:
        print(f"\n{action} {len(dirs_to_delete)} empty directories:")
        for d in dirs_to_delete:
            print(f"  {d.relative_to(SCRIPT_DIR)}/")

            if not dry_run:
                d.rmdir()

    if dry_run:
        print("\nRun without --dry-run to actually delete files.")
    else:
        print("\nCleanup complete.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )

    args = parser.parse_args()
    cleanup(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
