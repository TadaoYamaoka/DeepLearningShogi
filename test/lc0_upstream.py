"""Read the pinned upstream revision, independent of the sibling checkout HEAD."""
from pathlib import Path
import subprocess


def upstream_source(root: Path, relative: str | Path) -> bytes:
    checkout = (root.parent / "lc0").resolve()
    revision = (root / "lc0/UPSTREAM_REVISION").read_text().strip()
    return subprocess.run(
        ["git", "-c", f"safe.directory={checkout.as_posix()}", "-C", str(checkout),
         "show", f"{revision}:{Path(relative).as_posix()}"],
        check=True, capture_output=True,
    ).stdout


def modified_sources(root: Path) -> list[str]:
    vendor = root / "lc0"
    return [p.relative_to(vendor).as_posix()
            for p in sorted((vendor / "src").rglob("*"))
            if p.is_file() and p.read_bytes().replace(b"\r\n", b"\n") !=
            upstream_source(root, p.relative_to(vendor))]
