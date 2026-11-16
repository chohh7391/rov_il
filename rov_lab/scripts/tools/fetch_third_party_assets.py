"""Download the third-party scene/object assets listed in ``rov_lab/assets_src/third_party/manifest.json``.

Files go to ``rov_lab/assets/third_party/<source>/<id>/`` (not tracked in Git; converted USDs are published with the
rest of ``rov_lab/assets`` through the Hugging Face dataset). Every downloaded file is hashed into
``rov_lab/assets/third_party/download_log.json`` so a later fetch can be compared. Already-present files are skipped.

    python rov_lab/scripts/tools/fetch_third_party_assets.py            # everything
    python rov_lab/scripts/tools/fetch_third_party_assets.py --only github polyhaven

Only the standard library is used, so any Python >= 3.10 works.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO: Path = Path(__file__).resolve().parents[3]
MANIFEST: Path = REPO / "rov_lab" / "assets_src" / "third_party" / "manifest.json"
DEFAULT_OUT: Path = REPO / "rov_lab" / "assets" / "third_party"
FUEL_API: str = "https://fuel.gazebosim.org/1.0"
POLYHAVEN_API: str = "https://api.polyhaven.com"
USER_AGENT: str = "rov_lab-asset-fetch/1.0"


def _get(url: str, timeout: float = 120.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https hosts from the manifest)
        return resp.read()


def _download(url: str, dest: Path, timeout: float = 600.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as f:  # noqa: S310
        shutil.copyfileobj(resp, f, length=1 << 20)
    tmp.replace(dest)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_extract(zip_path: Path, out_dir: Path) -> list[str]:
    """Extract a zip, refusing absolute paths and ``..`` components (zip-slip)."""
    out_dir = out_dir.resolve()
    names: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            target = (out_dir / info.filename).resolve()
            if not target.is_relative_to(out_dir):
                raise RuntimeError(f"unsafe path in {zip_path.name}: {info.filename}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
            names.append(info.filename)
    return names


def fetch_fuel(entries: list[dict[str, Any]], out: Path, log: dict[str, Any]) -> None:
    for e in entries:
        dest_dir = out / "fuel" / e["id"]
        zip_path = dest_dir / f"{e['id']}.zip"
        if not zip_path.exists():
            name = urllib.parse.quote(e["name"])
            url = f"{FUEL_API}/{e['owner']}/models/{name}/{e['version']}/{name}.zip"
            print(f"[fuel] {e['owner']}/{e['name']} v{e['version']} -> {zip_path}", flush=True)
            _download(url, zip_path)
        files = _safe_extract(zip_path, dest_dir / "model")
        log[f"fuel/{e['id']}"] = {"zip_sha256": _sha256(zip_path), "zip_bytes": zip_path.stat().st_size,
                                  "files": len(files), "license": e["license"]}


def fetch_github(entries: list[dict[str, Any]], out: Path, log: dict[str, Any]) -> None:
    for e in entries:
        dest_dir = out / "github" / e["id"]
        for rel in e["files"]:
            dest = dest_dir / rel
            if not dest.exists():
                url = f"https://raw.githubusercontent.com/{e['repo']}/{e['commit']}/{urllib.parse.quote(rel)}"
                print(f"[github] {e['repo']}@{e['commit'][:8]} {rel}", flush=True)
                _download(url, dest)
            log[f"github/{e['id']}/{rel}"] = {"sha256": _sha256(dest), "bytes": dest.stat().st_size,
                                              "license": e["license"]}


def fetch_polyhaven(entries: list[dict[str, Any]], out: Path, log: dict[str, Any]) -> None:
    for e in entries:
        dest_dir = out / "polyhaven" / e["id"]
        files: dict[str, Any] = json.loads(_get(f"{POLYHAVEN_API}/files/{e['id']}"))
        for m in e["maps"]:
            info = files[m][e["resolution"]]["jpg"]
            dest = dest_dir / Path(urllib.parse.urlparse(info["url"]).path).name
            if not dest.exists():
                print(f"[polyhaven] {e['id']} {m} -> {dest.name}", flush=True)
                _download(info["url"], dest)
            digest = hashlib.md5(dest.read_bytes()).hexdigest()  # noqa: S324 (Poly Haven publishes md5)
            if info.get("md5") and digest != info["md5"]:
                raise RuntimeError(f"md5 mismatch for {dest}")
            log[f"polyhaven/{e['id']}/{m}"] = {"file": dest.name, "md5": digest, "bytes": dest.stat().st_size,
                                               "license": e["license"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--only", nargs="*", choices=["fuel", "github", "polyhaven"], default=None)
    args = ap.parse_args()
    manifest: dict[str, Any] = json.loads(MANIFEST.read_text())
    groups = args.only or ["github", "polyhaven", "fuel"]
    log_path: Path = args.out / "download_log.json"
    log: dict[str, Any] = json.loads(log_path.read_text()) if log_path.exists() else {}
    fetchers = {"fuel": fetch_fuel, "github": fetch_github, "polyhaven": fetch_polyhaven}
    for g in groups:
        fetchers[g](manifest[g], args.out, log)
        log["_updated"] = datetime.now(timezone.utc).isoformat()
        args.out.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(log, indent=1, sort_keys=True))
    print(f"done -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
