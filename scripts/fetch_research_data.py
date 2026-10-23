#!/usr/bin/env python3
"""Download the Hurricane Season research pack and extract it.

The case packet links this Google Drive file (hurricane-season-research.zip,
about 34MB compressed):

https://drive.google.com/file/d/1hWJ0S2K2FbG0xxLv8OEPL9zPjAg-CKuB/view

The zip extracts to hurricane-season-research/. history/ (600 seasons) is
gitignored. case.json and starter_kit/ are already in the repo; extracting
again overwrites them with the same bytes.

Uses only the Python standard library. Re-run after a fresh clone:

    python3 scripts/fetch_research_data.py
"""

from __future__ import annotations

import argparse
import http.cookiejar
import os
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from html.parser import HTMLParser
from pathlib import Path

FILE_ID = "1hWJ0S2K2FbG0xxLv8OEPL9zPjAg-CKuB"
USER_AGENT = "ctc-research-fetch/1.0"
MANIFEST = "hurricane-season-research/history/manifest.json"
CHUNK = 1 << 20


class _DownloadForm(HTMLParser):
    """The virus-scan warning page: a GET form with id, confirm, and uuid."""

    def __init__(self) -> None:
        super().__init__()
        self.action: str | None = None
        self.inputs: dict[str, str] = {}
        self._in_form = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        fields = dict(attrs)
        if tag == "form" and fields.get("id") == "download-form":
            self._in_form = True
            self.action = fields.get("action")
        elif self._in_form and tag == "input" and fields.get("name"):
            self.inputs[fields["name"]] = fields.get("value") or ""

    def handle_endtag(self, tag: str) -> None:
        if tag == "form":
            self._in_form = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _opener() -> urllib.request.OpenerDirector:
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.addheaders = [("User-Agent", USER_AGENT)]
    return opener


def _stream_to_temp(opener: urllib.request.OpenerDirector, url: str, directory: Path) -> tuple[Path, bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    fd, name = tempfile.mkstemp(dir=directory, suffix=".partial")
    os.close(fd)
    path = Path(name)
    try:
        with opener.open(request, timeout=180) as response, path.open("wb") as out:
            head = response.read(512)
            out.write(head)
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                out.write(block)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path, head


def _is_zip(head: bytes) -> bool:
    return head.startswith(b"PK")


def _confirm_url(html: str) -> str:
    form = _DownloadForm()
    form.feed(html)
    if not form.action or "id" not in form.inputs:
        raise SystemExit("Google Drive did not return the research zip or a download form.")
    query = urllib.parse.urlencode(form.inputs)
    joiner = "&" if "?" in form.action else "?"
    return form.action + joiner + query


def download_zip(dest: Path) -> None:
    """Save hurricane-season-research.zip to dest. Follow the Drive confirm form."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    opener = _opener()
    first = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    partial, head = _stream_to_temp(opener, first, dest.parent)
    try:
        if not _is_zip(head):
            html = partial.read_bytes().decode("utf-8", errors="replace")
            partial.unlink()
            partial, head = _stream_to_temp(opener, _confirm_url(html), dest.parent)
            if not _is_zip(head):
                raise SystemExit("Drive confirm step did not return a zip.")
        partial.replace(dest)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    print(f"downloaded {dest} ({dest.stat().st_size} bytes)")


def _zip_has_manifest(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return MANIFEST in archive.namelist()
    except zipfile.BadZipFile:
        return False


def extract_zip(archive_path: Path, dest: Path) -> None:
    """Extract under dest. Refuse members that would write outside dest."""
    dest = dest.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            target = (dest / info.filename).resolve()
            if target != dest and dest not in target.parents:
                raise SystemExit(f"refusing to extract {info.filename}")
        archive.extractall(dest)
    manifest = dest / MANIFEST
    if not manifest.is_file():
        raise SystemExit(f"extract finished without {manifest}")
    print(f"extracted {manifest}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=_repo_root(), help="directory that will contain hurricane-season-research/")
    parser.add_argument("--force", action="store_true", help="download and extract again even if history/manifest.json is already present")
    args = parser.parse_args()
    dest = args.dest.resolve()
    archive = dest / "hurricane-season-research.zip"
    manifest = dest / MANIFEST
    if manifest.is_file() and _zip_has_manifest(archive) and not args.force:
        print(f"already present: {manifest}")
        return 0
    if not _zip_has_manifest(archive) or args.force:
        download_zip(archive)
    if not _zip_has_manifest(archive):
        print("downloaded file is not the research zip", file=sys.stderr)
        return 1
    extract_zip(archive, dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
