"""Fetch pinned official portable engines, verify bytes; no installation."""

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    for engine in json.loads(
        (ROOT / "experiments/configs/paper_tools.json").read_text()
    )["engines"]:
        path = ROOT / engine["path"]
        if not path.exists():
            data = urllib.request.urlopen(engine["url"], timeout=120).read()
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                names = [name for name in z.namelist() if Path(name).name == path.name]
                assert len(names) == 1
                binary = z.read(names[0])
                assert hashlib.sha256(binary).hexdigest() == engine["sha256"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(binary)
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest() == engine["sha256"]
        ), f"Engine checksum mismatch: {path}"
        print(engine["version"], "verified")


if __name__ == "__main__":
    main()
