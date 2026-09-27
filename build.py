#!/usr/bin/env python3
"""Build index.html by inlining src/glnet.js into src/index.src.html."""
from pathlib import Path

root = Path(__file__).parent
page = (root / "src" / "index.src.html").read_text(encoding="utf-8")
engine = (root / "src" / "glnet.js").read_text(encoding="utf-8")
assert "/*GLNET*/" in page, "placeholder /*GLNET*/ not found in src/index.src.html"
assert "</script" not in engine, "glnet.js must not contain a closing script tag"
(root / "index.html").write_text(page.replace("/*GLNET*/", engine), encoding="utf-8")
print("built index.html")

# keep README.md in step with the app (version + changelog)
import runpy
runpy.run_path(str(root / "tools" / "readme_sync.py"))
