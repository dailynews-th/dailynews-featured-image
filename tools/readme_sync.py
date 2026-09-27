#!/usr/bin/env python3
"""Sync README.md with the app: current version + the full changelog (from VERSION / CHANGELOG in src/index.src.html).

Runs automatically from build.py. The changelog lives between the
<!-- CHANGELOG:START --> / <!-- CHANGELOG:END --> markers in README.md.
"""
import re
from pathlib import Path

root = Path(__file__).resolve().parent.parent
src = (root / "src" / "index.src.html").read_text(encoding="utf-8")
readme_p = root / "README.md"
readme = readme_p.read_text(encoding="utf-8")

version = re.search(r'const VERSION = "([^"]+)"', src).group(1)
block = src[src.index("const CHANGELOG = ["):]
block = block[: block.index("\n  ];")]
entries = re.findall(r'\{ v: "([^"]+)", date: "([^"]+)", items: \[(.*?)\]\}', block, re.S)

lines = []
for v, date, items in entries:
    lines.append(f"### v{v} — {date}" + (" (ล่าสุด)" if v == version else ""))
    for it in re.findall(r'"((?:[^"\\]|\\.)*)"', items):
        lines.append(f"- {it}")          # **name** stays bold in Markdown too
    lines.append("")
log = "\n".join(lines).rstrip() + "\n"

START, END = "<!-- CHANGELOG:START -->", "<!-- CHANGELOG:END -->"
section = f"{START}\n{log}{END}"
if START in readme:
    readme = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda m: section, readme, flags=re.S)
else:  # first run: add the section just before the credits
    anchor = "## เครดิตและลิขสิทธิ์"
    add = f"## ประวัติการอัปเดต (Changelog)\n\nสร้างอัตโนมัติจาก `CHANGELOG` ใน `src/index.src.html` ทุกครั้งที่รัน `python3 build.py`\n\n{section}\n\n---\n\n"
    readme = readme.replace(anchor, add + anchor, 1) if anchor in readme else readme.rstrip() + "\n\n---\n\n" + add
readme = re.sub(r"เวอร์ชันปัจจุบัน: \*\*v[^*]+\*\*", f"เวอร์ชันปัจจุบัน: **v{version}**", readme)
readme_p.write_text(readme, encoding="utf-8")
print(f"README.md synced (v{version}, {len(entries)} versions)")
