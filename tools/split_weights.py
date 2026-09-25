#!/usr/bin/env python3
"""Split weights.u8 (from convert_model.py) into model/w-0.wasm, w-1.wasm, w-2.wasm (< 15 MB each)
and copy bias.f32 / silueta.json into model/. Run from the repo root after convert_model.py."""
import shutil, sys
from pathlib import Path

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
out = Path("model"); out.mkdir(exist_ok=True)
data = (src / "weights.u8").read_bytes()
CHUNK = 14_800_000
for i in range(0, len(data), CHUNK):
    (out / f"w-{i // CHUNK}.wasm").write_bytes(data[i:i + CHUNK])
shutil.copy(src / "bias.f32", out / "bias.wasm")
shutil.copy(src / "silueta.json", out / "silueta.json")
print(f"wrote {-(-len(data) // CHUNK)} weight chunks to model/")
