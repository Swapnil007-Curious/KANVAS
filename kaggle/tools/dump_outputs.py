"""Print the text outputs of an executed notebook and save its images, for review."""
import base64
import sys
from pathlib import Path

import nbformat

nb_path, img_dir = Path(sys.argv[1]), Path(sys.argv[2])
img_dir.mkdir(parents=True, exist_ok=True)
nb = nbformat.read(nb_path, as_version=4)
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    for j, out in enumerate(cell.get("outputs", [])):
        if "text" in out:
            print(f"--- cell {i} [{out.get('name')}]\n{out['text'][:3000]}")
        data = out.get("data", {})
        if "image/png" in data:
            p = img_dir / f"cell{i:02d}_{j}.png"
            p.write_bytes(base64.b64decode(data["image/png"]))
            print(f"--- cell {i} image -> {p.name}")
        elif "text/markdown" in data:
            print(f"--- cell {i} markdown\n{data['text/markdown']}")
        elif "text/plain" in data:
            print(f"--- cell {i} value\n{data['text/plain'][:3000]}")
