"""Create a labeled contact sheet from an existing reference run; never modify raw data."""

import argparse
from pathlib import Path
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
sheet = Image.new("RGB", (960, 624), (24, 24, 28))
draw = ImageDraw.Draw(sheet)
columns = [
    ("scene_t0", "Scene at t=0"),
    ("scene_t1", "Scene at t=1 s"),
    ("oracle_recursive_t1", "Offline recursive reference at t=1 s"),
]
for row, scene in enumerate(["A", "B", "C"]):
    for column, (filename, label) in enumerate(columns):
        image = Image.open(args.run / scene / (filename + ".png"))
        sheet.paste(image, (column * 320, row * 208 + 28))
        draw.text((column * 320 + 8, row * 208 + 7), f"{scene}: {label}", fill="white")
args.output.parent.mkdir(parents=True, exist_ok=True)
sheet.save(args.output)
