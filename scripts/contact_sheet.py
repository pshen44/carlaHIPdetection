#!/usr/bin/env python3
"""Contact sheet of HIP crops across each burst, to eyeball flashing lights.

    python scripts/contact_sheet.py --data data/benchmark --out sheet.png --ids s00003 s00010
    python scripts/contact_sheet.py --data data/benchmark --out sheet.png --n 8   # first 8 visible HIPs
"""

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hipdet import dataset  # noqa: E402

CELL = 150
LABEL_W = 230


def row_for(sample, full_frame=False):
    frames = [Image.open(p).convert("RGB") for p in sample.frame_paths()]
    bb = sample.gt["target_bbox"]
    if full_frame or not bb:
        crops = [f.resize((CELL * 4 // 3, CELL)) for f in frames[:1]]
    else:
        x0, y0, x1, y1 = bb
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        h = max(x1 - x0, y1 - y0) // 2 + 10
        crops = [f.crop((cx - h, cy - h, cx + h, cy + h)).resize((CELL, CELL), Image.NEAREST) for f in frames]
    row = Image.new("RGB", (LABEL_W + sum(c.width for c in crops) + 4 * len(crops), CELL), "white")
    x = LABEL_W
    for c in crops:
        row.paste(c, (x, 0))
        x += c.width + 4
    s, g = sample.scenario, sample.gt
    text = (f"{sample.sample_id}  {s['weather']}\n{s['hip_type']}  lights={'on' if s['lights_on'] else 'off'}\n"
            f"{s['placement']}  {g['target_distance_m']} m\nHIP={g['hip_present']}  respond={g['response_required']}")
    ImageDraw.Draw(row).multiline_text((6, 30), text, fill=(20, 20, 20), spacing=6)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    samples = dataset.load(args.data)
    if args.ids:
        by_id = {s.sample_id: s for s in samples}
        chosen = [by_id[i] for i in args.ids]
    else:
        chosen = [s for s in samples if s.gt["target_visible"]][: args.n]
    rows = [row_for(s) for s in chosen]
    width = max(r.width for r in rows)
    sheet = Image.new("RGB", (width, sum(r.height + 4 for r in rows)), "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 4
    sheet.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
