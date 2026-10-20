"""frame_tags_layout.json -> synthetic reference sfmData for aliceVision_sfmAlignment --method from_markers.

    python layout_to_sfm.py frame_tags_layout.json OUT.sfm [--tags 0-13] [--unit 0.001]

One landmark per tag corner of the RIGID frame tags (default ids 0-13; screw/knob tags move and are excluded),
descType tag36h11, color.r = marker index = tagId*4 + k (k = layout corner TL, TR, BR, BL), X = layout corner
in the device frame (ear-axis midpoint origin, x right, y forward, z up) times --unit (mm -> m).
from_markers matches landmarks by (descType, color.r) and drops ids that occur twice in either file.
"""
import argparse
import json
from pathlib import Path


def parse_ids(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layout"); ap.add_argument("out")
    ap.add_argument("--tags", default="0-13"); ap.add_argument("--unit", type=float, default=0.001)
    a = ap.parse_args()
    L = json.load(open(a.layout))
    assert L["family"].upper().endswith("36H11")
    s = []
    for t in parse_ids(a.tags):
        for k, c in enumerate(L["frame"][str(t)]):
            idx = t * 4 + k
            s.append({"landmarkId": str(idx), "descType": "tag36h11", "color": [str(idx), "0", "0"],
                      "X": [repr(float(v) * a.unit) for v in c], "observations": []})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"version": ["1", "2", "14"], "views": [], "intrinsics": [], "poses": [], "structure": s}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(s)} corner landmarks of tags {a.tags} (unit {a.unit} m per layout unit)")


if __name__ == "__main__":
    main()
