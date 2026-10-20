"""Face landmarks in the device frame: MediaPipe FaceMesh per keyframe + robust multi-view triangulation.

    python face_landmarks.py GS_DATASET OUT_PREFIX

GS_DATASET = gs_export.py output (undistorted centred pinhole images + COLMAP model with poses in the device frame,
metres). Frames are rotated 90 deg clockwise for the detector (person upright) and the landmarks rotated back.
Per landmark: RANSAC over views (pairs, 6 px), then least-squares DLT on the inliers; reports the number of views,
the median reprojection residual and the 3-D position in mm. Landmarks occluded by the arc (e.g. 168 between the
eyes, under the sagittal arc) are still predicted by FaceMesh from context: trust them only if the residual is low.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

LM = {"between_eyes_168": 168, "nose_bridge_6": 6, "nose_tip_1": 1, "chin_152": 152, "forehead_10": 10,
      "eye_R_outer_33": 33, "eye_R_inner_133": 133, "eye_L_inner_362": 362, "eye_L_outer_263": 263,
      "mouth_R_61": 61, "mouth_L_291": 291, "face_R_tragus_234": 234, "face_L_tragus_454": 454,
      "subnasale_2": 2, "glabella_9": 9}


def read_model(ds):
    cam = open(ds / "sparse/0/cameras.txt").read().split("\n")[1].split()
    W, H, f, _, cx, cy = int(cam[2]), int(cam[3]), *map(float, cam[4:8])
    imgs = {}
    for line in open(ds / "sparse/0/images.txt"):
        p = line.split()
        if len(p) == 10 and not line.startswith("#"):
            q = [float(x) for x in p[1:5]]; t = np.array([float(x) for x in p[5:8]])
            R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            imgs[p[9]] = np.c_[R, t]
    K = np.array([[f, 0, cx - 0.5], [0, f, cy - 0.5], [0, 0, 1]])  # COLMAP centre -> OpenCV index coords
    return K, imgs, W, H


def tri(Ps, xs):
    A = []
    for P, (u, v) in zip(Ps, xs):
        A += [u * P[2] - P[0], v * P[2] - P[1]]
    X = np.linalg.svd(np.array(A))[2][-1]
    return X[:3] / X[3]


def reproj(P, X):
    x = P @ np.r_[X, 1]
    return x[:2] / x[2]


def main():
    import mediapipe as mp
    ds, out = Path(sys.argv[1]), sys.argv[2]
    K, imgs, W, H = read_model(ds)
    fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    obs = {k: [] for k in LM}
    nface = 0
    for name, Rt in sorted(imgs.items()):
        img = cv2.imread(str(ds / "images" / name))
        rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)                 # person upright
        res = fm.process(cv2.cvtColor(rot, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            continue
        nface += 1
        lm = res.multi_face_landmarks[0].landmark
        hr, wr = rot.shape[:2]
        for k, i in LM.items():
            xr, yr = lm[i].x * wr, lm[i].y * hr                       # rotated image coords
            u, v = yr, (H - 1) - xr                                    # undo 90 deg CW rotation: (x,y) -> (H-1-y, x)
            obs[k].append((K @ Rt, (u, v), name))
    res = {}
    rng = np.random.default_rng(0)
    for k, O in obs.items():
        if len(O) < 3:
            res[k] = {"views": len(O)}; continue
        best = None
        for _ in range(300):
            i, j = rng.choice(len(O), 2, replace=False)
            X = tri([O[i][0], O[j][0]], [O[i][1], O[j][1]])
            e = np.array([np.linalg.norm(reproj(P, X) - x) for P, x, _ in O])
            inl = e < 6.0
            if best is None or inl.sum() > best.sum():
                best = inl
        X = tri([O[i][0] for i in np.flatnonzero(best)], [O[i][1] for i in np.flatnonzero(best)])
        e = np.array([np.linalg.norm(reproj(P, X) - x) for P, x, _ in O])
        res[k] = {"mm": (X * 1000).round(2).tolist(), "views": len(O), "inliers": int((e < 6).sum()),
                  "median_reproj_px": float(np.median(e[e < 6])) if (e < 6).any() else None}
    json.dump({"frames_with_face": nface, "landmarks": res}, open(out + ".json", "w"), indent=1)
    print(f"face found in {nface}/{len(imgs)} frames")
    for k, r in res.items():
        print(f"{k:22s} views {r['views']:3d} inliers {r.get('inliers', 0):3d} reproj {r.get('median_reproj_px') or float('nan'):5.2f} px  X mm {r.get('mm')}")


if __name__ == "__main__":
    main()
