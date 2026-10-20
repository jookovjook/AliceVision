"""Face landmarks ON the reconstructed head: render the device-free head (step 7, vertex colours) from the front at a
few yaw/pitch offsets, run MediaPipe FaceMesh on each render, lift every landmark to 3-D with the render's depth
buffer (exact surface point in the device frame), and take the median over renders (spread = consistency).

    python face_landmarks_render.py HEAD_CLOSED.ply OUT_PREFIX

Device frame: origin = ear-axis midpoint, x right (wearer's), y forward, z up, metres in the mesh, mm in the output.
Needs mediapipe 0.10.21 (legacy solutions API) and pyrender with EGL.
"""
import json
import os
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import cv2
import numpy as np
import pyrender
import trimesh

LM = {"between_eyes_168": 168, "nose_bridge_6": 6, "glabella_9": 9, "forehead_10": 10, "nose_tip_1": 1,
      "subnasale_2": 2, "chin_152": 152, "eye_R_outer_33": 33, "eye_R_inner_133": 133, "eye_L_inner_362": 362,
      "eye_L_outer_263": 263, "mouth_R_61": 61, "mouth_L_291": 291, "face_R_234": 234, "face_L_454": 454}
W, H, FOV = 900, 1100, 30.0


def look_at(eye, target, up):
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r); u = np.cross(r, f)
    T = np.eye(4); T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = r, u, -f, eye
    return T


def main():
    import mediapipe as mp
    mesh_path, out = sys.argv[1], sys.argv[2]
    m = trimesh.load(mesh_path, process=False)
    scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[1.0, 1.0, 1.0])
    scene.add(pyrender.Mesh.from_trimesh(m, smooth=False))
    cam = pyrender.PerspectiveCamera(yfov=np.radians(FOV), aspectRatio=W / H)
    cn = scene.add(cam)
    r = pyrender.OffscreenRenderer(W, H)
    fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.3)
    target = np.array([0.0, 0.06, -0.02])                    # mid-face in the device frame
    fy = (H / 2) / np.tan(np.radians(FOV) / 2)
    pts = {k: [] for k in LM}; used = 0
    for yaw in (-20, -10, 0, 10, 20):
        for pitch in (-25, -15, -5, 5):                       # the face looks down in the device frame
            a, b = np.radians(yaw), np.radians(pitch)
            dirv = np.array([np.sin(a) * np.cos(b), np.cos(a) * np.cos(b), np.sin(b)])
            T = look_at(target + 0.45 * dirv, target, np.array([0, 0, 1.0]))
            scene.set_pose(cn, T)
            col, dep = r.render(scene)
            res = fm.process(np.ascontiguousarray(col[..., :3]))
            if not res.multi_face_landmarks:
                continue
            used += 1
            lm = res.multi_face_landmarks[0].landmark
            if used == 1 or (yaw == 0 and pitch == -15):
                vis = col.copy()
                for k, i in LM.items():
                    cv2.circle(vis, (int(lm[i].x * W), int(lm[i].y * H)), 5, (255, 0, 0), -1)
                cv2.imwrite(out + f"_render_y{yaw}_p{pitch}.png", vis[..., ::-1])
            for k, i in LM.items():
                u, v = lm[i].x * W, lm[i].y * H
                ui, vi = int(round(u)), int(round(v))
                if not (0 <= ui < W and 0 <= vi < H) or dep[vi, ui] <= 0:
                    continue
                z = dep[vi, ui]                                # distance along the camera -z axis
                pc = np.array([(u - W / 2) / fy * z, -(v - H / 2) / fy * z, -z])
                pts[k].append((T[:3, :3] @ pc + T[:3, 3]) * 1000)
    resj = {}
    for k, P in pts.items():
        if P:
            P = np.array(P); med = np.median(P, 0)
            resj[k] = {"mm": med.round(2).tolist(), "renders": len(P), "spread_mm": float(np.median(np.linalg.norm(P - med, axis=1)))}
    json.dump({"renders_with_face": used, "landmarks": resj}, open(out + ".json", "w"), indent=1)
    print(f"face found in {used}/20 renders")
    for k, v in resj.items():
        print(f"{k:18s} renders {v['renders']:2d} spread {v['spread_mm']:4.1f} mm  X mm {v['mm']}")


if __name__ == "__main__":
    main()
