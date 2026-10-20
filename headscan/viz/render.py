"""Pictures of intermediate results (headless, pyrender + EGL on the GPU box).

    python render.py sfm   SFM.json OUT.png [--title T]
    python render.py mesh  SFM.json MESH.obj OUT.png [--crop R] [--title T] [--untextured]

SFM.json is an AliceVision sfmData JSON (aliceVision_convertSfMFormat --output x.json). The head centre is the
point closest to all camera viewing rays (the cameras orbit the head), "up" is the mean camera -x axis (the
phone was held in portrait, the video is stored landscape). AliceVision OBJ meshes are stored as (x,-y,-z)
of the SfM world; we undo that so mesh and cameras share one frame.
"""
import argparse
import json
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_sfm(path):
    d = json.load(open(path))
    views = {v["viewId"]: v for v in d.get("views", [])}
    poses = {}
    for p in d.get("poses", []):
        t = p["pose"]["transform"]
        R = np.array([float(x) for x in t["rotation"]]).reshape(3, 3, order="F")  # c_R_w, Eigen column-major
        poses[p["poseId"]] = (R, np.array([float(x) for x in t["center"]]))
    cams = []
    for vid, v in views.items():
        if v.get("poseId") in poses:
            R, C = poses[v["poseId"]]
            cams.append((int(vid), R, C))
    cams.sort()
    X = np.array([[float(c) for c in l["X"]] for l in d.get("structure", [])]).reshape(-1, 3)
    col = np.array([[float(c) for c in l.get("color", [128, 128, 128])] for l in d.get("structure", [])]).reshape(-1, 3)
    return d, cams, X, col


def head_frame(cams):
    """Least-squares intersection of the camera optical axes, up vector, median camera distance."""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for _, R, C in cams:
        z = R[2]  # optical axis in world (third row of c_R_w)
        P = np.eye(3) - np.outer(z, z)
        A += P; b += P @ C
    ctr = np.linalg.solve(A, b)
    up = -np.mean([R[0] for _, R, _ in cams], axis=0)
    up /= np.linalg.norm(up)
    dist = float(np.median([np.linalg.norm(C - ctr) for _, _, C in cams]))
    return ctr, up, dist


def look_at(eye, target, up):
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = r, u, -f, eye  # OpenGL camera: looks down -z
    return T


def basis(up):
    a = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(up, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    return e1, e2


def plot_sfm(args):
    d, cams, X, col = load_sfm(args.sfm)
    ctr, up, dist = head_frame(cams)
    e1, e2 = basis(up)
    nviews = len(d.get("views", []))
    keep = np.linalg.norm(X - ctr, axis=1) < 3 * dist
    Xl = np.stack([(X - ctr) @ e1, (X - ctr) @ e2, (X - ctr) @ up], 1) / dist
    Cl = np.array([[(C - ctr) @ e1, (C - ctr) @ e2, (C - ctr) @ up] for _, _, C in cams]) / dist
    fig = plt.figure(figsize=(16, 5.6))
    for k, (el, az, name) in enumerate([(25, -60, "oblique"), (89, -90, "top (along up)"), (0, -90, "side")]):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.scatter(*Xl[keep].T, c=col[keep] / 255.0, s=0.3, depthshade=False)
        ax.plot(*Cl.T, "-", color="tab:red", lw=0.8)
        ax.scatter(*Cl.T, color="tab:red", s=4)
        ax.scatter([0], [0], [0], color="k", marker="+", s=80)
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_zlim(-1.5, 1.5)
        ax.view_init(el, az); ax.set_title(name); ax.set_axis_off()
    fig.suptitle(f"{args.title}   {len(cams)}/{nviews} cameras posed, {len(X)} points "
                 f"(units: median camera-to-head distance)")
    plt.tight_layout(); plt.savefig(args.out, dpi=110); print("wrote", args.out)


def render_mesh(args):
    import pyrender
    import trimesh

    d, cams, X, col = load_sfm(args.sfm)
    ctr, up, dist = head_frame(cams)
    loaded = trimesh.load(args.mesh, process=False)
    geoms = list(loaded.geometry.values()) if isinstance(loaded, trimesh.Scene) else [loaded]
    flip = np.diag([1.0, -1.0, -1.0])
    nf = 0
    tex_meshes, plain_meshes = [], []
    for g in geoms:
        g = g.copy()
        g.vertices = g.vertices @ flip
        if args.crop:
            keepf = np.linalg.norm(g.vertices[g.faces].mean(1) - ctr, axis=1) < args.crop * dist
            g.update_faces(keepf); g.remove_unreferenced_vertices()
        nf += len(g.faces)
        if len(g.faces):
            tex_meshes.append(g)
            p = trimesh.Trimesh(g.vertices, g.faces, process=False)
            p.visual.vertex_colors = [200, 200, 200, 255]
            plain_meshes.append(p)
    rows = [("textured", tex_meshes, True)] if not args.untextured else []
    rows.append(("geometry", plain_meshes, False))
    e1, e2 = basis(up)
    azs = np.radians(np.arange(0, 360, 360 / args.nviews))
    W, H = 520, 620
    r = pyrender.OffscreenRenderer(W, H)
    fig, axs = plt.subplots(len(rows), len(azs), figsize=(2.6 * len(azs), 3.2 * len(rows)), squeeze=False)
    rd = args.view_dist * dist
    for ri, (name, ms, flat) in enumerate(rows):
        scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[0.35, 0.35, 0.35])
        for m in ms:
            scene.add(pyrender.Mesh.from_trimesh(m, smooth=False))
        cam = pyrender.PerspectiveCamera(yfov=np.radians(args.fov))
        cn = scene.add(cam)
        ln = scene.add(pyrender.DirectionalLight(intensity=3.0))
        for ai, a in enumerate(azs):
            eye = ctr + rd * (np.cos(a) * e1 + np.sin(a) * e2) * np.cos(np.radians(10)) + rd * np.sin(np.radians(10)) * up
            T = look_at(eye, ctr, up)
            scene.set_pose(cn, T); scene.set_pose(ln, T)
            flags = pyrender.RenderFlags.FLAT if flat else pyrender.RenderFlags.NONE
            c, _ = r.render(scene, flags=flags)
            axs[ri, ai].imshow(c); axs[ri, ai].set_axis_off()
            axs[ri, ai].set_title(f"{name} {np.degrees(a):.0f} deg", fontsize=8)
    fig.suptitle(f"{args.title}   {nf} faces" + (f" within {args.crop:g} x camera distance of the head centre" if args.crop else ""))
    plt.tight_layout(); plt.savefig(args.out, dpi=100); print("wrote", args.out)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("sfm"); a.add_argument("sfm"); a.add_argument("out"); a.add_argument("--title", default="")
    b = sp.add_parser("mesh"); b.add_argument("sfm"); b.add_argument("mesh"); b.add_argument("out")
    b.add_argument("--crop", type=float, default=None); b.add_argument("--title", default="")
    b.add_argument("--untextured", action="store_true"); b.add_argument("--nviews", type=int, default=6)
    b.add_argument("--view-dist", type=float, default=0.75); b.add_argument("--fov", type=float, default=45)
    args = ap.parse_args()
    plot_sfm(args) if args.cmd == "sfm" else render_mesh(args)


if __name__ == "__main__":
    main()
