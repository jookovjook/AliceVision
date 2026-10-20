"""AliceVision pipeline driver for the head + arc captures (our build, /opt/av/env.sh).

    python pipeline.py OUT_DIR --capture CAP --frames FRAMES_DIR [--poses none|arkit-locked|arkit-refine]
        [--describers dspsift[,tag36h11]] [--matching frustum|sequential-voctree] [--dense] [--bbox-half 0.2]
        [--stages a,b,...] [--force]

Stages (each writes OUT_DIR/<nn>_<stage>/ and is skipped when its output exists, unless --force):
  init      stray_to_sfm.py (views, ARKit intrinsics + LUT distortion [+ ARKit poses]) -> cameraInit
  features  featureExtraction            matching  imageMatching + featureMatching
  tracks    tracksBuilding               sfm       sfmBootstrapping+sfmExpanding (no poses) | sfmExpanding (poses)
  dense     intrinsicsTransforming, exportImages, depthMapEstimation, depthMapFiltering, meshing (head box),
            meshFiltering, texturing
Every command line and its log go to OUT_DIR/<stage>/cmd.log. Always long flags (several tools share short ones).
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HS = Path(__file__).resolve().parent
AV = os.environ.get("ALICEVISION_ROOT", "/opt/av/install")
BIN = Path(AV) / "bin"
PY = [sys.executable, "-I"]
VOCTREE = os.environ.get("ALICEVISION_VOCTREE", f"{AV}/share/aliceVision/vlfeat_K80L3.SIFT.tree")


def run(cmd, logf, cwd=None):
    cmd = [str(c) for c in cmd]
    t0 = time.time()
    with open(logf, "a") as lf:
        lf.write("\n$ " + " ".join(shlex.quote(c) for c in cmd) + "\n"); lf.flush()
        p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=cwd)
    dt = time.time() - t0
    print(f"  {Path(cmd[0]).name} {'ok' if p.returncode == 0 else 'FAILED'} ({dt:.0f} s)", flush=True)
    if p.returncode:
        print(open(logf).read()[-3000:])
        raise SystemExit(f"command failed: {cmd[0]} (log {logf})")


def av(tool):
    return BIN / f"aliceVision_{tool}"


class P:
    def __init__(self, a):
        self.a, self.out = a, Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.d = {s: self.out / f"{i:02d}_{s}" for i, s in enumerate(["init", "features", "matching", "tracks", "sfm", "dense"])}

    def stage(self, name, done_file):
        d = self.d[name]
        if (d / done_file).exists() and not self.a.force:
            print(f"[{name}] done, skipping"); return None
        d.mkdir(parents=True, exist_ok=True)
        print(f"[{name}]", flush=True)
        return d

    def init(self):
        d = self.stage("init", "cameraInit.sfm")
        if d is None: return
        flags = []
        if self.a.poses != "none":
            flags += ["--poses"] + (["--lock-poses"] if self.a.poses == "arkit-locked" else [])
        if self.a.lock_intrinsics:
            flags += ["--lock-intrinsics"]
        run(PY + [HS / "stray_to_sfm.py", self.a.capture, self.a.frames, d / "viewpoints.sfm", "--intrinsics", self.a.intrinsics,
                  "--distortion", self.a.distortion] + flags, d / "cmd.log")
        run([av("cameraInit"), "--input", d / "viewpoints.sfm", "--output", d / "cameraInit.sfm",
             "--sensorDatabase", f"{AV}/share/aliceVision/cameraSensors.db", "--verboseLevel", "info"], d / "cmd.log")

    def features(self):
        d = self.stage("features", "done")
        if d is None: return
        run([av("featureExtraction"), "--input", self.d["init"] / "cameraInit.sfm", "--output", d,
             "--describerTypes", self.a.describers, "--describerPreset", self.a.preset, "--describerQuality", "high",
             "--forceCpuExtraction", "1", "--maxThreads", "0", "--verboseLevel", "info"], d / "cmd.log")
        (d / "done").touch()

    def matching(self):
        d = self.stage("matching", "done")
        if d is None: return
        sfm = self.d["init"] / "cameraInit.sfm"
        if self.a.matching == "frustum":
            m = ["--method", "FrustumOrVocabularyTree"]
        else:
            m = ["--method", "SequentialAndVocabularyTree", "--nbNeighbors", "10"]
        run([av("imageMatching"), "--input", sfm, "--featuresFolders", self.d["features"], "--output", d / "pairs.txt",
             "--tree", VOCTREE, "--maxDescriptors", "500", "--nbMatches", "40"] + m + ["--verboseLevel", "info"], d / "cmd.log")
        run([av("featureMatching"), "--input", sfm, "--featuresFolders", self.d["features"], "--imagePairsList", d / "pairs.txt",
             "--output", d, "--describerTypes", self.a.describers, "--guidedMatching", "1", "--verboseLevel", "info"], d / "cmd.log")
        (d / "done").touch()

    def tracks(self):
        d = self.stage("tracks", "tracks.json")
        if d is None: return
        run([av("tracksBuilding"), "--input", self.d["init"] / "cameraInit.sfm", "--featuresFolders", self.d["features"],
             "--matchesFolders", self.d["matching"], "--describerTypes", self.a.describers, "--output", d / "tracks.json",
             "--verboseLevel", "info"], d / "cmd.log")

    def sfm(self):
        d = self.stage("sfm", "sfm.json")
        if d is None: return
        sfm_in, tr = self.d["init"] / "cameraInit.sfm", self.d["tracks"] / "tracks.json"
        if self.a.poses == "none":
            run([av("relativePoseEstimating"), "--input", sfm_in, "--tracksFilename", tr, "--output", d / "pairs",
                 "--enforcePureRotation", "0", "--verboseLevel", "info"], d / "cmd.log")
            run([av("sfmBootstrapping"), "--input", sfm_in, "--tracksFilename", tr, "--pairs", d / "pairs",
                 "--output", d / "bootstrap.sfm", "--verboseLevel", "info"], d / "cmd.log")
            sfm_in = d / "bootstrap.sfm"
        lock = "1" if self.a.poses == "arkit-locked" else "0"
        run([av("sfmExpanding"), "--input", sfm_in, "--tracksFilename", tr, "--output", d / "sfm.sfm",
             "--outputViewsAndPoses", d / "cameras.sfm", "--lockScenePreviouslyReconstructed", lock,
             "--minAngleForTriangulation", "2.0", "--verboseLevel", "info"], d / "cmd.log")
        run([av("sfmColorizing"), "--input", d / "sfm.sfm", "--output", d / "sfm_col.sfm", "--verboseLevel", "info"], d / "cmd.log")
        # .sfm is JSON already; sfm.json = same scene (convertSfMFormat with an empty --describerTypes would drop every landmark)
        run([av("convertSfMFormat"), "--input", d / "sfm_col.sfm", "--output", d / "sfm.json", "--describerTypes", self.a.describers,
             "--verboseLevel", "info"], d / "cmd.log")

    def head_box(self):
        """Head centre = least-squares intersection of the camera optical axes; box = cube of half-size bbox_half (m)."""
        sys.path.insert(0, str(HS / "viz"))
        from render import load_sfm, head_frame
        _, cams, _, _ = load_sfm(self.d["sfm"] / "sfm.json")
        ctr, up, dist = head_frame(cams)
        return ctr, up, dist

    def dense(self):
        d = self.stage("dense", "texturing/texturedMesh.obj")
        if d is None: return
        sfm = self.d["sfm"] / "sfm_col.sfm"
        log = d / "cmd.log"
        run([av("intrinsicsTransforming"), "--input", sfm, "--output", d / "sfm_pinhole.sfm", "--type", "pinhole",
             "--inputTracks", "", "--outputTracks", d / "unused_tracks.json", "--verboseLevel", "info"], log)
        run([av("exportImages"), "--input", sfm, "--target", d / "sfm_pinhole.sfm", "--output", d / "images",
             "--outputSfMData", d / "sfm_undist.sfm", "--namingMode", "viewid", "--outputFileType", "exr", "--verboseLevel", "info"], log)
        run([av("depthMapEstimation"), "--input", d / "sfm_pinhole.sfm", "--imagesFolder", d / "images", "--output", d / "depth",
             "--downscale", str(self.a.dm_downscale), "--maxTCams", "10", "--minViewAngle", "2", "--maxViewAngle", "70",
             "--sgmMaxTCamsPerTile", "4", "--sgmUseSfmSeeds", "1", "--sgmDepthListPerTile", "1", "--refineEnabled", "1",
             "--verboseLevel", "info"], log)
        run([av("depthMapFiltering"), "--input", d / "sfm_pinhole.sfm", "--depthMapsFolder", d / "depth",
             "--output", d / "depthFiltered", "--nNearestCams", "10", "--minNumOfConsistentCams", "3",
             "--minNumOfConsistentCamsWithLowSimilarity", "4", "--verboseLevel", "info"], log)
        ctr, up, dist = self.head_box()
        h = self.a.bbox_half * (dist if self.a.poses == "none" else 1.0)  # metres when ARKit poses, else camera units
        bbox = ",".join(f"{v:.6f}" for v in [*ctr, 0, 0, 0, h, h, h])
        json.dump({"centre": ctr.tolist(), "up": up.tolist(), "cam_dist": dist, "half": h}, open(d / "head_box.json", "w"))
        (d / "mesh").mkdir(exist_ok=True)
        run([av("meshing"), "--input", d / "sfm_pinhole.sfm", "--depthMapsFolder", d / "depthFiltered",
             "--outputMesh", d / "mesh" / "mesh.obj", "--output", d / "mesh" / "densePointCloud.abc",
             "--boundingBox", bbox, "--maxPoints", "5000000", "--minVis", "2", "--colorizeOutput", "1",
             "--verboseLevel", "info"], log)
        (d / "meshF").mkdir(exist_ok=True)
        run([av("meshFiltering"), "--inputMesh", d / "mesh" / "mesh.obj", "--outputMesh", d / "meshF" / "mesh.obj",
             "--keepLargestMeshOnly", "1", "--smoothingIterations", "3", "--verboseLevel", "info"], log)
        run([av("texturing"), "--input", d / "mesh" / "densePointCloud.abc", "--imagesFolder", d / "images",
             "--inputMesh", d / "meshF" / "mesh.obj", "--output", d / "texturing", "--outputMeshFileType", "obj",
             "--colorMappingFileType", "png", "--textureSide", "4096", "--unwrapMethod", "Basic",
             "--verboseLevel", "info"], log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--capture", required=True); ap.add_argument("--frames", required=True)
    ap.add_argument("--poses", choices=["none", "arkit-locked", "arkit-refine"], default="arkit-locked")
    ap.add_argument("--intrinsics", choices=["shared", "perframe"], default="shared")
    ap.add_argument("--lock-intrinsics", action="store_true")
    ap.add_argument("--distortion", choices=["none", "lut", "lut-inv"], default="lut-inv")
    ap.add_argument("--describers", default="dspsift"); ap.add_argument("--preset", default="normal")
    ap.add_argument("--matching", choices=["frustum", "sequential-voctree"], default="sequential-voctree")
    ap.add_argument("--dense", action="store_true"); ap.add_argument("--bbox-half", type=float, default=0.2)
    ap.add_argument("--dm-downscale", type=int, default=2)
    ap.add_argument("--stages", default="init,features,matching,tracks,sfm,dense"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    p = P(a)
    for s in a.stages.split(","):
        if s == "dense" and not a.dense:
            continue
        getattr(p, s)()


if __name__ == "__main__":
    main()
