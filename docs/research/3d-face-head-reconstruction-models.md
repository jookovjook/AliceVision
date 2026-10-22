# Open Multi-View 3D Face & Head Reconstruction Models (2024–2026)

As of Oct 22, 2026. Live version with comments: https://claude.ai/artifact/EPgLDnDXu1JSyhwPEgePuw

## TL;DR

Use **UVFaceFusion** for the face and **MapAnything** for real-world size and the ears. No single open model reconstructs forehead to ears accurately, at true scale, under a commercial licence.

- **[UVFaceFusion](https://github.com/grignarder/UVFaceFusion)** (Tsinghua, July 2026) is the most accurate open model for a face from several photos: about 1 mm error from 4–16 photos with unknown camera positions, in under 3 s. It covers forehead, temples, nose, cheeks and chin, but not the ears, and its output is not to scale.
- **[MapAnything](https://github.com/facebookresearch/map-anything)** (Meta, September 2025, Apache weights) recovers the cameras and a point cloud in real-world units from the same photos, ears included. Its size accuracy at head distance is untested. It is not trained on faces, so it supplies size and the ears, not facial detail.
- **[GNM Head](https://github.com/google/GNM)** (Google, July 2026, Apache-2.0) is a metric full-head model with ears. Fitting it to the two outputs above gives one complete head mesh. Google ships no photo-fitting code, so this step is built in-house.
- **Licensing is the main risk.** Every face-specific model depends on non-commercial parts (the FLAME face model, Pixel3DMM, training data). Prototype with UVFaceFusion now, request licences, and build a licence-clean MapAnything + GNM version in parallel.
- **Coverage:** 192 search hits, about 86 distinct models, 69 checked against GitHub, arXiv and Hugging Face. Several of 2026's most relevant papers, such as Google's SHELLS and Apple's HeadsUp, publish no code or weights.

## Scope and selection criteria

A model made the list only if it passed the four filters below; two borderline building blocks are marked as such. Everything else is listed under Excluded with the reason.

| Filter | Rule |
| --- | --- |
| Recency | First public release (arXiv v1 or first code drop) on or after 2024-10-22, i.e. within the last 2 years |
| Open code | Inference code is public |
| Open weights | Pretrained checkpoints can be downloaded today (gated or sign-up is noted, "coming soon" does not count) |
| Learned at scale | A network trained on a large dataset, not a per-person optimisation with no pretrained model |

Ranking then weighed four things: how well the model uses several photos at once, how faithful the shape is (not just a nice-looking avatar), coverage from forehead and temples down to the chin and out to the ears, and how easy it is to ship (license, dependencies, maturity, GPU cost).

The capture assumed is 8–16 handheld phone photos or a short video around the head (front, three-quarter and profile views), with unknown camera positions. Fourteen lower-relevance generic or generative models were found but not checked one by one: Fast3R, CUT3R, MV-DUSt3R+, MUSt3R, Fin3R, FLARE, HAMSt3R, PEAR, RecGen3D, MV-SAM3D, SAM 3D Body, SyncHuman, DiffPortrait360 and UMI3D.

## Comparison of the qualifying models

Only two models reconstruct a face from several unposed photos and report accuracy on real people: UVFaceFusion and its predecessor VGGTFace. Both stop in front of the ears and lose real-world size.

Rows are ordered by overall fit for this use case. Accuracy figures use each paper's own metric and benchmark, so they are not directly comparable. Most were measured after rescaling the result onto the scan, which hides errors in absolute size. "Unposed" means camera positions are unknown; "metric" means real-world units. NeRSemble, H3DS and NoW are benchmarks built from studio scans. "Commercial" means usable in a paid product without extra licences.

| Model | Released | Photos in | What it outputs | Best reported accuracy | Commercial |
| --- | --- | --- | --- | --- | --- |
| [UVFaceFusion](https://github.com/grignarder/UVFaceFusion) (Tsinghua) | 2026-07 | 4–16 unposed, native | Face mesh, forehead to chin, stops in front of ears; not to scale | 0.91 mm NeRSemble, 1.12 mm H3DS (16 views); 1.06 / 1.31 mm (4 views) | No |
| [MapAnything](https://github.com/facebookresearch/map-anything) (Meta) | 2025-09 | Any number, unposed, native | Cameras plus metric point cloud of everything visible, ears included | No face benchmark | Yes (map-anything-apache checkpoint only) |
| [GNM Head](https://github.com/google/GNM) (Google) | 2026-07 | None; a shape model you fit | Metric full-head model with ears, 17,821 vertices | 0.75 mm fit to 15,000 held-out scans (FLAME 0.97 mm); no accuracy from photos | Yes, Apache-2.0 |
| [RealDenseFace](https://github.com/LinzhouLi/RealDenseFace) (Zhejiang) | 2026-08 | 1 image (optional multi-photo FLAME fit) | FLAME full head; ears and skull from the prior | 1.61 mm NeRSemble single-view (rank 1) | No |
| [VGGTFace](https://github.com/grignarder/vggtface) (Tsinghua) | 2025-11 | Up to 16 unposed, native | Face mesh only; not to scale | 0.98 mm NeRSemble, 1.18 mm H3DS | No |
| [Pixel3DMM](https://github.com/SimonGiebenhain/pixel3dmm) (TUM) | 2025-05 | 1 image (optional multi-photo FLAME fit) | FLAME full head; ears and skull from the prior | 1.66 mm NeRSemble single-view; NoW 0.87 mm median | No |
| [Face Anything](https://github.com/kocasariumut/FaceAnything) (TUM) | 2026-04 | Many unposed, native | Per-photo face depth incl. hair and visible ears; no mesh | Best face depth on NeRSemble and Ava-256 | No |
| [VGGT](https://github.com/facebookresearch/vggt) (Meta, Oxford) | 2025-03 | Unposed, native | Cameras plus point cloud; not to scale | Backbone of both top face methods | Yes, Commercial checkpoint |
| [MOCHI](https://github.com/filby89/mochi) (Athena, MPI) | 2026-05 | Calibrated studio rig | Metric FLAME full head incl. ears | 0.26 mm median, FaMoS studio data | No |
| [MATCH](https://github.com/malteprinzler/match) (ETH, Google) | 2026-03 | Calibrated studio rig | Full head incl. ears | 0.88 mm face, 2.10 mm full head, Ava-256 | No |
| [DAViD](https://github.com/microsoft/DAViD) (Microsoft) | 2025-07 | 1 image | Per-photo relative depth and normals | Normals 17.2° mean error, Goliath-Face | Yes, MIT |
| [KaoLRM](https://github.com/CyberAgentAILab/KaoLRM) (CyberAgent) | 2026-01 | 1 image | FLAME head | NoW val 0.99 mm | No |
| [SHeaP](https://github.com/nlml/sheap) (Toyota, TUM) | 2025-04 | 1 image | FLAME head | NoW 0.95 mm median | No |
| [DenseMarks](https://github.com/diddone/densemarks) (Pozdeev et al.) | 2025-11 | 1 image | Dense full-head correspondences incl. ears; no geometry | 2D only | No |

Every model above was first released after 2024-10-22 and has public code. Two are borderline inclusions: VGGTFace ships no trained network of its own (it optimises per person using VGGT and Pixel3DMM weights), and GNM Head is a statistical shape model, not an image model. MapAnything, VGGT and DAViD are generic building blocks, not face reconstructors. Head-avatar and image-to-3D generators that also passed are covered under Other candidates.

## Top picks in detail

Pick UVFaceFusion for face accuracy and MapAnything for real-world size, cameras and the ears; GNM Head fills in the rest of the head. No single model does all of it.

### 1. UVFaceFusion: the most accurate face from several photos

[Paper](https://arxiv.org/abs/2607.18798) · [Code](https://github.com/grignarder/UVFaceFusion) · [Weights](https://cloud.tsinghua.edu.cn/f/a76f042852bb4191bf2f/) · Tsinghua University, July 2026

- **What it does:** turns 4–16 uncalibrated photos into one face mesh with a fixed 6,986-vertex layout, in under 3 s on an RTX 4090.
- **Why it wins:** best measured accuracy in the study: 0.91 mm on NeRSemble and 1.12 mm on H3DS with 16 views, 1.06 / 1.31 mm with 4 views. It beats VGGTFace, HRN, Pixel3DMM and DECA on the same benchmarks. Its trained fusion step adds measurable accuracy (1.22 mm without it on H3DS).
- **How it works:** VGGT estimates the cameras and depth for every photo. Pixel3DMM predicts which point of a standard face each pixel belongs to. A trained network merges all views in that shared face layout and outputs the mesh.
- **Coverage:** chin, jaw, cheeks, nose, lips, forehead up to about 6.5 cm above the eyes, and part of the temples. The mesh stops in front of the ears, and areas near the ears come from the template, not the photos. No ears, scalp or neck: the code masks ears and hair out of the input.
- **Risks:**
  - Output is not to scale; size must come from elsewhere.
  - Non-commercial dependencies (see Licensing); likely fine for internal evaluation, but confirm with legal.
  - Young code: 7 commits, about 28 stars, "relatively messy" per its README.
  - Two bugs found by reading the code: it center-crops with no face detection, and it only reads `.png` files, so phone `.jpg` photos fail. Both are easy to patch.
  - The face network saw only 256 people from one studio rig. Expression changes between handheld shots will cost accuracy.
- **How to run:** clone the repo; download VGGT-1B and Pixel3DMM's `uv.ckpt` only (skip its FLAME and MICA steps); download `fusion_net.pth` (1.43 GB). Crop each photo around the face and save as `.png`, then run `preprocess.py` and `infer.py`. Output: `result.ply`. Needs a CUDA GPU (bf16 on Ampere or newer, fp16 otherwise).

### 2. MapAnything (Apache weights): real-world size, cameras and the ears

[Paper](https://arxiv.org/abs/2509.13414) · [Code](https://github.com/facebookresearch/map-anything) · [Weights](https://huggingface.co/facebook/map-anything-apache) · Meta and CMU, September 2025

- **What it does:** a general multi-photo 3D model. From unposed photos it predicts every camera and a metric point cloud of everything visible, ears and sides of the head included.
- **Why it is the second pick:** it is the only commercially licensed multi-view model found that outputs real-world size (Apache-2.0 code and weights, not gated). It fills exactly the gaps UVFaceFusion leaves: size and ears. It can also take the phone's focal length from EXIF.
- **Risks:** never trained or tested on faces, so facial detail is smooth and hair comes out as a surface. Input is 518 px, so ear detail is coarse. Its size accuracy at 0.3–1 m from a head is unverified. It assumes the head stays still.
- **How to run:** `pip install -e .` from the repo, then `MapAnything.from_pretrained("facebook/map-anything-apache")` and `model.infer(views)`. Read `pts3d`, `camera_poses`, `intrinsics` and `conf` per photo.

### Supporting piece: GNM Head, to complete the full head

[Paper](https://arxiv.org/abs/2607.23687) · [Code](https://github.com/google/GNM) · [Weights](https://huggingface.co/google/gnm-v3) · Google, July 2026

- **What it is:** a metric 3D model of the whole human head (17,821 vertices) with labelled regions for ears, forehead, temples, nose, cheeks and chin. It was built from about 150,000 scans of about 5,000 people, and fits scans better than FLAME (0.75 mm vs 0.97 mm).
- **Role:** fit it to the UVFaceFusion face plus MapAnything's ear and side points to get one metric full-head mesh. It is also the core of a fully commercial pipeline: Apache-2.0, no FLAME.
- **Risk:** Google released the model but no code to fit it to photos, so that fitting step has to be built in-house.

## Other candidates and why they ranked lower

RealDenseFace is the best fallback if fewer than about 4 usable photos exist. Everything else is either weaker on measured shape, needs a studio rig, or invents the parts it cannot see.

| Model | Best use here | Why not a top pick |
| --- | --- | --- |
| [RealDenseFace](https://arxiv.org/abs/2608.09238) | Fallback when only 1–3 photos are usable; 65–180 ms per image | Single-image model; shape limited to FLAME, ears from the prior. Released version lacks the MICA step (2.09 mm vs 1.86 mm with it). Multi-photo mode only reuses frame 0's pose. Default focal range likely misses phone portraits. MIT badge, but FLAME and non-commercial training data underneath |
| [VGGTFace](https://arxiv.org/abs/2511.20366) | None; superseded | Same authors as UVFaceFusion and worse on every benchmark (e.g. 1.18 vs 1.12 mm H3DS); no ears; repo frozen |
| [Pixel3DMM](https://arxiv.org/abs/2505.00615) | Its UV predictor runs inside UVFaceFusion | Paper says it cannot use multi-view information; multi-image mode is unbenchmarked. FLAME fits from many views reach only about 1.45 mm. Minutes per subject. CC BY-NC plus MICA and InsightFace at runtime |
| [Face Anything](https://arxiv.org/abs/2604.19702) | Non-commercial alternative to MapAnything for side-of-head depth | CC BY-NC code, weights and backbone; point clouds only, no mesh; scale uncertain; 15.5 GB checkpoint |
| [VGGT-1B-Commercial](https://huggingface.co/facebook/VGGT-1B-Commercial) | Camera and depth backbone, A/B against MapAnything | Not to scale; gated; health-use clause needs legal review |
| [Depth Anything 3](https://arxiv.org/abs/2511.10647) (Apache Small and Base) | A/B against MapAnything for cameras and shape | Apache checkpoints are not to scale; metric multi-view needs the CC BY-NC Nested model; worse face depth than VGGT, Sapiens and DAViD in Face Anything's tests |
| [VGGT-Ω](https://arxiv.org/abs/2605.15195), [π³ / Pi3X](https://arxiv.org/abs/2507.13347) | Stronger generic geometry | Weights non-commercial; faces were blurred or filtered out of VGGT-Ω's training data |
| [MOCHI](https://arxiv.org/abs/2605.01450), [MATCH](https://arxiv.org/abs/2603.15811) | Full head incl. ears in a studio | Need calibrated, synchronised studio cameras; non-commercial; MATCH's ear correspondence error is 9.1 mm |
| [DAViD](https://arxiv.org/abs/2507.15365) | MIT per-photo normals as an extra fitting term | One photo at a time, relative depth only, no size; repo inactive since July 2025 |
| [MoGe-2](https://arxiv.org/abs/2507.02546) (Microsoft) | MIT backup to DAViD; estimates focal length when EXIF is missing | One photo at a time; no face evaluation |
| [DenseMarks](https://arxiv.org/abs/2511.02830) | Ear correspondences for fitting | No geometry of its own; CC BY-NC weights |
| [Sapiens2](https://arxiv.org/abs/2604.21681) (Meta) | None | Licence bans any use "for biometric processing" and restricts health inference; one photo at a time |
| [WorldMirror 2.0](https://arxiv.org/abs/2604.14268), [Hunyuan3D-2mv](https://arxiv.org/abs/2501.12202) (Tencent) | None | Licence excludes the EU, UK and South Korea, outputs included; health clauses; not to scale; Hunyuan3D invents unseen parts |
| OmniVGGT, HyDen, Dens3R | None | Effectively non-commercial (VGGT-1B fine-tune, FAIR Noncommercial, CC BY-NC-SA); no face evidence |
| SHeaP, KaoLRM, TEASER | Quick single-photo FLAME shape | One photo only; FLAME-limited (NoW about 0.92–0.99 mm median); non-commercial or no licence |
| UIKA, FlexAvatar (Kirschstein et al.), FastAvatar (two papers), LAM, CAP4D, OMG-Avatar, LHM++ | Animated avatars | Optimised for how the head looks, not its shape; no geometry accuracy reported; non-commercial parts |
| FaceLift, HQ-Head, SOAP, TOPOS, TRELLIS, TRELLIS.2, ReconViaGen, LSRM, Pixal3D, SpinMeRound | Plausible full-head meshes | Generators that invent the sides, ears and back; scale normalised; no face accuracy reported |

## Excluded models

Several of the most relevant 2026 papers are not usable: most publish no code or no weights. Checked on 2026-10-22.

| Model | Released | Why excluded |
| --- | --- | --- |
| [SHELLS](https://arxiv.org/abs/2605.31283) (Google) | 2026-05 | Multi-view full head incl. ears, very relevant; no code or weights |
| [HeadsUp](https://arxiv.org/abs/2605.04035) (Apple) | 2026-05 | Multi-view Gaussian heads; no code or weights; needs a studio rig |
| [GHARP](https://arxiv.org/abs/2610.10945) (Apple) | 2026-10 | No code or weights; built on unreleased HeadsUp |
| [Skullptor](https://arxiv.org/abs/2602.21100) (Ubisoft) | 2026-02 | Maintainer confirmed the trained model will not be released |
| [GLVD](https://arxiv.org/abs/2510.06046) (Crisalix) | 2025-10 | Face shape for aesthetic medicine; no code or weights, private data |
| [CUBE](https://arxiv.org/abs/2604.12894) (Google) | 2026-04 | No code or weights |
| [Pix2NPHM](https://arxiv.org/abs/2512.17773) (TUM) | 2025-12 | Repo says "code coming soon"; no weights |
| [Avat3r](https://arxiv.org/abs/2502.20220) (TUM, Meta) | 2025-02 | Empty repo; no weights |
| [PercHead](https://arxiv.org/abs/2511.02777) (TUM) | 2025-11 | README-only repo for a year; single image |
| [Multiple View Neural Regression](https://arxiv.org/abs/2608.22655) (Clemson) | 2026-08 | Dissertation; no code, proprietary data |
| [ProfileSynth](https://arxiv.org/abs/2605.01746) (Keio) | 2026-05 | "Code coming soon"; small synthetic training set |
| [FNR2R](https://arxiv.org/abs/2601.01950) (Tianjin) | 2026-01 | Face normals only; small-data method, not trained at scale |
| [Pippo](https://github.com/facebookresearch/pippo) (Meta) | 2025-02 | Code only, no pretrained weights per its README; outputs images, not 3D |

Head-avatar papers with no code or weights: FlexAvatar (Peng et al.), FFAvatar (two different papers), FastGHA, MeshLAM, FHAvatar, SpatialAvatar-0, FA-LAM, FiCA, 3DRealHead, Any3DAvatar, BeautyNet. They also target appearance rather than measured shape. ImHead, a full-head shape model (ICCV 2025), released only a dataset, with no model or code.

Too old (first released before 2024-10-22), even though widely used: DECA, EMOCA, MICA, HRN, 3DDFA-V3, SMIRK, MonoNPHM, DUSt3R, MASt3R, Sapiens v1, GAGAvatar and VHAP. Several of them are still used inside the newer models listed above.

## Licensing and shipping caveats

No face-specific model found can ship in a paid product as-is. Each one depends on a non-commercial face model, non-commercial weights, or weights trained on non-commercial datasets.

- **FLAME** (the face model behind most of these methods) is under an academic licence. It bans commercial use, redistribution, and training networks for commercial use. Commercial licences come from MPI (ps-license@tue.mpg.de). FLAME 2023 Open (Nov 2025) is CC-BY-based with extra restrictions. It is not a drop-in swap: expression parameters need conversion, and no recommended model has been tested with it.
- **Training data** behind the best face models is non-commercial: Ava-256 (CC BY-NC 4.0), FaceScape and NPHM (research only). A licence from a model's authors may not cover this.
- **UVFaceFusion** needs four rights holders cleared: Tsinghua for the fusion weights (no licence stated), TUM for the Pixel3DMM UV predictor (CC BY-NC 4.0), Meta for VGGT, and MPI for the FLAME-derived face template. Its face parser was trained on CelebAMask-HQ (non-commercial), so it needs replacing. Its code is GPL-3.0, which is fine on a server but forces source release if shipped inside an app.
- **VGGT-1B-Commercial** is gated. Its acceptable use policy bans the unlicensed practice of medical or health professions and inferring health information without consent. A health product needs legal review of that clause.
- **MapAnything**: only `facebook/map-anything-apache` and its archived -v1 copy are Apache-2.0. `facebook/map-anything` and `-v1` are CC BY-NC 4.0.
- **Misleading licence badges.** RealDenseFace (MIT label), LAM, UIKA and FastAvatar show MIT or Apache badges but embed FLAME assets, CC BY-NC weights or non-commercial code. Legal should review the full dependency chain, not the repo badge.
- **Size.** None of the face models that work from uncalibrated photos outputs real-world size; their millimetre errors are measured after rescaling. Plan a scale source: phone depth (TrueDepth or LiDAR), a card of known size in one photo, or a metric model such as MapAnything.
- **Fragile hosting.** UVFaceFusion weights sit on a Tsinghua share link; RealDenseFace, MATCH and MOCHI on Google Drive. Mirror them with checksums and save the licence text on the download date.
- **Biometric data.** Face geometry is biometric data in many jurisdictions, so plan consent, retention and deletion. DAViD's model card warns against using its output as the sole input for medical decisions; the same applies to every model here. Meta's Sapiens2 licence bans biometric processing outright, and Tencent's licences (WorldMirror, Hunyuan3D) do not apply in the EU, UK or South Korea.

## Recommended pipeline and next steps

Run two tracks from the same phone capture. Track A gives the best accuracy now, for internal testing. Track B uses only commercially licensed parts, so it can ship if the licences for Track A never come through.

![Recommended pipeline: one capture, two tracks, one licence decision](face-head-reconstruction-pipeline.png)

Both tracks end in the same metric full-head mesh layout (GNM), so the product code downstream does not change when you switch.

Next steps (rough estimates, not measured):

1. **Week 1:** run UVFaceFusion, MapAnything (with a scale check against the card), Pixel3DMM and RealDenseFace on 10–20 internal phone captures to see real-world quality. Patch UVFaceFusion's crop and `.jpg` bugs first.
2. **Week 1:** request licences from Tsinghua (UVFaceFusion weights), TUM (Pixel3DMM `uv.ckpt`), Meta (VGGT-1B-Commercial) and MPI (FLAME), and have legal review the VGGT health clause.
3. **Week 1:** test whether UVFaceFusion still works with the VGGT-1B-Commercial checkpoint. If it does, Meta's part is covered by the self-serve Commercial licence; if not, Track A needs retraining or a separate deal with Meta.
4. **Weeks 1–3:** collect ground truth: scan 15–30 varied people, ears included, with structured light or a calibrated multi-camera photogrammetry rig (AliceVision/Meshroom can produce this).
5. **Weeks 2–4:** build the GNM fitting step (one shape shared across all photos, point and landmark terms). Both tracks use it.
6. **Capture app:** add on-device checks for blur, face found in every frame and enough side-to-side coverage, so retakes happen before upload. Have the phone move around a still head; don't have the person turn in front of a fixed phone.

## Sources

All pages below were opened on 2026-10-22. Links for the other models are in the tables above.

- UVFaceFusion: [paper](https://arxiv.org/abs/2607.18798), [code](https://github.com/grignarder/UVFaceFusion), [licence](https://raw.githubusercontent.com/grignarder/UVFaceFusion/main/LICENSE), [weights](https://cloud.tsinghua.edu.cn/f/a76f042852bb4191bf2f/)
- MapAnything: [paper](https://arxiv.org/abs/2509.13414), [code](https://github.com/facebookresearch/map-anything), [Apache weights](https://huggingface.co/facebook/map-anything-apache), [licence fix, issue #139](https://github.com/facebookresearch/map-anything/issues/139)
- GNM Head: [paper](https://arxiv.org/abs/2607.23687), [code](https://github.com/google/GNM), [weights](https://huggingface.co/google/gnm-v3), [landmark mapping, issue #53](https://github.com/google/GNM/issues/53)
- VGGTFace: [paper](https://arxiv.org/abs/2511.20366), [code](https://github.com/grignarder/vggtface), [AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37754)
- Pixel3DMM: [paper](https://arxiv.org/abs/2505.00615), [code](https://github.com/SimonGiebenhain/pixel3dmm), [weights](https://huggingface.co/SimonGiebenhain/Pixel3dmm)
- RealDenseFace: [paper](https://arxiv.org/abs/2608.09238), [code](https://github.com/LinzhouLi/RealDenseFace), [project page](https://gapszju.github.io/RealDenseFace/)
- Face Anything: [paper](https://arxiv.org/abs/2604.19702), [code](https://github.com/kocasariumut/FaceAnything)
- VGGT: [paper](https://arxiv.org/abs/2503.11651), [code](https://github.com/facebookresearch/vggt), [licence](https://raw.githubusercontent.com/facebookresearch/vggt/main/LICENSE.txt), [Commercial checkpoint](https://huggingface.co/facebook/VGGT-1B-Commercial)
- Depth Anything 3: [paper](https://arxiv.org/abs/2511.10647), [code](https://github.com/ByteDance-Seed/Depth-Anything-3)
- Sapiens2: [licence](https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md)
- Not released: [SHELLS](https://arxiv.org/abs/2605.31283), [HeadsUp](https://arxiv.org/abs/2605.04035), [HeadsUp repo (project page only)](https://github.com/apple/ml-headsup)
- [FLAME model licence](https://flame.is.tue.mpg.de/modellicense.html)
