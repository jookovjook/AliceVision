# MRI face vs scanned face: where they disagree, and what to change (revised)

**TL;DR**
1. Two mismatches matter more than face-surface agreement. FaceMesh puts landmarks 5–6 mm apart on the MRI render vs the phone scan (+0.58/+0.80 mm mean error). The back and temporal screw tips sit 4–10 mm inside the MRI skin (+0.47 to +0.76 mm) **[sim, Jack's n=1 patterns]**.
2. I applied Jack's landmark field, his tip pattern and his ear offset to every test head at once. Mean error went from 2.73 to **5.29 mm** (p95 8.3), which is more than the parts add up to (+1.55). Device frame, frame augmentation and no-face variants all score 5.28–5.45. Only dropping sag7+8 helps (4.31) **[sim, new run]**.
3. The expert's thin-tissue idea is only **weakly** supported on Jack (Spearman ρ 0.24–0.45). Nose cartilage agrees well, but zygoma and preauricular skin are off by 1.7 mm (MRI pads). Zone choice alone moves the targets 0.3–2.5 mm **[n=1]**.
4. **The ear-axis idea is untested.** Device-frame output ties on clean data (2.706 vs 2.733) and gains only ~10 % on Jack's real landmark field. Removing the face costs +0.53 mm (3.24), because under today's isotropic placements the face pose is what locates the frame **[sim]**.
5. Jack's readings are out of distribution: Mahalanobis² (M², distance from the training cloud) is 71, against an empirical p99 of 35. His head shape and seat are inside the cloud (18–23). The tip-vs-skin offsets push the readings out **[n=1]**.
6. One subject cannot measure accuracy: older-model variants with CV 2.9–3.5 mm gave 1.9–5.5 mm on Jack. Measure each component without an MRI, then run the measured errors through the 336-head simulator.

**Markers.** **[n=1]** = measured on Jack only (one head, one frame-on capture, two TrueDepth scans, one MRI). **[sim]** = 336-head leave-one-site-out simulator (free model unless named). **[new]** = run for this revision, read-only, results in REV. **[lit]** = literature. **[est]** = my arithmetic, not run. **[unverified]** = nobody has checked it. Jack is now a **development subject**: his patterns were used to choose the remedies, so he cannot also check them. Path aliases are at the end.

---

## 1. The problem in one picture

```
            TRAINING (336 MRI heads, simulated)                     USE (new person, no MRI)
 face    MRI skin: supine, eyes closed (Jack; sites unknown), pads    phone scan: upright, eyes open
         -- render --> FaceMesh <------------- [1] -------------->  FaceMesh <-- photo / shaded render
 frame   "face frame" = Kabsch of ~430 landmarks <---- [1] ------>  "face frame" from scan landmarks
 seat    isotropic 6-dof cloud around rule A <-------- [6] ------>  real seat: operator + clamping,
         (balls ~7 mm 3-D off the canals, collisions ignored)        nose piece at nasion/glabella
 screws  gap = ray to RIGID outer skin <-------------- [2] ------>  tip pressed into tissue / hair
 ears    canal-mouth centroid <----------------------- [3] ------>  ball centre, 5-8 mm higher (n=1)
 geom    rigid CAD, exact scale <--------------------- [7,8,9] -->  arch flexed 2.4-2.6 deg, scale unknown
 output  targets in face frame (truth also there)                   face frame --[4]--> device --> aim
```

There are **two separate alignment problems**:
- **Deployment (no MRI).** The inputs and the frame must be computed the same way on MRI and on scan. Errors here reach every patient.
- **Validation (subject with an MRI).** The MRI has to be placed on the scan to get the true targets. On Jack this placement alone is uncertain by 1–4 mm, as large as the model error.

**No accuracy requirement is stated in any file I read.** Every number below is a mean over heads. A targeting device needs a per-target tolerance, split along and across the beam axis. I added per-target p95 to the new runs. On clean input the worst target is the brainstem, with p95 6.4 mm, and 6 % of head-target errors exceed 5 mm **[sim, new]**.

### Error budget: deployment chain

**Δ model error** is the increase in mean 3-D error over 336 test heads when only the test inputs are perturbed. Clean is 2.733 mm. **Target shift** is how far the predictions move.

| # | Mismatch | Measured size | Type | Target shift | Δ model error | Evidence |
|---|---|---|---|---|---|---|
| 1 | FaceMesh landmarks, MRI render vs phone | 5.1–5.7 mm median per landmark (jaw/chin 18.7); frame tilt 1.9–2.6° **[n=1]** | systematic | 1.6–3.0 mm | **+0.58** photo / **+0.80** shaded. The transplant adds Jack's head-frame field to heads of other shapes, an unchecked approximation | STUDY/free_sens.json `jack_field_*`; ME/align.json `landmarks` |
| 2 | Screw-tip contact (and/or MRI placement error). **Two estimates.** *Along-axis* (measured gap minus rigid gap along the screw, which is how the model reads a gap, MODEL/scripts/arc_free_poses.py:31-41): sag7/8 6.5–9.7 mm inside, cor1/cor4 4.3–5.0 inside, front 1.3–5.2 outside. *Closest-point* (S/mri_tipfit/tip_fit.json): sag7/8 5.2–9.3 inside, cor1/cor4 1.7–3.0 **[n=1]** | systematic; varies by head (sag8 tissue 6–30 mm, p5–p95) | 1.8–3.5 mm, mostly forward (accumbens/amygdala +1.8–2.9 mm) | Along-axis: **+0.76** (tip-fit placement), +1.63 (face-only placement, which puts front tips impossibly outside the skin). Closest-point: +0.47 / +1.20 | STUDY/free_model_bias.json; free_sens.json `screws_jack_*`; REV/extra_base.json |
| 3 | Ear point: ball centre vs canal-mouth centroid | 5.4 / 8.0 mm above, 0.4–1.0 mm medial, +3.3 / −4.5 mm in y **[n=1; MRI placed by face-only bridge, about ±1°]** | systematic | 0.7 mm at 4 mm up | 4 mm up +0.08; **6.7 mm up +0.22; 8 mm up +0.31** **[new]**. LPA/RPA as ears: +1.03 | free_model_bias.json; REV/extra_base.json; STUDY/ears_com.json |
| 4 | Face-to-frame pose ("gauge") | Face 1.2 mm; targets 1.8 mean / 2.0 max mm between two pose pipelines (0.56°). This is a disagreement between pipelines, not an error against truth **[n=1]** | systematic per capture | ~1:1 into the aim | **Not scored** in the face-frame model: truth stays in the face frame (MODEL/qc_viewer/error_ladder_v2.py:565). **[est]** +0.2–0.5. The device-frame model counts it: +0.05 at 1.5 mm, +0.18 at 3 mm **[sim]** | STUDY/gauge_probes.py; free_sens_devframe.json |
| 5 | Readings outside the training cloud | M² 70.8 vs empirical p99 35.2 (size 116). Rule-A gaps 20.2; rigid gaps at the real pose 18.3 / 22.7 **[n=1]** | systematic per person | unknown | unknown (linear extrapolation) | REV/ood_decomp.txt; STUDY/critic/ood_check.txt |
| 6 | Real seat vs rule A | 2.3° (1.6° nose-up), +4.7 mm up **[n=1]** | systematic | inside the 10 mm training spread | ≈ +0.03 **[est, proxy from the random-spread sweep; no systematic-offset test]** | STUDY/seat_jack.json |
| 7 | Device-frame definition, scan vs simulation | Ball midpoint 2.9 mm and ball axis 1.95° off the tag-fit frame **[n=1]** | systematic | 1:1 if the frames get mixed **[est 1–3 mm]** | +0.05–0.13 as a pose-input bias (arcpose-sweep proxy) | HS/tag_poses.py:326-337 vs MODEL/scripts/frame_rigid_geom.py:3-4 |
| 8 | Frame flex | Arch bends 2.4–2.6° worn vs 0.8–1.6° on the table; tips 0.5–3.5 mm off the rigid CAD **[n=1]** | systematic per seat | unknown | unknown | STUDY/flex_tips.py; S/fastres/6ba4b7d84f/cad/cad_fit.json |
| 9 | Scale | TrueDepth ×1.032–1.035 (anisotropic: lateral 1.9 %, depth correct); print −0.4 % (inferred, not measured); free face-fit scale 0.99–1.06 **[n=1]** | systematic per phone/app/print | 0.6–1.6 mm per 1 % (rigid fit); 2.4–3.8 mm if fitted freely | ±3 % on face+ears: +0.28–0.32 | quant/align_study.txt, align_extra.txt (d); free_sens.json `scale_*` |
| 10 | Supine vs upright face | Tissue < 8 mm: 0.3–0.5 mm (in-sample); lateral face 1.3–2.1 mm wider supine; chin +3.6; jaw angle −3.9 **[n=1 + lit]** | systematic | 1.0–1.2 mm for a landmark frame **[est]** | inside row 1 | quant/thickness.json |
| 11 | Skin segmentation definition | charm vs threshold skin | systematic per site | — | +0.50 | MODEL/results/ladder/v2/free/sec_skin.json |
| 12 | Screw readout | 0.32 mean / 0.60 max mm vs a ruler with 0.5 mm steps; turn-slip margin only 0.3 mm on sag1 **[n=1]** | random; rare 2.5 mm slip | — | +0.01 | readouts.json; HS/apply_calibration.py:28-30 |
| — | Floors | Brain shift prone→supine 0.5–0.8 mm (supine→upright not measured; phantom ≤ 2 mm) **[lit]**; MRI distortion ~1 mm **[lit]**; truth definition 2.74 / 2.82 / 2.90 **[sim]**; skull acoustic aberration not quantified | — | — | — | — |

### Joint stress test (new)

Every test head received Jack's photo landmark field, the along-axis tip-fit tip pattern and the ears moved up 6.7 mm, all at once **[sim, new; REV/extra_*.json]**.

| Model | Clean mean | Joint mean | Joint p95 over heads | Worst-target p95 | Head-target errors > 5 mm (clean → joint) |
|---|---|---|---|---|---|
| Face frame (today) | 2.733 | **5.29** | 8.3 | 9.9 | 6 % → 53 % |
| Device frame (free_dev) | 2.706 | 5.28 | 8.2 | 9.7 | 6 % → 52 % |
| Device frame, **no face** | 3.240 | 5.39 | 8.6 | 9.9 | 12 % → 56 % |
| Face frame + frame augmentation | 2.732 | 5.45 | 8.4 | 9.8 | 6 % → 57 % |
| Face frame, **no sag7/sag8** | 2.808 | **4.31** | 7.4 | 9.0 | 7 % → 33 % |

- **The terms add more than linearly.** The single terms sum to +1.55, but together they cost +2.56, because the shifts point the same way. The draft's "4–5 mm, upper end" was wrong.
- **This is a stress test, not a prediction.** It assumes every head carries Jack's exact biases. The tip pattern contains MRI-placement error, and the landmark field and tip pattern come from different MRI placements.
- Adding a 2 mm gauge error to the device-frame joint case adds only +0.04 (5.32).

### Validation chain: placing Jack's MRI on his scan

| Choice | Target shift **[n=1]** | Evidence |
|---|---|---|
| Face-only vs tip-constrained placement | **3.0 mm** median (2.4–3.5); 0.8–4.2 across the assumed compression and weights. The correction is mostly translation (+2.0 mm forward, +2.2 mm up) plus 1.37° (pitch −0.64, roll 0.95, yaw −0.77) | quant/tipfit_targets.txt/.json; S/mri_tipfit/tip_fit.json `correction` |
| Which face zones are fitted (rigid) | 0.3–2.5 mm (0.6–1.5°) | quant/align_extra.txt (a) |
| Free scale on face patches | +2.4–3.8 mm | quant/align_study.txt |
| FaceMesh Kabsch vs surface ICP | 2.3 mm deep, 6.2 mm at the back | ME/align.json `start_vs_icp_mm` |

Jack's truth in device coordinates is therefore uncertain by **1–4 mm**. His targets also come from a provisional MI-SyN registration (ME/targets_mri_mi.json).

---

## 2. Alignment and a common coordinate frame

### Recommendation

1. **Output frame = the segment that carries the transducer.** Code one definition and share it between the simulation and the scan pipeline. No file I searched says which segment carries the transducer (no "transducer" hit in HS). If it is part B, the worn flex (2.4–2.6°; A–B joint 1.3° / 3.3 mm) enters the aim at about 1:1.
2. **The device frame only approximates an ear-canal axis.** On Jack the ball axis is 1.95° and 2.9 mm off the tag frame. Against the canal-mouth axis, the ball centres sit 5.4 / 8.0 mm higher and +3.3 / −4.5 mm apart in y, which is about 3.3° of yaw and 1.1° of roll **[n=1, est from ears_com.json]**. Knob tilt adds 1.5–4.5 mm of ball position in y/z, and the two pipelines differ by 2.7 mm in ear half-width. Sub-mm applies only to the knob travel K (~0.2 mm), not to where the axis is. The medial offset is also unresolved: a 13 mm ball (frame_rigid_geom.py:15) should sit *lateral* of a 7–9 mm canal mouth, so either the MRI placement is off or the ball rests on the concha or tragus. **The scan sees only the ball, never the canal.**
3. **If face inputs stay, use a rigid surface fit with thickness weights at a fixed scale.** Use the **same population-atlas weights on MRI and on scan**. Landmarks only initialise the fit.
4. **Take scale from the frame tags/CAD after a caliper check.** Correct TrueDepth through its intrinsics, not one isotropic factor.

### 2a. What the frame choice does

Values are Δ mean error in mm **[sim]**. The no-face column is **[new]**.

| Test-time perturbation | Face frame (today) | Device frame | Device frame, no face | Face frame + augmentation |
|---|---|---|---|---|
| Clean (mean) | 2.733 (CI 2.63–2.85) | 2.706 (CI 2.60–2.83), **a tie** | **3.240** | 2.732 |
| Face frame rotated 3° | +0.43–0.65 | +0.26–0.38 | 0 | +0.03–0.14 |
| Random face rotation, SD 2° | +0.67 | +0.43 | 0 | +0.13 |
| Face shifted 3 mm laterally | +0.54 | +0.32–0.35 | 0 | +0.12–0.14 |
| Jack's landmark field, photo / shaded | +0.58 / +0.80 | +0.51 / +0.72 (only ~10–12 % lower) | 0 | +0.37 / +0.53 |
| Face→device gauge 1.5 / 3 mm | not counted | +0.05 / +0.18 (counted) | 0 | not counted |
| Jack tips, closest-point / along-axis (tip fit) | +0.47 / +0.76 | +0.56 / +0.90 | +0.75 / +1.14 | +0.58 / +0.97 |
| Ears 6.7 mm up | +0.22 | +0.21 | **+0.65** | +0.33 |
| Joint stress test (absolute) | 5.29 | 5.28 | 5.39 | 5.45 |

Sources: STUDY/free_sens.json, free_sens_devframe.json, free_sens_aug_rot1.5_sh1.5.json; REV/extra_{base,dev,dev_noface,aug}.json.

- **Device frame.** Tips, ears, pose and truth are in the device frame. The scalp map is still fitted in the face frame, and face PCA is unchanged (STUDY/free_dev.py:1-8). The gain comes from the **output** no longer depending on the face fit, not from ears. Ears carry little weight: all-but-ears costs only +0.016 (sec_ablate.json). Synthetic face errors drop about 40 %, but Jack's real field drops only 10–12 %.
- **No-face model (ears + tips, device frame).** It is immune to every face error by construction, which I confirmed (face rotation, landmark field and gauge all give Δ 0). But it costs +0.53 clean and leans much harder on ears and tips. A face-shape-only variant gives the same 3.25, so the information sits in the **face pose relative to the frame**. That is an artefact of the isotropic training placements: the shift SD is 4.1 mm/axis, so balls sit ~7 mm (3-D) off the canals, and 2.3 % of readings are negative. **The decisive next test:** simulate placements with the balls held in a canal seat (pitch about the ball axis plus a small offset distribution) and rerun no-face vs with-face.
- **Augmentation** cuts synthetic face sensitivity 3–8× at zero clean cost. It worsens tips (+0.76 → +0.97) and ears (6.7 mm up +0.22 → +0.33; 5 mm down +0.13 → +0.20).
- **Lever arm.** 1° about the ear-axis midpoint moves the targets 0.46–0.63 mm; about the face centroid, 0.89–1.46 mm (quant/lever_arm.json). The canal axis still needs ~1 mm point accuracy, and surface concha points were off by 7.7 / 9.4 mm, which gives 8.9 mm at the targets.

### 2b. Face-fit recipe, identical on MRI and scan (only if face inputs survive §2a's test)

1. **Initialise with FaceMesh.** On Jack, the FaceMesh-only pose is **3.47°** from the surface fit, at the edge of the tested 3° / 3 mm convergence basin. A direct MRI→head ICP once slid to a wrong pose about 26° away. Test starts of 5° or more and add a wrong-attractor check.
2. **Weight each vertex by w = 1/(σ² + (κ·t)²)**, with σ ≈ 0.5 mm, κ ≈ 0.1–0.2 and t = tissue thickness from **one population atlas on the template**, carried over by the initial fit. Use the same weights on both sides. Charm misses the nasal bones (bridge 11.1 mm vs an expected 2–4 mm), so the 6/8 mm masks are uncertain by ~1 mm.
3. **Exclude** eyes, mouth, jaw/chin, face edge, ears, neck, hair, and about 15 mm around the nose pad. Do not trust bone distance alone:
   - The nasal dorsum is cartilage (23 mm "thick") yet agrees best (0.20 mm).
   - The **zygoma (−1.71 mm) and preauricular skin (−1.75 mm) are biased** by MRI pads, even inside the fit.
4. **Rigid point-to-plane ICP at a fixed metric scale.**
5. **QC:** check residuals on held-out zones, not in-sample ones. Mongen & Willems 2019 found that a navigation system's own error estimate did not track the true error (R² = 0.04) **[lit]**.

**Zone sets on Jack** **[n=1]** (quant/align_study.txt, align_extra.txt, tips_vs_variants.txt):

| Zone set | Points | Residual med/p90 mm (in-sample) | Target mm per mm, thickness-weighted deformation | Same, uniform deformation | Worst-case leverage (mm per mm RMS) | Front tips outside skin, mm |
|---|---|---|---|---|---|---|
| A: full face | 31347 | 1.11 / 3.85 | 0.90 | **1.01** | **3.53** | 4.20 |
| B: tissue < 6 mm | 4163 | 0.37 / 0.91 | 0.50 | 1.37 | 5.56 | 2.14 |
| B8: tissue < 8 mm | 6905 | 0.41 / 1.15 | 0.57 | 1.43 | 4.98 | 2.76 |
| C: forehead + nasion + bridge | 3675 | 0.34 / 0.82 | 0.83 | 1.65 | 6.94 | 2.67 |
| D: "reliable" (current) | 10574 | 0.47 / 1.29 | 0.91 | 1.37 | 4.20 | 3.87 |
| E1: B + preauricular | 4396 | 0.39 / 1.00 | **0.44** | 1.21 | 5.42 | 1.98 |
| E2: E1 + ears | 6895 | 0.67 / 3.55 | 0.71 | 1.12 | 4.19 | 3.75 |

Reading the table:
- **The residuals are internal consistency only.** Small, smooth patches always fit tighter, so a low residual does not mean a better pose.
- **The front-tip column is weak evidence.** The FaceMesh-only pose scores 1.49, better than every surface fit.
- **E1's deformation advantage is synthetic.** On real data its preauricular skin is biased by 1.75 mm.
- **The two families disagree.** {A, D, E2} and {B, B8, C, E1} differ by 1.1–2.5 mm at the targets, and one subject cannot say which is right.
- **The soft weighting is untested.** Only hard zone sets were run, and none in the 336-head simulator.

**Expected deep-target error from a face fit [est, n=1].**

| Component | Size |
|---|---|
| ICP noise | 0.05–0.15 mm |
| Deformation | 0.44–0.91 mm per mm (thickness-weighted) or 1.0–1.65 (uniform) |
| Zone choice | up to 2.5 mm |
| Bony-forehead bias outside the fit | +1.27 mm: the Oct-21 TrueDepth sits outside the MRI skin at z 20–45 (forehead_band.py) |
| **Total** | **about 0.5–2.5 mm** |

Literature: Shamir 2009 gives 2.0 / 3.2 mm at 60 / 105 mm from the face, i.e. about 2.3–3.0 mm at Jack's 72–98 mm **[lit, interpolated]**.

**Zones hidden by the frame.** At deployment the nose piece and arc cover the nasion, glabella and midline forehead (HS/align_face_scan.py:134-135, :146-152).
- Only a **frame-off** TrueDepth scan recovers these zones, bridged through the lateral forehead band (±0.45 mm out-of-fit agreement **[n=1]**).
- A frame-on TrueDepth scan is still useful, but for a different purpose: gauge and scale from the tags.
- Without the covered zones, little thin tissue is left. Even patch C, which includes them, has a 13° weakest mode. This argues for settling the no-face question first.

### 2c. What not to use

| Option | Why not |
|---|---|
| FaceMesh Kabsch as the frame (error_ladder_v2.py:327-340) | 1.9–2.6° between MRI and phone renders; photo vs shaded render of the same scan differs by 3.4 mm at the targets. Restricting the landmarks gives inconsistent results (1.2–7.3 mm depending on subset, render and script), and no subset is reliably better **[n=1]** |
| Surface-detected ear/concha points | 7.7 and 9.4 mm apart between scan and MRI, giving 8.9 mm at the targets **[n=1]** |
| Free scale in any face fit | Small patches drift to ×1.032–1.035, giving +2.4–3.8 mm **[n=1]** |
| LPA/RPA as ear points | The canal mouths are 16–17 mm lower and ~7 mm medial; cost +1.03 mm **[sim]** |
| Cheeks, jaw, lips, zygoma vs a supine MRI | 1.3–3.9 mm disagreement **[n=1]**; buccal tissue moves 3–7 mm between postures **[lit, Hernandez 2022, Ozsoy 2015]**. The literature's "zygoma is reliable" (Jayaratne 2012) comes from same-posture CBCT/photo pairs |

Literature for comparison **[lit]**:
- Surface fit beats landmarks in a TMS simulation: 1.0 vs 2.9 mm (Nieminen 2022).
- In the clinic, however, surface matching was 2× worse than adhesive markers: 5.35 vs 2.49 mm (Mongen & Willems 2019).
- Midline tissue: supraglabella 4.3–5.2 mm, nasion 7.4–9.0, philtrum 12.7–16.4 (Open Dent J 11:375).

---

## 3. Model changes: make training see the scan's degrees of freedom

| Degree of freedom | Plausible range | How to simulate | Status |
|---|---|---|---|
| Output frame | — | Device/transducer frame (STUDY/free_dev.py) | Tested: a tie on clean data (2.706); counts the gauge (+0.05 at 1.5 mm); **no help on the joint case** (5.28) |
| No face input | — | Ears + tips in the device frame | **Tested [new]**: 3.24 clean (+0.53); immune to face errors; ear 6.7 mm up costs +0.65. Must be retested with canal-anchored placements |
| Placement prior | Real seat: balls in the canals, clamped; Jack 1.6° nose-up, +4.7 mm up **[n=1]** | Rotation about the ball axis + a small seat-offset distribution; treat collisions; width taken from re-seat data | **Not tested.** Today: isotropic, balls ~7 mm 3-D off the canals, 2.3 % negative readings |
| Face-frame rigid error | 1.9–2.6°, ~1 mm **[n=1]** | Rotation SD 1.5°, shift 1.5 mm, applied to tips, pose, ears and truth | Tested: clean 2.732; synthetic sensitivity 3–8× lower; Jack field +0.58 → +0.37. **Worse on tips (+0.97) and ears**; joint 5.45 |
| FaceMesh domain | 5–6 mm per landmark **[n=1]** | Geometry-only shaded renders on both sides, or template surface fit (§2b), or drop face PCA (worth ≤ 0.06 mm: all-but-face 2.794) | Field transplant only |
| Tip compression | Strain 0.28–0.58 at sag7/8, cor1/cor4 **[n=1]**; sag8 tissue 6–30 mm across heads | reading = rigid gap + ε_k·h_k (ε 0.1 front, 0.35 back/temporal) | See the compression table below |
| Back screws over thick tissue | sag8 at 58.5 of 60 mm reach; 21–28 mm tissue on Jack | Drop sag7+8, or down-weight them | See the drop table below |
| Ear point | Ball centre 5.4–8.0 mm above the mouth **[n=1]** | On the MRI, the centre of a 13 mm sphere seated on the charm skin at the concha; on the scan, the knob-derived ball centre. Or a +5 mm z prior (SD 3) | 6.7 / 8 mm up: +0.22 / +0.31 **[new]** |
| Device-frame definition | 2.9 mm and 1.95° between definitions **[n=1]** | One canonical frame on both sides | Pose-input bias +0.05–0.13 |
| Frame flex | Arch ±1.5°, A–B joint ±1° / ~3 mm **[n=1]** | Bend segments in simulation; build real tips from fitted segments | Not tested; also a candidate cause of Jack's OOD readings |
| Screw axis tilt/offset | 0.6–2.6°, 0.15–1.6 mm **[n=1]** | Tilt SD 1.5°, offset SD 1 mm | Not tested |
| Scale | MRI ~±1 % **[lit]**; TrueDepth anisotropic | ±1 % MRI scale augmentation | ±3 % face+ears costs +0.28–0.32 |
| Supine/upright face, eye/jaw state | Lateral face −1.3 to −2.1, chin +3.6, eye zone −2.0 **[n=1]**. Training sites' eye/jaw/earplug state unknown | Thickness-proportional deformation; exclude eyes/mouth/jaw | Not tested |
| Skin definition | charm vs threshold skin +0.50; per-site bias never measured | Train on both, or standardise per site | Measured only as a whole-pipeline swap |
| Out-of-distribution input | Jack M² 71 | Gate on the **empirical** training quantile per size (p99 ≈ 35 at size 116; χ² 26 would flag 3.4 % of training rows). First check that error grows with M² | — |

**Compression term** (STUDY/free_model_compression.json; REV/compression_extra.json **[new]**). Values are mean error in mm.

| Test condition | Trained without compression | Trained with ε·h |
|---|---|---|
| Clean | 2.726 | — |
| ε·h compression present | 2.858 (shift 1.17 mm) | 2.657. *Below clean, because ε·h leaks bone depth into the reading. Optimistic* |
| h-independent compression, same median size | 2.918 (shift 1.14) | **2.776** (shift 0.04) |
| Jack along-axis pattern | 3.483 | **3.213** |
| Jack closest-point pattern | 3.194 | 3.026 |
| No compression present | 2.726 | **3.047** |

- The real gain is **removing the ~1.1 mm forward shift**, not improving accuracy.
- On Jack's patterns, the term recovers only 0.17–0.27 of the +0.47–0.76.
- It costs +0.32 if real compression is absent, so measure the size first.
- Random press augmentation did not help (2.784).

**Dropping sag7+8** (STUDY/free_sens_drop7_8.json; REV/extra_drop78.json **[new]**). Values are mean error in mm; each Δ is against the current model's clean 2.733.

| Condition | Current model | Without sag7+8 |
|---|---|---|
| Clean | 2.733 | 2.808 (+0.08) |
| Jack closest-point pattern | 3.200 | 2.851 (**net +0.12**) |
| Jack along-axis pattern | 3.489 | 2.932 (**net +0.20**) |
| Joint stress test | 5.29 | **4.31** |

- This is the best single remedy found. It was chosen and scored on Jack's own pattern, so confirm it on other people.
- It leaves cor1/cor4 (4.3–5.0 mm along-axis) untouched.
- It does **not** fix the OOD score: 50.9 without sag7+8.

### Seating simulation: make simulated readings match real ones

| Step | Today | Change | Evidence |
|---|---|---|---|
| Reference in the scan report | HS/step8_report.py:29-30 uses the obsolete LPA/RPA seat: **14.7 mm RMS** off Jack | Rule A (sag 7.3 5.4 5.4 1.5 13.7 26.9 40.3 46.9; cor 16.7 4.1 4.4 15.1): **6.5 mm RMS**. Even rigid gaps at Jack's **real** pose miss by **5.1 (face-only) / 3.8 (tip-fit) mm RMS**, so the contact model limits the match, not the seat | seat_jack.json; ears_com.json `rms_vs_measured` |
| Ear balls | Points at canal-mouth centroids (MODEL/scripts/ear_canals.py:12) | 13 mm sphere seated on the MRI skin at the concha | ears_com.json |
| Nose contact | First rigid contact on the dorsum (z −11) | At the real pose the nose piece is **4.1–5.2 mm inside the MRI skin at z ≈ −2**, where the nasion tissue is only 5.2 mm (p10–p90 4.8–6.3). Compressing 80–100 % of it is implausible. Candidates: MRI-placement error (~1°, 2–3 mm), CAD/nose-piece offset, some compression. **Measure before modelling** (frame-on vs frame-off scan) | seat_jack.json; quant/thickness.txt |
| Pitch | Rule A prior | Gravity gives **no reliable** pitch preference: the approximate CoM (non-watertight STLs, no knobs or balls) is 0.3 mm forward of and 86.5 mm above the axis. Clamping sets the pitch | ears_com.json |
| Screws | Distance to rigid skin | + compression and a hair offset, sized from measurement (§5) | above |
| Check | — | Real readings from people **without MRI** should fall inside the simulated cloud (empirical M²) | — |

**Why Jack is out of distribution** **[n=1]** (REV/ood_decomp.txt). Rule-A gaps score 20.2 and rigid gaps at his real pose 18.3–22.7; both are in distribution. Adding only the measured front offsets gives 36–46, and adding only the back sag6–8 offsets gives 40. Removing either group from the measured readings still leaves 44–54. The sagittal readings alone score 24.0 (p99 28.2) and the coronal alone 9.0 (p99 15.1). **No single group explains it.** The cause is the combined contact pattern (front stops early, back/temporal goes deep) together with the sag–cor relation. Flex, contact and placement error cannot be separated on n=1. The older pin arc showed the same signature: back gaps 9–15 mm longer, and a 0.95 head scale fit (ME/arc_fit_me.json).

---

## 4. Scan pipeline changes

**Capture protocol**
- **Seating.** Ear balls first, then the nose piece, then the front screws to touch, then the back and temporal screws **last, at light touch**. Better: torque-limiting or spring tips.
- **Contact measurement.** Read each back/temporal screw **at first touch and again clamped**. The difference is the indentation in mm, in frame coordinates, with no alignment needed.
- **Hair.** Part it at each tip and around the ears. No numbers exist yet: the tip-on-hair offset of 0–2 mm is a guess, and TrueDepth above z 45 is hair-contaminated (6.4–11.9 mm).
- **TrueDepth.**
  - **Frame-off** scan for the hidden bony zones; **frame-on** scan (Test 3) for the gauge and scale from the tags.
  - Same session, mouth closed, neutral face.
  - "Eyes closed" matches **Jack's** MRI. The eye/jaw/earplug state of IXI/CC-359 is unknown.
- **Video.** Head or chin rest; slow close passes over sag7/sag8 and both knobs (≥ 60 views per screw, both stickers); diffuse light. Today 108 of 248 views see ≤ 2 tags **[n=1]**.

**Calibration**
- **TrueDepth.** Correct the focal length / intrinsics, or use raw ARKit depth with its own intrinsics. The error is anisotropic (x and z ×0.981, depth correct), and a rigid full-face fit turns that into 2.1 mm at the targets (quant/align_extra.txt d). In the bridge, rescale the Oct-14 scan consistently (HS/mri_via_phone.py:74 fits rigidly; ~1.0–1.4 % unmodelled, ≤ 1 mm **[est]**).
- **Printed frame.** Caliper the tag layout. The −0.4 % figure is inferred, not measured.
- **Screw constants.** Recalibrate b_k and φ_k after any sticker re-glue. Add a table capture with the screws loaded inward to explain the 0.5 mm table-vs-head offset.
- **Joint bundle adjustment** (f +0.5–0.8 %, principal point 10–15 px) is a **candidate**. Validate it by split-half or on the phantom before relying on it.

**Geometry code**
- Build the tips from the fitted per-segment part-B poses and fitted screw axes. Whether S/tips_fast_m.npy does this is **unverified**. It also matters for the OOD question.
- Use one device/transducer-frame definition, shared with the simulation (rows 4 and 7).
- Remove the hard-coded subject constants: `NOSE_DEV` (HS/mri_via_phone.py:23, HS/mri_tip_fit.py:27), `cr` (HS/align_face_scan.py:132) and the device-coordinate masks. They would fail silently on subject 2.
- **Before sonication:** re-read the tags and screws and compare with the scan. Flag slip above a threshold. Frame stability after the scan has not been studied.

**Automatic QC** (add to HS/steps/fast_capture.sh)

| Check | Flag when | Why |
|---|---|---|
| Fine vs coarse readout | difference > 0.8 mm | turn slip = 2.5 mm; sag1 at 0.90–0.94 of 1.25 |
| Screw coverage | < 60 views or one sticker | sag8 seen in only 38 views |
| Gap near reach | > 55 mm | > 60 mm counts as missing; sag8 was 58.5 |
| Arch bend | > table baseline + 1.5° | flex |
| ARKit prior residual | > 3 SD | head motion or frame slip |
| Split-half and tag-only vs joint pose at targets | > ~1 mm (suggested) | gauge error bar for this capture |
| TrueDepth free scale vs calibrated constant | > 0.5 % | scale |
| Out-of-fit forehead band | median > 0.7 mm | pitch of the face fit (residual QC may not track target error) |
| Knob asymmetry | > 4 mm | off-centre seat |
| Readout M² vs training | > empirical p99 per size (~35 at size 116) | out of distribution (Jack 71). Calibrate the false-reject rate on the re-seat cohort first |

---

## 5. Validation without many subjects

| Method | Validates | Needs MRI? | Effort |
|---|---|---|---|
| **Transplant measured patterns** into the 336-head simulator, singly and **jointly** (done here) | Population cost of each mismatch and of their combination | No | minutes per pattern |
| **Canal-anchored placement rerun** (with-face vs no-face) | Whether face alignment matters at deployment at all | No | ~0.5 day |
| **Touch-vs-clamp screw readings**, 5–10 people | Indentation (mm) per screw per person. ε needs h (MRI or ultrasound) or a population-level fit. A swim cap does not work: it measures cap plus hair, TrueDepth misses the occiput, and the supine MRI occiput is already flattened on the coil | No | ½ day |
| **Re-seat study**: 3–4 people × 5 re-seats × 2 helpers | Spread of gauge, seat, flex and readings; real readings vs the simulated cloud. **Invariance test:** targets in a head-fixed frame must not move between re-seats. Back-to-back re-seats may not be independent if indentation persists | No | 1 day of captures |
| **Phantom**: printed skull with a **compliant layer of known thickness** and printed canals from charm labels | Absolute geometry, scale, gauge, readouts, compression model, ball seating, with exact truth. A rigid skin print tests geometry only | Uses Jack's MRI | print + 1 capture |
| **Per-scan consistency**: thin-tissue face vs ball axis vs tip scalp fit | Per-scan error estimate / QC | No | part of QC |
| **Locked test set** (112 heads, never read) | Final check, once, after the protocol is pre-registered | — | 1 run |
| **5–10 people with existing MRIs** | Per-person terms: compression vs pitch, FaceMesh gap, ear seating, supine-upright | Yes | longer term |

I did not check whether any public dataset pairs MRI with 3-D face scans.

**What Jack's head CAN tell us:** biases far above the model error (5 mm FaceMesh gap, 4–10 mm tip offsets, the 14.7 mm old-seat error), patterns to transplant, pipeline geometry (flex, scale, gauge disagreement), and an OOD warning.

**What it CANNOT tell us:**
- **Accuracy.** 40 variants of the *older pin-arc model* (td_arc, 8 gaps) with CV 2.92–3.46 mm gave 1.89–5.49 mm on Jack's phone inputs and 1.93–7.58 mm on his MRI inputs (ME/td_variants*.json).
- **Which variant is better**, now that remedies were picked on his patterns.
- **Pitch vs compression**, since both explain the tip pattern.
- **The truth**, which is provisional; truth definitions disagree by ~1.2 mm.

**Not covered, unknown:**
- Training-MRI acquisition state: eyes, jaw, earplugs, pad compression, occiput on the coil.
- Population coverage: only frame size 116 was analysed; the expected OOD rate for real users is unknown.
- Swelling and diurnal or time effects.
- FaceMesh behaviour across skin tones on photo textures **[unverified]**.
- Dataset licences for a commercial model: CC-359 is CC BY-ND (MODEL/CLAUDE.md:34); IXI is CC BY-SA from general knowledge, not checked here.
- Skull acoustic aberration in the end-to-end budget.

---

## 6. Prioritised plan

The order follows dependencies: the decision experiments come first, retraining waits for measurements.

| # | Action | Why / expected result | Effort | Work |
|---|---|---|---|---|
| 1 | **Decision runs.** (a) Canal-anchored placements, then with-face vs no-face device-frame models. (b) Joint scenario extended with gauge, scale and flex. (c) Per-target p95 and bias vectors in beam coordinates. (d) **Write down an accuracy requirement** (per-target p95, lateral vs axial) | Tells whether MRI-vs-scan face alignment matters at deployment. Today no-face costs +0.53 under unrealistic placements, and every frame variant fails the joint case (5.3–5.5 mm) | ~1 day | Claude runs; Jack sets the requirement |
| 2 | **One canonical definition per quantity**: transducer segment, device frame, ear point (sphere seated on the MRI concha vs ball centre), step8 SIM → rule A, empirical OOD gate | Avoids 1:1 aim bias when frames mix **[est 1–3 mm]**, the +1.03 LPA/RPA trap and the +0.22–0.31 ear offset; reference readouts 14.7 → 6.5 mm RMS | 0.5 day | Claude, Jack reviews |
| 3 | **Measure without an MRI**: touch-vs-clamp readings, re-seat study, compliant phantom | Indentation, placement prior, gauge, flex as distributions instead of n=1; tells whether the OOD readings come from contact or flex | Jack 2 days + Claude 1 day | Jack + Claude |
| 4 | **Retrain with the measured distributions**: compression (h-independent noise), placement prior, flex; decide on sag7/8 (currently the best remedy: joint 5.29 → 4.31) | Removes the ~1.1 mm forward shift; reduces tip sensitivity | 1 day | Claude |
| 5 | **Protocol and calibration**: light-touch back screws (or spring tips), hair parting, frame-off + frame-on TrueDepth, intrinsics-corrected TrueDepth, calipered frame | Less contact variance; removes the 3.2–3.5 % anisotropic scale bias; gauge toward sub-mm **[unverified]** | Jack 1 day + Claude ½ day | Jack + Claude |
| 6 | **Face surface fit (§2b)**, only if face inputs survive step 1 | Replaces the detector term (+0.58–0.80) with a surface-fit term of **unknown size** (zone ambiguity 0.3–2.5 mm on Jack); net effect untested | 1–2 days | Claude |
| 7 | **Automatic QC** (§4 table) | Catches turn slips, OOD readings, head motion and bad scale; the mean is unchanged | 1 day | Claude |
| 8 | **Traceability**: move the study scripts and JSON (now in volatile /private/tmp, partly another session's scratch) to a versioned repo; pin MediaPipe 0.10.21 (MODEL/qc_viewer/charm_chain.py:151-157) and charm; name human owners; trace error source → hazard → control → test (ISO 14971) | Results must be reproducible before any clinical use | 0.5 day + ongoing | Jack owns |

After these: 5–10 people with existing MRIs, to measure the per-person terms that the phantom and re-seats cannot reach.

---

**Path aliases**
- STUDY = /private/tmp/.../scratchpad/study, archived at gdrive:frame_capture/mri_scan_agreement_study/study (quant/ = STUDY/quant)
- REV = STUDY/revise. New for this revision: extra_runs.py, extra_{base,dev,dev_noface,dev_shape,aug,drop78}.json, compression_extra.py/.json, ood_decomp.py/.txt
- MODEL = /Users/jack/aspect/tfus-simulation/.claude/worktrees/ventricle-study-continue-97ed7a
- ME = /Users/jack/aspect/tfus-simulation/results/me
- S = /private/tmp/claude-501/-Users-jack-aspect-AliceVision/71b5fabf-b856-436a-82cc-30d99c2fa7f6/scratchpad
- HS = /Users/jack/aspect/AliceVision/headscan

No repository was modified; the new runs import MODEL code read-only and write only to REV. The old figures reproduced exactly in the new runs: clean 2.733 / 2.706 / 2.732 / 2.808, tip patterns +0.467 / +0.756, and compression cases A–D.