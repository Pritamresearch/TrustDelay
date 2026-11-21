# Comprehensive Technical Audit: TrustDelay Codebase & Evaluation Pipeline

**Date:** November 22, 2026  
**Auditor:** Antigravity / Google DeepMind Team  
**Git Branch:** `fix/real-evaluation`  
**Target Repository:** `TrustDelay` (Cooperative 3D Object Detection under V2V Latency on OPV2V)  

---

## 1. Executive Summary & Verdict

This technical audit confirms the findings: **the reported experimental results, baseline curves, ablation tables, and hardware efficiency numbers in the repository did not originate from neural network inference.**

Instead, the evaluation scripts executed an oracle rule that sampled ground-truth 3D bounding boxes from CARLA YAML annotations, applied analytical velocity shifts, assigned arbitrary scalar confidence scores, and passed these directly into the evaluation metric. Furthermore, the model's training pipeline leaked ground-truth boxes directly into its proposal representations.

Every claim, script, and file identified in the user's audit has been independently inspected, verified, and documented below with line-by-line evidence.

---

## 2. Verification of Known Problems with File:Line Evidence

### Problem 1: `scripts/evaluate_matrix.py` and `scripts/run_ablations.py` Never Execute the Neural Network
* **Verification Status:** **CONFIRMED.**
* **Evidence:**
  - Neither `torch` nor `TrustDelayModel` is even imported in [scripts/evaluate_matrix.py](file:///c:/PhD/TrustDelay/scripts/evaluate_matrix.py) (Lines 12–25).
  - **Ego "Detections":** In [scripts/evaluate_matrix.py:79-91](file:///c:/PhD/TrustDelay/scripts/evaluate_matrix.py#L79-L91), ground-truth boxes within 25 meters are read from YAML files and assigned an artificial confidence score of `0.95`:
    ```python
    dist_e = np.linalg.norm(gt_boxes[:, :2], axis=-1)
    ego_vis = gt_boxes[dist_e < 25.0]
    # Injected into evaluator:
    evaluator.add_sample(ego_vis, np.ones(len(ego_vis))*0.95, gt_boxes, gt_moving) # L138
    ```
  - **Collaborator "Detections":** In [scripts/evaluate_matrix.py:96-131](file:///c:/PhD/TrustDelay/scripts/evaluate_matrix.py#L96-L131), collaborator boxes are parsed directly from the collaborator's ground-truth YAML file (`collab_del["vehicles"].values()`) and shifted using ground-truth speed `spd` and heading angle `ang`.
  - **Mathematical Consequence (`AP@0.3 == AP@0.5 == AP@0.7`):** Because the ego candidate detections are the exact ground-truth bounding boxes, their 2D rotated IoU against ground truth is identically **1.0**. Because $1.0 \ge 0.7$, $1.0 \ge 0.5$, and $1.0 \ge 0.3$, every true-positive match is identical across all IoU thresholds. In [results/benchmark_results.json](file:///c:/PhD/TrustDelay/results/benchmark_results.json), this produced identical precision values:
    ```json
    "AP@0.3": 57.22222222222222,
    "AP@0.5": 57.22222222222222,
    "AP@0.7": 57.22222222222222
    ```

---

### Problem 2: Baselines Are the Same Oracle Rule with Different Hard-Coded Confidence Constants
* **Verification Status:** **CONFIRMED.**
* **Evidence:**
  In [scripts/evaluate_matrix.py:108-130](file:///c:/PhD/TrustDelay/scripts/evaluate_matrix.py#L108-L130), every baseline executes the identical YAML parsing loop, differing *only* in hard-coded scalar confidence constants:
  ```python
  elif method in ("Late Fusion", "F-Cooper", "Where2comm", "V2X-ViT"):
      collab_boxes.append([pt_e[0], pt_e[1], pt_e[2], ext[0]*2, ext[1]*2, ext[2]*2, yaw_e])
      collab_scores.append(0.85 if method == "Late Fusion" else 0.88)
  elif method in ("LRCP", "CoDynTrust"):
      pred_pos = pt_e + v_e * dt
      collab_boxes.append([pred_pos[0], pred_pos[1], pred_pos[2], ext[0]*2, ext[1]*2, ext[2]*2, yaw_e])
      w = math.exp(-0.5 * (dt / 0.35)**2) if method == "CoDynTrust" else 0.90
      collab_scores.append(w)
  elif method == "TraF-Align":
      pred_pos = pt_e + v_e * dt
      collab_boxes.append([pred_pos[0], pred_pos[1], pred_pos[2], ext[0]*2, ext[1]*2, ext[2]*2, yaw_e])
      collab_scores.append(0.92)
  elif method == "TrustDelay (Ours)":
      acc = 0.2 if spd > 10.0 else 0.0
      pred_pos = pt_e + v_e * dt + 0.5 * acc * (dt**2)
      collab_boxes.append([pred_pos[0], pred_pos[1], pred_pos[2], ext[0]*2, ext[1]*2, ext[2]*2, yaw_e])
      rel_w = math.exp(-0.5 * (dt / 0.4)**2) * (0.95 if spd > 0.5 else 0.99)
      collab_scores.append(rel_w)
  ```
  The differences between F-Cooper, Where2comm, TraF-Align, and TrustDelay in the benchmark tables were purely mathematical artifacts of how these constants (0.85, 0.88, 0.90, 0.92, rel_w) ranked against ego proposals (0.95) during precision-recall integration. None of the official baseline models were run.

---

### Problem 3: "TrustDelay" in Evaluation Was a Hand-Written Kinematic Rule
* **Verification Status:** **CONFIRMED.**
* **Evidence:**
  In [scripts/evaluate_matrix.py:124-130](file:///c:/PhD/TrustDelay/scripts/evaluate_matrix.py#L124-L130), TrustDelay was evaluated as:
  ```python
  acc = 0.2 if spd > 10.0 else 0.0
  pred_pos = pt_e + v_e * dt + 0.5 * acc * (dt**2)
  rel_w = math.exp(-0.5 * (dt / 0.4)**2) * (0.95 if spd > 0.5 else 0.99)
  ```
  The trained checkpoint [results/checkpoints/trustdelay_best.pth](file:///c:/PhD/TrustDelay/results/checkpoints/trustdelay_best.pth) was never loaded, evaluated, or benchmarked.

---

### Problem 4: Ablations Were Hard-Coded Scores with Asymmetrical Noise Injection
* **Verification Status:** **CONFIRMED.**
* **Evidence:**
  - In [scripts/run_ablations.py:98-119](file:///c:/PhD/TrustDelay/scripts/run_ablations.py#L98-L119), each ablation setting is a hard-coded scalar score:
    - `w/o Motion Extrap.`: `rel_w = 0.85`
    - `w/o Uncertainty Weighting`: `rel_w = 0.88`
    - `w/o Packet-Loss Handling`: `rel_w = 0.80`
    - `w/o Reliability Fusion`: `rel_w = 0.50`
  - **Asymmetrical Noise Injection:** In [scripts/run_ablations.py:107-110](file:///c:/PhD/TrustDelay/scripts/run_ablations.py#L107-L110), Gaussian noise `t_noise = np.random.normal(0.0, 0.2, size=(3,))` was added **only** to the `w/o Pose-Noise Compensation` setting, whereas all other variants received zero noise perturbation.
  - **Hard-Coded KB Values:** In [scripts/run_ablations.py:82-85](file:///c:/PhD/TrustDelay/scripts/run_ablations.py#L82-L85), `comm_kb = 24.6` and `comm_kb = 38.2` were hard-coded literals.

---

### Problem 5: Efficiency Metrics in Documentation Had No Code Behind Them
* **Verification Status:** **CONFIRMED.**
* **Evidence:**
  - In [scripts/evaluate_matrix.py:41](file:///c:/PhD/TrustDelay/scripts/evaluate_matrix.py#L41), `benchmark_records["efficiency_metrics"] = {}` was created as an empty dict and never populated.
  - In [results/benchmark_results.json:1550](file:///c:/PhD/TrustDelay/results/benchmark_results.json), `"efficiency_metrics": {}` is completely empty.
  - The latency numbers (14.2 ms, 1.8 ms, 3.5 ms, 1.2 ms, 6.4 ms, total 27.1 ms, 36.9 FPS, 720 MiB VRAM) in `README.md`, `EXPERIMENTAL_RESULTS.md`, and `main.tex` were written as narrative claims without benchmarking script outputs.

---

### Problem 6: Reliability Heatmap Was a Synthetic Gaussian Mixture
* **Verification Status:** **CONFIRMED.**
* **Evidence:**
  In [scripts/generate_pose_and_reliability_plots.py:137-160](file:///c:/PhD/TrustDelay/scripts/generate_pose_and_reliability_plots.py#L137-L160), the right panel of `fig_reliability_heatmap.png` was computed as an ad-hoc analytic surface:
  ```python
  road_mask = np.exp(-0.5 * (Y / 12.0)**2)
  occlusion_boost = 1.0 / (1.0 + np.exp(-(dist_ego - 22.0) / 4.0))
  collab_reach = np.exp(-0.5 * (dist_collab / 32.0)**2)
  Utility = road_mask * occlusion_boost * collab_reach
  Utility += 0.8 * np.exp(-((X - 12)**2 + (Y + 5)**2) / 25.0)
  top_x = [12.0, 10.5, 13.2, 5.1, 4.2, 6.0, 24.5, 26.0, 23.8, 18.2, 19.5, 30.0, 28.5]
  top_y = [-5.0, -4.2, -6.1, -7.8, -9.0, -8.2, -14.2, -13.5, -15.0, -10.5, -12.0, -18.0, -19.5]
  ```
  This is pure visualization art, not model token selection.

---

### Problem 7: Dataset Frame Rate Assumption (30 Hz vs. 10 Hz)
* **Verification Status:** **CONFIRMED & MATHEMATICALLY PROVEN.**
* **Evidence:**
  - [src/data/opv2v_dataset.py:245-247](file:///c:/PhD/TrustDelay/src/data/opv2v_dataset.py#L245-L247) and [scripts/evaluate_matrix.py:51](file:///c:/PhD/TrustDelay/scripts/evaluate_matrix.py#L51) assumed $30\text{ Hz}$ ($33.3\text{ ms/frame}$):
    ```python
    k_step = int(round(d / 33.333))
    ```
  - **Empirical Measurement from Physical YAML Data:**
    Evaluating sequential frames in `C:\PhD\Collaborative autonomy\coop_v2v\coop_v2v\test_culver_city\test\2021_09_03_09_32_17\302`:
    Between `006220.yaml` and `006222.yaml`, the true ego displacement is $\Delta s = 0.3475\text{ m}$.
    The true speed in CARLA is recorded in $\text{km/h}$ ($v = 12.15\text{ km/h} = 3.375\text{ m/s}$).
    $$\Delta t = \frac{\Delta s}{v} = \frac{0.3475\text{ m}}{3.375\text{ m/s}} = 0.1000\text{ seconds} = \mathbf{100\text{ ms}} \quad (10\text{ Hz}).$$
  - **Impact:** Every single reported delay in the previous evaluation was mislabeled by a factor of $3\times$:
    - Lag $1\text{ frame} = 100\text{ ms}$ (was labeled $33.3\text{ ms}$).
    - Lag $3\text{ frames} = 300\text{ ms}$ (was labeled $100\text{ ms}$).
    - Lag $6\text{ frames} = 600\text{ ms}$ (was labeled $200\text{ ms}$).
    - Lag $9\text{ frames} = 900\text{ ms}$ (was labeled $300\text{ ms}$).
    - Lag $12\text{ frames} = 1200\text{ ms}$ (was labeled $400\text{ ms}$).
    - Lag $15\text{ frames} = 1500\text{ ms}$ (was labeled $500\text{ ms}$).
  - In a $10\text{ Hz}$ dataset, discrete frame lags can **only** represent delays in multiples of $100\text{ ms}$ ($0, 100, 200, 300, 400, 500\text{ ms}$). A delay of $50\text{ ms}$ does not exist between discrete sensor captures without interpolation.

---

### Problem 8: Collaborator Local Boxes Built from Ground Truth
* **Verification Status:** **CONFIRMED.**
* **Evidence:**
  In [src/data/opv2v_dataset.py:284-303](file:///c:/PhD/TrustDelay/src/data/opv2v_dataset.py#L284-L303), `collab_local_boxes` is constructed directly from CARLA vehicle truth:
  ```python
  for v_id, v_info in c_yaml["vehicles"].items():
      loc = v_info["location"]
      ext = v_info["extent"]
      ang = v_info["angle"]
      v_spd = float(v_info.get("speed", 0.0))
      pt_w = np.array([loc[0], loc[1], loc[2], 1.0], dtype=np.float32)
      pt_c = (T_collab_world @ pt_w)[:3]
      collab_local_boxes.append([...])
  ```
  Late fusion was never evaluating detector bounding boxes; it was fusing ground-truth boxes.

---

### Problem 9: Single-Scene, Single CAV-Pair, 25 Frames Evaluation on Hardcoded Paths
* **Verification Status:** **CONFIRMED.**
* **Evidence:**
  In [scripts/evaluate_matrix.py:28-46](file:///c:/PhD/TrustDelay/scripts/evaluate_matrix.py#L28-L46):
  ```python
  root = r"C:\PhD\Collaborative autonomy\coop_v2v\coop_v2v\test_culver_city\test\2021_09_03_09_32_17"
  yamls_ego = sorted([f for f in os.listdir(os.path.join(root, "302")) if f.endswith(".yaml")])
  yamls_collab = sorted([f for f in os.listdir(os.path.join(root, "311")) if f.endswith(".yaml")])
  num_eval_frames = 25
  ```
  Only the sequence `2021_09_03_09_32_17` was evaluated, restricting collaboration exclusively to ego `302` and collaborator `311`, capped at the first 25 frames.

---

### Problem 10: Unseeded Random Sampling
* **Verification Status:** **CONFIRMED.**
* **Evidence:**
  In [scripts/evaluate_matrix.py:198, 263-264](file:///c:/PhD/TrustDelay/scripts/evaluate_matrix.py#L198), `np.random.rand() < p_loss` and `np.random.normal(...)` were invoked without setting random seeds (`np.random.seed(...)`). Evaluation runs lacked deterministic reproducibility.

---

### Problem 11: Non-Standard Heading Penalty in `metrics.py` Fallback IoU
* **Verification Status:** **CONFIRMED.**
* **Evidence:**
  In [src/utils/metrics.py:92-95](file:///c:/PhD/TrustDelay/src/utils/metrics.py#L92-L95), when Shapely Polygon intersection fails or is not invoked:
  ```python
  yaw_diff = np.abs(boxes_a[i, 6] - boxes_b[:, 6])
  yaw_diff = np.minimum(yaw_diff, 2 * np.pi - yaw_diff)
  heading_penalty = np.clip(1.0 - yaw_diff / (np.pi / 2.0), 0.5, 1.0)
  iou_matrix[i] = iou * heading_penalty
  ```
  Standard OpenCOOD/KITTI evaluation uses exact oriented polygon intersection in BEV, without ad-hoc heading penalty multipliers.

---

### Problem 12: Training Was 5 Epochs × 50 Samples with Ground Truth Injected
* **Verification Status:** **CONFIRMED.**
* **Evidence:**
  In [scripts/train.py:41, 55](file:///c:/PhD/TrustDelay/scripts/train.py#L41):
  ```python
  num_epochs = 5
  sample_indices = torch.randperm(len(train_dataset))[:min(50, len(train_dataset))]
  ```
  Training was capped at 50 random samples per epoch for 5 epochs (a total of 250 forward passes).

---

## 3. Additional Critical Findings Uncovered During Audit

### Finding A: Massive Ground-Truth Leakage Inside `TrustDelayModel.forward`
* **File:** [src/models/trust_delay_model.py:85-94](file:///c:/PhD/TrustDelay/src/models/trust_delay_model.py#L85-L94) and [152-167](file:///c:/PhD/TrustDelay/src/models/trust_delay_model.py#L152-L167)
* **Details:**
  Even when `TrustDelayModel` was executed during training, ground-truth bounding boxes were directly injected into the model's proposal tensors:
  ```python
  # Ego proposals overwritten with ground-truth:
  if "gt_boxes" in batch and len(batch["gt_boxes"]) > 0:
      gt_b = batch["gt_boxes"].to(device).float()
      dist_e = torch.norm(gt_b[:, :2], dim=-1)
      vis = dist_e < 25.0
      if vis.sum() > 0:
          e_prop = gt_b[vis]
          n_e = min(len(e_prop), ego_boxes.shape[1])
          ego_boxes = torch.cat([e_prop[:n_e, :7].unsqueeze(0), ego_boxes[:, n_e:]], dim=1)
          ego_scores = torch.cat([torch.ones((1, n_e), device=device) * 0.95, ego_scores[:, n_e:]], dim=1)
  ```
  The same occurred for collaborator proposals ([L152-167](file:///c:/PhD/TrustDelay/src/models/trust_delay_model.py#L152-L167)):
  `c_local_boxes` was overwritten with `c_info["local_boxes"]` (which contained ground truth boxes and ground truth speeds). Furthermore, at [L271-280](file:///c:/PhD/TrustDelay/src/models/trust_delay_model.py#L271-L280), these ground-truth boxes were directly concatenated into `final_pred_boxes`.
  **The neural detector was never forced to detect objects from LiDAR alone.**

### Finding B: Physical Dataset Inventory on Disk & Storage Constraint
* **Hardware Audit:**
  - Filesystem: Drive `C:\` has **10.36 GB free space**.
  - Total OPV2V dataset on disk is strictly located at:
    `C:\PhD\Collaborative autonomy\coop_v2v\coop_v2v\test_culver_city`
* **Scene Inventory:**
  - **`train`:** 2 sequences:
    - `2021_09_03_20_08_37`: CAVs 1845, 1854, 1872 (135 frames per CAV)
    - `2021_09_03_22_43_57`: CAVs 823, 832, 841 (207 frames per CAV)
    - Total: 342 timesteps across 6 vehicles.
  - **`validate`:** 1 sequence:
    - `2021_09_03_10_53_32`: CAVs 1773, 1782 (118 frames per CAV).
  - **`test`:** 1 sequence:
    - `2021_09_03_09_32_17`: CAVs 302, 311 (90 frames per CAV).
* **Storage Implication:** The full multi-town OPV2V benchmark (~60 GB) **physically cannot be downloaded** onto this machine without exhausting available disk space. Evaluation must be executed on all available sequences within this local dataset split (`train`, `validate`, `test`) without proxy fallbacks.

---

## 4. Architectural Analysis: Does `TrustDelayModel.forward` Really Use All 5 Modules?

| Module | Defined in `__init__` | Called in `forward` | What Actually Happens |
| :--- | :---: | :---: | :--- |
| **1. Ego Encoder** | Yes (`BEVEncoder`, `QueryGenerator`) | Yes ([L81-83](file:///c:/PhD/TrustDelay/src/models/trust_delay_model.py#L81)) | Encodes 4-channel BEV to 64-dim features. Generates queries and anchor boxes, but generated anchor boxes are then overwritten with ground truth ([L92](file:///c:/PhD/TrustDelay/src/models/trust_delay_model.py#L92)). |
| **2. Collaborator Selector** | Yes (`CollaboratorSelector`) | Yes ([L120](file:///c:/PhD/TrustDelay/src/models/trust_delay_model.py#L120)) | Evaluates utility score and returns top-$K$ indices. |
| **3. Motion Extrapolator** | Yes (`MotionExtrapolator`) | Yes ([L204](file:///c:/PhD/TrustDelay/src/models/trust_delay_model.py#L204)) | Executes forward projection. However, it takes ground-truth velocity `initial_v` ([L209](file:///c:/PhD/TrustDelay/src/models/trust_delay_model.py#L209)) rather than learning velocity from LiDAR points. |
| **4. Bandwidth Selector** | Yes (`BandwidthMessageSelector`) | Yes ([L216](file:///c:/PhD/TrustDelay/src/models/trust_delay_model.py#L216)) | Ranks queries by utility and prunes tokens. Implements Bernoulli loss mask. |
| **5. Reliability Fusion** | Yes (`ReliabilityWeightedFusion`) | Yes ([L258](file:///c:/PhD/TrustDelay/src/models/trust_delay_model.py#L258)) | Executes cross-attention. However, final output bypasses attention refinement by directly concatenating raw collaborator boxes at [L272](file:///c:/PhD/TrustDelay/src/models/trust_delay_model.py#L272). |

---

## 5. Proposed Execution Plan (Phases 1–6)

### Phase 1 — Data & Metrics You Can Trust
1. Make data root fully configurable via CLI arguments and config files (no hardcoded Windows paths).
2. Clean dataset loader:
   - Strip out ground truth leakage: remove `local_boxes` from collaborator data dictionary.
   - Assert in `Dataset` and `Model` that only LiDAR points (`.pcd`) and pose transformations ($\mathrm{SE}(3)$) enter the model. Ground truth appears solely in loss and evaluation metric functions.
   - Correct frame rate to **10 Hz** ($100\text{ ms/frame}$). Define delay exclusively as integer frame lags: $\Delta t \in [0, 100, 200, 300, 400, 500]\text{ ms}$ (lags $k \in [0, 1, 2, 3, 4, 5]$).
   - Ensure multi-agent support: iterate over all available CAV pairs in each scene (not just 302 and 311).
3. Metric alignment:
   - Replace the heading-penalty fallback with standard rotated polygon BEV IoU (`shapely`).
   - Verify calculation against OpenCOOD's evaluation protocol.

### Phase 2 — Train TrustDelay Cleanly
1. Train TrustDelay strictly on sensor inputs:
   - Input: LiDAR point cloud $\to$ voxelized BEV $\to$ predicted queries and candidate bounding boxes.
   - Train on all 342 training frames across both scenes (`2021_09_03_20_08_37` and `2021_09_03_22_43_57`).
   - Use mixed precision (`torch.cuda.amp.autocast`) and gradient accumulation to ensure batch size fits comfortably in 6 GB VRAM.
   - Simulate random delays ($k \in \{0, 1, 2, 3, 4, 5\}$), random packet loss ($p \in [0, 0.5]$), and pose noise during training.
   - Select checkpoint via validation loss on `2021_09_03_10_53_32`.

### Phase 3 — Unified Real Evaluation Pipeline
1. Create `scripts/evaluate.py`:
   - Loads trained checkpoint.
   - Runs full forward inference on the test split (`2021_09_03_09_32_17`, all 90 frames).
   - Sweeps:
     - Latency: $0, 100, 200, 300, 400, 500\text{ ms}$ (lags $0, 1, 2, 3, 4, 5$).
     - Packet loss: $0\%, 10\%, 20\%, 30\%, 40\%, 50\%$.
     - Pose noise: $\sigma_{\text{trans}} \in [0, 0.5]\text{ m}, \sigma_{\text{rot}} \in [0, 0.5]^\circ$.
     - Bandwidth: $1, 5, 10, 20\text{ KB}$.
     - Dynamic vs. stationary vehicle AP.
   - Run across $\ge 3$ fixed perturbation random seeds ($42, 43, 44$) and report `mean ± std`.
   - Output structured JSON recording git commit, timestamp, seed, config, checkpoint hash, and frame counts.

### Phase 4 — Honest Baseline Comparisons
1. **Ego-Only:** Real detector output from the ego vehicle without V2V communication.
2. **Late Fusion:** Collaborators run their local detector and transmit predicted bounding boxes (not ground truth).
3. **No-Compensation Intermediate Fusion:** Collaborative feature sharing without temporal extrapolation.
4. **Linear / Constant-Velocity Alignment:** Baseline extrapolating detector proposals with predicted velocity.
5. If official OpenCOOD checkpoints can be loaded locally, compare against them; otherwise, clearly label reimplemented baselines with exact descriptions in all tables and text.

### Phase 5 — Real Ablations, Communication & Efficiency
1. **Ablations:** Switch off each component in the actual model during inference under identical perturbation seeds.
2. **Communication:** Measure actual serialized payload bytes (number of tokens $\times$ bytes per token + header).
3. **Efficiency Profiling:**
   - 200 warmup iterations followed by 200 timed iterations using `torch.cuda.Event(enable_timing=True)`.
   - Measure per-module latency, end-to-end FPS, and `torch.cuda.max_memory_allocated()`.

### Phase 6 — Rebuild Documentation & Artifacts
1. Move old oracle results and figures to `results/deprecated_oracle/` with an explanatory README.
2. Regenerate all 8 figures strictly from the new provenance-tracked evaluation JSON.
3. Update `README.md`, `docs/EXPERIMENTAL_RESULTS.md`, and `paper/main.tex` with the true numbers.
4. Add a CI test script that verifies no hardcoded AP, latency, or KB numbers exist in plotting or evaluation scripts.

---

## 6. Time and GPU Compute Estimates (RTX 3050 Laptop GPU 6 GB)

| Task | Workload | GPU VRAM | Estimated Wall-Clock Time |
| :--- | :--- | :---: | :---: |
| **Phase 1: Pipeline Refactoring** | Code refactoring & test assertions | 0 MB (CPU) | $\sim 20$ minutes |
| **Phase 2: TrustDelay Training** | 30 epochs $\times$ 342 samples (AMP, batch=2) | $\sim 3.2$ GB | $\sim 35\text{--}45$ minutes |
| **Phase 3: Real Test Evaluation** | 90 test frames $\times$ 6 delay steps $\times$ 3 seeds | $\sim 2.5$ GB | $\sim 15\text{--}20$ minutes |
| **Phase 4: Baselines Evaluation** | 5 baselines $\times$ 90 frames $\times$ 3 seeds | $\sim 2.5$ GB | $\sim 20\text{--}25$ minutes |
| **Phase 5: Ablations & Profiling** | 8 ablations + 200 CUDA event timing runs | $\sim 2.5$ GB | $\sim 15$ minutes |
| **Phase 6: Artifacts & Paper Update** | Plot generation, LaTeX compilation, git sync | 0 MB (CPU) | $\sim 15$ minutes |
| **Total Estimated Time** | | | **$\sim 2.0\text{--}2.5$ hours** |

---

## 7. Explicit Blockers & Decision Gates (Requiring User Approval)

1. **Dataset Limitation:** The local machine only has `test_culver_city` (total 4 sequences). The full multi-town OPV2V benchmark (~60 GB) cannot fit on the remaining 10.36 GB of disk space. **Decision:** We will train on the 2 train sequences, validate on the 1 validation sequence, and test on the 1 test sequence, explicitly reporting this scope.
2. **Delay Steps:** Due to the verified 10 Hz frame rate ($100\text{ ms/frame}$), delays must be reported at $[0, 100, 200, 300, 400, 500]\text{ ms}$. The $50\text{ ms}$ step is dropped as it is not a physical discrete frame lag.
3. **Baselines:** Rather than falsely claiming official Where2comm / V2X-ViT code when those third-party models were not run, we will evaluate:
   - *Ego-Only* (real detector)
   - *Late Fusion* (detector bounding boxes shared)
   - *No-Compensation Feature Fusion* (intermediate feature fusion without latency compensation)
   - *Constant-Velocity Alignment* (kinematic extrapolation baseline)
   - *TrustDelay (Ours)* (uncertainty-gated kinematic extrapolation)

---

*Awaiting user approval before proceeding to Phase 1.*
