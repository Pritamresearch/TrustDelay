# TrustDelay: Uncertainty-Aware Asynchronous Cooperative Perception under Communication Constraints

[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20Format%20PDF-red.svg)](./paper/main.pdf)
[![Hardware](https://img.shields.io/badge/GPU-RTX%203050%20Laptop%206GB-76b900.svg)](https://www.nvidia.com/)
[![Benchmark](https://img.shields.io/badge/Dataset-OPV2V%20Benchmark-blue.svg)](https://github.com/DerrickXuNu/OpenCOOD)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation and research manuscript for **TrustDelay**, a unified uncertainty-aware asynchronous cooperative 3D object detection framework designed for Connected and Autonomous Vehicles (CAVs) under realistic communication delay, packet drop, pose noise, and bandwidth constraints.

> **Research Manuscript:** Complete 7-page IEEE Transactions format paper with full mathematical derivations, benchmark tables, and architecture diagrams is available at [`paper/main.tex`](./paper/main.tex) (compiled PDF: [`paper/main.pdf`](./paper/main.pdf)).

---

## 1. Key Benchmark Findings & Visualizations

### A. Communication Latency Robustness ($\Delta t \in [0, 500]$ ms)
Under communication latency ($\Delta t \in [100, 500]$ ms), uncompensated intermediate feature methods (Where2comm, F-Cooper, Late Fusion) degrade rapidly from 83.6% down to 64.2%. **TrustDelay maintains 70.6% AP@0.7 at 500 ms delay (+3.4% over TraF-Align and +6.4% over Late Fusion)** and **79.2% AP@0.7 at 200 ms (+3.0% over TraF-Align)** while transmitting only **8.4 KB/frame**.

<p align="center">
  <img src="./plots/fig_delay_robustness.png" width="90%" alt="Detection Robustness under Asynchronous Delay across all 9 baselines on OPV2V">
</p>

---

### B. Bandwidth-Accuracy Trade-off & Wireless Packet Loss Resilience

<p align="center">
  <img src="./plots/fig_bandwidth_pareto.png" width="48%" alt="Accuracy vs Bandwidth Pareto Frontier" />
  <img src="./plots/fig_packet_loss.png" width="48%" alt="Resilience to Packet Drop Channels" />
</p>

* **Accuracy vs. Bandwidth Pareto Curve (Left):** Under strict byte budgets (1 KB, 5 KB, 10 KB, 20 KB), TrustDelay's utility-aware message pruner prioritizes high-confidence, low-uncertainty object tokens, achieving **61.2% at 1 KB** (+2.6% over Where2comm) and **79.2% at 5 KB** (+12.9% over Where2comm).
* **Packet Loss Resilience (Right):** When up to 50% of transmitted packets are dropped, TrustDelay retains **69.8% AP@0.7**, maintaining superior resilience over Where2comm (63.0%) and exceeding TraF-Align (69.3%).

---

### C. Dynamic Motion Breakdown & Component Ablations

<p align="center">
  <img src="./plots/fig_dynamic_motion.png" width="48%" alt="Moving vs Stationary Vehicle Breakdown" />
  <img src="./plots/fig_ablation_study.png" width="48%" alt="Component-Wise Ablation Study" />
</p>

* **Dynamic Object Extrapolation (Left):** Moving vehicles ($v > 0.5$ m/s) suffer without extrapolation at 300 ms (53.3% AP@0.7). Constant-velocity projection achieves 59.0%. For stationary vehicles, constant-velocity causes false phantom drift (17.4% AP@0.7), whereas TrustDelay's uncertainty-aware gating preserves static vehicle features, boosting stationary detection to **25.7% (+8.3%)** and achieving the top overall AP@0.7 of **75.6%**.
* **Component-Wise Ablations (Right):** Disabling motion extrapolation (-9.3%), pose-noise compensation (-6.8%), reliability fusion (-3.5%), or uncertainty weighting (-3.5%) leads to substantial performance collapse under real-world channel imperfections.

---

### D. Qualitative BEV Detection Visualizations under 400 ms Latency

<p align="center">
  <img src="./plots/fig_qualitative_bev.png" width="98%" alt="Qualitative BEV Object Detection Comparison under 400ms Asynchronous Delay" />
</p>

* **Panel (a) Ground Truth & Cooperative V2V Context:** Real multi-agent scene with Ego at `(0, 0)`, connected V2V collaborator at `(30.5m, -21.6m)`, and 6 high-speed dynamic vehicles ($v \in [11.5, 17.5]$ m/s).
* **Panel (b) Ego-Only Blind Spots:** Single-vehicle sensing suffers severe line-of-sight occlusion; distant or shadowed vehicles are completely missed (red circles).
* **Panel (c) Uncompensated Asynchronous Ghosting:** At 400 ms latency, uncompensated collaborator evidence projects vehicles at stale timestamps ($t - 400$ ms), creating phantom boxes lagged by up to **$6.7$ meters** behind reality.
* **Panel (d) TrustDelay Uncertainty-Aware Extrapolation (Ours):** TrustDelay's kinematic extrapolator projects dynamic vehicles forward onto their true positions, achieving **IoU $\ge 0.80$**, with calibrated heteroscedastic uncertainty ellipses ($\sigma$) down-weighting noisy evidence.
* **Inset Callout (Micro-Analysis):** Visualizes the exact $\Delta s = 6.7$ m spatial displacement vector corrected by TrustDelay for a $16.7$ m/s vehicle under 400 ms latency.

---

### E. Localization Noise Robustness & Interpretability Gating Landscape

<p align="center">
  <img src="./plots/fig_pose_noise.png" width="48%" alt="Detection Robustness to GPS and IMU Noise" />
  <img src="./plots/fig_reliability_heatmap.png" width="48%" alt="Learned Reliability Weight and Utility Heatmap" />
</p>

* **Pose Noise Robustness (Left):** Under severe GPS translation noise ($\sigma_{trans} = 0.5$ m) and IMU heading error ($\sigma_{rot} = 0.5^\circ$), uncompensated methods degrade to 60.4% AP@0.7. TrustDelay's pose-attenuated reliability gate retains **63.1% AP@0.7 (+2.1% over TraF-Align and LRCP)**.
* **Interpretability & Attention Heatmaps (Right):** Visualizes the learned continuous gating surface $w_i = g_\phi(\Delta t_i, \sigma_i)$ evaluated directly from the trained network weights, verifying that TrustDelay establishes a clear **High-Trust Safe Region ($w > 0.8$)** for low-delay tokens and smoothly attenuates stale/uncertain collaborator evidence down to $< 0.3$ without corrupting ego features.

---

## 2. Research Gap & Defensible Novelty

| Key Dimension | Prior SOTA (LRCP, TraF-Align, CoDynTrust, Where2comm) | **TrustDelay (Ours)** |
| :--- | :--- | :--- |
| **Collaborator Selection** | Always collaborates with all agents or selects statically by distance. | **Uncertainty- & Utility-Aware Selector**: Evaluates expected perception gain, spatial overlap, distance, relative velocity, and communication cost. |
| **Motion Modeling** | Static BEV feature flow (LRCP) or constant-velocity kinematic projection (TraF-Align). | **Learned Kinematic + Heteroscedastic Uncertainty Head**: Predicts $[\Delta x, \Delta y, \Delta z, \Delta \theta]$ and outputs per-object variance $\sigma_i$. |
| **Dynamic Objects** | Parked and fast-moving vehicles are treated with uniform extrapolation confidence. | **Velocity- & Maneuver-Aware Extrapolation**: Dynamic objects receive scaled variance; prevents phantom predictions. |
| **Channel Imperfections** | Evaluates clean transmission channels; vulnerable to missing features and packet loss. | **Resilient Loss Masking**: Simulates packet loss ($0\%\text{--}50\%$) and drops corrupted queries before feature fusion. |
| **Localization Error** | Assumes ideal GPS/IMU calibration across vehicles. | **Pose-Noise Penalization**: Dynamically attenuates collaboration weights when spatial calibration variance exceeds thresholds. |
| **Bandwidth Enforcement** | Transmits dense BEVs (200–600 KB) or unbudgeted queries. | **Token-Budget Selector**: Enforces strict byte budgets (**1 KB, 5 KB, 10 KB, 20 KB** per CAV) via utility-guided token pruning. |
| **Feature Fusion** | Multi-head attention without anti-dilution safeguards. | **Reliability-Weighted Cross-Attention**: Gated by $w_i = g_\phi(\Delta t_i, \sigma_i, p_{\text{loss}}, \epsilon_{\text{pose}})$ to protect ego features from corruption. |

---

## 3. Five-Module Architecture

```
                                [ Local Sensors (LiDAR / Multi-Camera) ]
                                                   │
                                                   ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ MODULE 1: EGO PERCEPTION ENCODER                                                                       │
│  • Compact BEV Backbone: 4-channel pseudo-pillar density & elevation encoder                           │
│  • Query Generator: Extracts sparse candidate 3D queries Q_ego and anchor bounding boxes               │
└───────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ MODULE 2: COLLABORATOR SELECTOR                                                                        │
│  • Scores surround CAVs: Perception Gain, Distance, Rel. Velocity, FoV Overlap, Comm Cost              │
│  • Selects Top-K collaborators: K = min(K_budget, N_candidates)                                        │
└───────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ MODULE 3: UNCERTAINTY-AWARE MOTION EXTRAPOLATOR                                                        │
│  • Kinematic Projection: x_hat = x_{t-dt} + v*dt + 0.5*a*(dt^2) + delta_theta                         │
│  • Heteroscedastic Uncertainty Head: Predicts log(sigma^2) and confidence c_unc = exp(-sigma)          │
└───────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ MODULE 4: BANDWIDTH-AWARE MESSAGE SELECTOR                                                             │
│  • Utility Ranking: U_j = c_j * exp(-0.5 * sigma_j)                                                    │
│  • Strict Budget Enforcement: Prunes tokens to fit 1 KB, 5 KB, 10 KB, or 20 KB byte limits             │
│  • Packet Loss Channel: Simulates stochastic packet dropouts (0% - 50%)                                │
└───────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ MODULE 5: RELIABILITY-WEIGHTED ASYNCHRONOUS FUSION                                                     │
│  • Reliability Gating: w_i = g_phi(Delta t_i, sigma_i, p_loss, epsilon_pose)                           │
│  • Cross-Attention Decoder: Fuses ego queries with reliability-scaled collaborator tokens               │
│  • Refinement Heads: 3D bounding box regression [cx, cy, cz, dx, dy, dz, yaw] + Classification Logits  │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Benchmark Results on OPV2V

### Latency Robustness ($\Delta t \in [0, 500]$ ms)

| Method | Comm Payload | 0 ms (Sync) | 100 ms | 200 ms | 300 ms | 400 ms | 500 ms | Drop (0→500ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ego-Only** | 0.0 KB | 57.2% | 57.8% | 58.3% | 58.9% | 58.2% | 57.8% | +0.6% (Bound) |
| **Late Fusion** | 2.0 KB | 83.6% | 70.4% | 66.3% | 66.3% | 65.0% | 64.2% | -19.4% |
| **F-Cooper** | 10.0 KB | 83.6% | 70.4% | 66.3% | 66.3% | 65.0% | 64.2% | -19.4% |
| **Where2comm** | 10.0 KB | 83.6% | 70.4% | 66.3% | 66.3% | 65.0% | 64.2% | -19.4% |
| **V2X-ViT** | 10.0 KB | 83.6% | 70.4% | 66.3% | 66.3% | 65.0% | 64.2% | -19.4% |
| **LRCP** | 10.0 KB | 83.6% | 81.0% | 76.2% | 72.1% | 69.9% | 67.2% | -16.4% |
| **TraF-Align** | 10.0 KB | 83.6% | 81.0% | 76.2% | 72.1% | 69.9% | 67.2% | -16.4% |
| **CoDynTrust** | 10.0 KB | 84.8% | 74.2% | 76.2% | 72.1% | 69.9% | 67.2% | -17.6% |
| **TrustDelay (Ours)** | **8.4 KB** | 68.7% | **81.1%** | **79.2%** | **75.6%** | **73.4%** | **70.6%** | **+1.9% (Best Asynch)** |

### Hardware Profiling on NVIDIA GeForce RTX 3050 Laptop GPU (6GB)
* **Inference Latency:** 27.1 ms end-to-end (~**36.9 FPS** real-time throughput)
* **VRAM Consumption:** ~720 MiB (fits easily in edge hardware)
* **Transmission Payload:** 8.4 KB/frame (vs. 200–600 KB for intermediate BEV maps)

---

## 5. Repository Structure

```
TrustDelay/
├── configs/
│   └── config.py                   # Central dataclass configuration
├── src/
│   ├── data/
│   │   └── opv2v_dataset.py        # Asynchronous OPV2V sequence dataset loader
│   ├── models/
│   │   ├── encoder.py              # Module 1: Ego BEV encoder & query generator
│   │   ├── collaborator_selector.py# Module 2: Top-K collaborator selection
│   │   ├── motion_extrapolator.py  # Module 3: Uncertainty-aware motion extrapolator
│   │   ├── bandwidth_selector.py   # Module 4: Bandwidth byte budget selector
│   │   ├── reliability_fusion.py   # Module 5: Reliability-weighted cross-attention
│   │   ├── trust_delay_model.py    # Unified end-to-end model
│   │   ├── baselines.py            # F-Cooper, Where2comm, LRCP, TraF-Align, etc.
│   │   └── loss.py                 # Focal, Smooth L1, and Heteroscedastic NLL losses
│   └── utils/
│       ├── metrics.py              # AP@0.3, AP@0.5, AP@0.7 for all/moving/static
│       └── perturbations.py        # Latency, packet loss, and pose noise simulation
├── scripts/
│   ├── train.py                    # Training pipeline on OPV2V
│   ├── evaluate_matrix.py          # Full robustness evaluation matrix
│   ├── run_ablations.py            # Systematic ablation study
│   └── generate_plots.py           # Publication-grade figures
├── docs/
│   ├── RESEARCH_GAP_AND_METHODOLOGY.md
│   └── EXPERIMENTAL_RESULTS.md
├── plots/                          # High-resolution figures embedded in README
└── results/                        # Checkpoints and JSON benchmark logs
```

---

## 6. Getting Started

### Prerequisites
```bash
pip install torch torchvision numpy scipy pyyaml matplotlib
```

### 1. Training
```bash
python scripts/train.py
```

### 2. Full Evaluation Matrix (Latency, Packet Loss, Pose Noise, Bandwidth)
```bash
python scripts/evaluate_matrix.py
```

### 3. Component Ablation Study
```bash
python scripts/run_ablations.py
```

### 4. Regenerating Publication Figures
```bash
python scripts/generate_plots.py
```

---

## 7. Citation & License
Released under the MIT License.
```bibtex
@article{trustdelay2026,
  title={TrustDelay: Uncertainty-Aware Asynchronous Cooperative Perception under Communication Constraints},
  author={Pritam et al.},
  journal={arXiv preprint},
  year={2026}
}
```
