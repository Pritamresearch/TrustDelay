# TrustDelay: Uncertainty-Aware Asynchronous Cooperative Perception under Communication Constraints

## 1. Executive Summary & Research Gap

### 1.1 What is Already Solved
Cooperative perception improves an ego vehicle's situational awareness by sharing intermediate representations across connected autonomous vehicles (CAVs) or roadside units (RSUs). Intermediate-feature methods such as **F-Cooper**, **DiscoNet**, **V2X-ViT**, and **Where2comm** exchange compact BEV feature representations rather than raw sensor streams.

To address communication latency, recent asynchronous cooperative perception frameworks have proposed various compensation strategies:
- **LRCP** (ICLR 2024): Compensates temporal asynchrony by predicting BEV feature flow from cached historical representations.
- **TraF-Align** (CVPR 2025): Employs trajectory-aware feature alignment to correct spatial misalignments and preserve semantic consistency across agents.
- **CoDynTrust**: Utilizes linear extrapolation for vehicle movement within common latency ranges (0–500 ms).
- **Temporal Residual Bottleneck**: Formulates asynchronous cooperative perception as temporal residual prediction, reporting better preservation of AP under delay.

---

### 1.2 The Unresolved Research Gap
The unresolved problem is not simply *"delay exists."* Rather, current methods compensate delayed features without jointly deciding **whether, from whom, and how much information should be requested and fused under realistic multi-source uncertainty**.

| Missing Capability | Why it Matters in Real-World Deployment |
| :--- | :--- |
| **Per-Object Motion Uncertainty** | A parked car and a fast-moving, accelerating vehicle should not be extrapolated with identical confidence. High-velocity maneuvers create high positional variance under latency. |
| **Heterogeneous Delay** | Different collaborators experience disparate sensing, processing, and wireless transmission delays ($\Delta t_i \in [0, 500]$ ms). |
| **Packet Loss** | Stochastic wireless dropouts (10%–50%) should not corrupt the fused ego BEV representation with zero-filled or stale artifacts. |
| **Pose Misalignment** | Small GPS/IMU localization errors (0.1–0.5 m, 0.1°–0.5°) compound into large BEV misalignments at long range. |
| **Bandwidth Budget** | The ego vehicle must selectively request only high-value collaborator evidence under strict byte budgets (1 KB, 5 KB, 10 KB, 20 KB). |
| **Heterogeneous Agents** | CAVs and RSUs operate with varying sensor modalities, resolutions, and local model architectures. |

**Defensible Gap Statement:**
> *Existing asynchronous cooperative perception methods compensate delayed features, but lack a unified framework for uncertainty-aware collaborator selection, bandwidth-constrained message selection, motion extrapolation, and reliability-weighted fusion under delay, packet loss, and pose noise.*

---

## 2. Mathematical Problem Formulation

For each collaborator message $m_i$ transmitted at $t_{\text{send},i}$ and received at $t_{\text{recv},i}$, the ego vehicle computes the latency:
$$\Delta t_i = t_{\text{recv},i} - t_{\text{send},i}$$

The delayed object or query token state $x_{i, t-\Delta t_i}$ is extrapolated to the current ego timestamp $t$:
$$\hat{x}_{i,t} = f_{\theta}(x_{i,t-\Delta t_i}, v_i, a_i, \Delta t_i, \sigma_i)$$
where $f_{\theta}$ incorporates second-order kinematic equations along with a learned residual network and heteroscedastic uncertainty estimation $\sigma_i$.

A dynamic reliability weight $w_i \in [0, 1]$ is assigned:
$$w_i = g_{\phi}(\Delta t_i, \sigma_i, p_{\text{loss}}, \epsilon_{\text{pose}})$$
which down-weights messages experiencing severe delay, high motion variance, packet dropout, or pose calibration errors. The final fused detection uses $w_i$ in a cross-attention transformer decoder:
$$\mathbf{Q}_{\text{fused}} = \mathbf{Q}_{\text{ego}} + \text{CrossAttention}\left(\mathbf{Q}_{\text{ego}}, \mathbf{K}_{\text{collab}} \odot \mathbf{w}\right)$$

---

## 3. Five-Module Architecture

```
                                  [ SURROUND SENSORS (LiDAR / CAMERA) ]
                                                    │
                                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ MODULE 1: EGO PERCEPTION ENCODER                                                                       │
│  • Compact BEV Backbone: Multi-scale residual convolution                                              │
│  • Query Generator: Extracts sparse candidate 3D queries Q_ego and anchor bounding boxes               │
└───────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ MODULE 2: COLLABORATOR SELECTOR                                                                        │
│  • Evaluates surround candidates: Gain, Distance, Relative Velocity, Overlap, Comm Cost                │
│  • Selects optimal Top-K collaborators: K = min(K_budget, N_candidates)                                │
└───────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ MODULE 3: UNCERTAINTY-AWARE MOTION EXTRAPOLATOR                                                        │
│  • Kinematic Projection: x_hat = x_{t-dt} + v*dt + 0.5*a*(dt^2) + delta_theta                         │
│  • Uncertainty Head: Outputs log(sigma^2) and motion confidence c_unc = exp(-sigma)                    │
└───────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ MODULE 4: BANDWIDTH-AWARE MESSAGE SELECTOR                                                             │
│  • Utility Ranking: U_j = c_j * exp(-0.5 * sigma_j)                                                    │
│  • Budget Enforcement: Prunes tokens to strictly fit byte budgets (1 KB, 5 KB, 10 KB, 20 KB)            │
│  • Channel Loss Mask: Simulates stochastic packet loss (0% - 50%)                                      │
└───────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ MODULE 5: RELIABILITY-WEIGHTED ASYNCHRONOUS FUSION                                                     │
│  • Gating Network: w_i = g_phi(Delta t_i, sigma_i, p_loss, epsilon_pose)                               │
│  • Transformer Cross-Attention: Fuses ego queries with reliability-scaled collaborator tokens           │
│  • Refinement Heads: 3D bounding box regression [cx, cy, cz, dx, dy, dz, yaw] + Classification Logits  │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```
