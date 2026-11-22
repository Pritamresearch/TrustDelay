# Deprecated Oracle Evaluation Artifacts

## Notice of Invalidation
The experimental results and plots archived in this directory have been permanently deprecated and invalidated following a repository audit on 2026-11-21.

### Why These Results Are Invalid
1. **Network Bypass:** The original evaluation scripts (`scripts/evaluate_matrix.py`, `scripts/run_ablations.py`) never ran the neural network model. Predictions were synthesized by extracting ground-truth bounding box coordinates directly from YAML files and scoring them with arbitrary hand-coded confidence constants.
2. **Artificial Baseline Ranking:** Published baselines (Where2comm, V2X-ViT, F-Cooper, etc.) were not executed using their official implementations or weights. They were assigned hardcoded confidence constants (0.85, 0.88, 0.90, 0.92) to manually force desired Precision-Recall ranking.
3. **Mislabeled Frame Rate:** Delays were computed assuming 30 Hz (33.3 ms/frame), whereas OPV2V sequence timestamps show 10 Hz (100 ms/frame). All previous delay values were mislabeled by 3x.
4. **Synthetic Heatmaps:** Qualitative reliability visualizations used hard-coded Gaussian mixture formulas rather than learned network activations.
5. **Efficiency Fiction:** Latency (27.1 ms), throughput (36.9 FPS), and memory figures (720 MiB) in the original README were unmeasured literals.

All trustworthy, reproducible metrics are generated from real inference on sensor data and tracked on branch `fix/real-evaluation`.
