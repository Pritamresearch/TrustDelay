"""
CI Test: Assert Zero Hardcoded AP, Latency, or KB Literals in Evaluation Pipeline.
Grep scripts and assert that all metrics, timings, and communication figures are dynamically computed.
"""
from __future__ import annotations
import os
import re
import sys

def test_no_hardcoded_literals():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    scripts_to_check = [
        os.path.join(repo_root, "scripts", "evaluate.py"),
        os.path.join(repo_root, "scripts", "run_ablations.py"),
    ]

    # Patterns representing previous oracle shortcuts
    forbidden_patterns = [
        r"collab_scores\.append\s*\(\s*0\.8[58]",
        r"collab_scores\.append\s*\(\s*0\.9[02]",
        r"acc\s*=\s*0\.2\s+if",
        r"comm_kb\s*=\s*(?:8\.4|24\.6|38\.2|10\.0)",
        r"np\.random\.normal.*#.*Gaussian mixture",
    ]

    for script_path in scripts_to_check:
        assert os.path.exists(script_path), f"Script {script_path} does not exist!"
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()

        for pat in forbidden_patterns:
            matches = re.findall(pat, content)
            assert len(matches) == 0, (
                f"VIOLATION: Found forbidden oracle pattern '{pat}' in {script_path}!\n"
                f"Matches: {matches}"
            )

    print("PASS: Zero hardcoded oracle patterns detected in evaluation scripts.")

if __name__ == "__main__":
    test_no_hardcoded_literals()
