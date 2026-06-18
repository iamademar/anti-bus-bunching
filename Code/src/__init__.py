"""Anti-Bunching Copilot — experiment source package.

Modules:
    config     - load config.yaml and resolve dataset paths
    headway    - load OD, reconstruct per-bus trajectories, forward headway, segment/dwell
    features   - causal feature builder + forward-looking bunching label
    baselines  - simple non-ML baselines (threshold / persistence / historical average)
    experiment - CapyMOA prequential run (HAT / ARF) + metrics + drift log
    plots      - figures (string diagram, rolling metrics, lead-time histogram)
"""
