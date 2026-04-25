"""
analyze_min_noise_levels.py — Analyze t_i^min values from find_min_noise_levels.py output.

Loads the JSON produced by find_min_noise_levels.py and prints summary statistics
in both the original t-space and log-SNR space.

For rectified flow (alpha = 1-t, sigma = t):

    log-SNR_i = log((1 - t_i) / t_i)

This maps t in (0, 1) to log-SNR in (-inf, +inf):
  t → 0   =>  log-SNR → +inf  (clean, no noise needed)
  t = 0.5 =>  log-SNR = 0
  t → 1   =>  log-SNR → -inf  (fully noisy)

Usage:
    python scripts/analyze_min_noise_levels.py \
        --input /deepfreeze/pnlong/ambient/spdmx_min_noise_levels.json

Options:
    --clip-inf FLOAT   Replace ±inf with ±FLOAT for clipped stats (default: 20.0)
    --bins INT         Number of histogram bins in log-SNR space (default: 20)
    --tau FLOAT        Tau used during annotation (display only)
"""

import argparse
import json
import numpy as np


def log_snr(t: np.ndarray) -> np.ndarray:
    """log-SNR = log((1-t) / t) for rectified flow."""
    return np.log((1.0 - t) / t)


def print_summary(label: str, values: np.ndarray, n_posinf: int = 0, n_neginf: int = 0):
    finite = values[np.isfinite(values)]
    print(f"\n--- {label} ---")
    if n_posinf or n_neginf:
        print(f"  +inf: {n_posinf}   -inf: {n_neginf}   finite: {len(finite)}")
    print(f"  mean:   {finite.mean():.4f}")
    print(f"  std:    {finite.std():.4f}")
    print(f"  min:    {finite.min():.4f}")
    print(f"  p5:     {np.percentile(finite,  5):.4f}")
    print(f"  p25:    {np.percentile(finite, 25):.4f}")
    print(f"  median: {np.median(finite):.4f}")
    print(f"  p75:    {np.percentile(finite, 75):.4f}")
    print(f"  p95:    {np.percentile(finite, 95):.4f}")
    print(f"  max:    {finite.max():.4f}")


def print_histogram_fixed(values: np.ndarray, buckets: list, label: str, n_total: int, bar_width: int = 40):
    print(f"\n  Distribution of {label}:")
    for lo, hi in zip(buckets[:-1], buckets[1:]):
        count = int(((values >= lo) & (values < hi)).sum())
        bar = "#" * (count * bar_width // max(n_total, 1))
        print(f"    [{lo:.2f}, {hi:.2f})  {count:7d}  {bar}")


def print_histogram_auto(values: np.ndarray, n_bins: int, label: str, bar_width: int = 40):
    finite = values[np.isfinite(values)]
    edges = np.linspace(finite.min(), finite.max(), n_bins + 1)
    counts, _ = np.histogram(finite, bins=edges)
    max_count = counts.max()
    print(f"\n  Distribution of {label}:")
    for i, count in enumerate(counts):
        bar = "#" * (count * bar_width // max(max_count, 1))
        print(f"    [{edges[i]:7.3f}, {edges[i+1]:7.3f})  {count:7d}  {bar}")


def main():
    parser = argparse.ArgumentParser(description="Analyze t_i^min values in t-space and log-SNR space.")
    parser.add_argument("--input", required=True, help="Path to the JSON output from find_min_noise_levels.py.")
    parser.add_argument("--clip-inf", type=float, default=20.0,
                        help="Clamp ±inf to ±FLOAT for clipped stats (default: 20.0).")
    parser.add_argument("--bins", type=int, default=20,
                        help="Number of histogram bins in log-SNR space (default: 20).")
    parser.add_argument("--tau", type=float, default=None,
                        help="Optional tau used during annotation (for display only).")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    t_values = np.array(list(data.values()), dtype=float)
    n_total = len(t_values)
    tau_str = f"tau={args.tau}" if args.tau is not None else "tau=?"

    print(f"\nLoaded {n_total} t_i^min values from {args.input}")

    # ------------------------------------------------------------------
    # Raw t-space
    # ------------------------------------------------------------------
    print_summary(f"t_i^min  ({tau_str})", t_values)

    buckets_t = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]
    print_histogram_fixed(t_values, buckets_t, "t_i^min", n_total)

    # ------------------------------------------------------------------
    # Log-SNR space: log((1-t) / t)
    # ------------------------------------------------------------------
    n_zero = int((t_values == 0.0).sum())   # t=0  → log-SNR = +inf
    n_one  = int((t_values == 1.0).sum())   # t=1  → log-SNR = -inf

    t_safe   = np.clip(t_values, 1e-12, 1.0 - 1e-12)
    lsnr     = log_snr(t_safe)
    lsnr[t_values == 0.0] =  np.inf
    lsnr[t_values == 1.0] = -np.inf

    print_summary(
        "log-SNR  =  log((1-t) / t)",
        lsnr,
        n_posinf=n_zero,
        n_neginf=n_one,
    )

    lsnr_clipped = np.clip(lsnr, -args.clip_inf, args.clip_inf)
    print(f"\n  (clipped to ±{args.clip_inf})")
    print(f"  mean:   {lsnr_clipped.mean():.4f}")
    print(f"  std:    {lsnr_clipped.std():.4f}")
    print(f"  median: {np.median(lsnr_clipped):.4f}")

    # Use the same t-bucket boundaries converted to log-SNR so that counts
    # correspond directly to the t-space histogram above.
    # log-SNR is a decreasing function of t, so the bucket edges flip sign:
    #   t in [lo, hi)  ↔  log-SNR in (log_snr(hi), log_snr(lo)]
    # We represent this as buckets in decreasing log-SNR order to match reading order.
    inner_t_edges = [t for t in buckets_t[1:-1] if 0.0 < t < 1.0]
    lsnr_edges = sorted(
        [log_snr(np.array([t]))[0] for t in inner_t_edges],
        reverse=True,
    )
    # Prepend +inf sentinel and append -inf sentinel to capture the full range
    lsnr_buckets = [np.inf] + lsnr_edges + [-np.inf]
    finite_lsnr = lsnr[np.isfinite(lsnr)]
    print(f"\n  Distribution of log-SNR  (buckets correspond to same t-bins above):")
    for lo_s, hi_s in zip(lsnr_buckets[:-1], lsnr_buckets[1:]):
        lo_str = f"{lo_s:7.3f}" if np.isfinite(lo_s) else "    +inf"
        hi_str = f"{hi_s:7.3f}" if np.isfinite(hi_s) else "    -inf"
        count = int(((lsnr >= hi_s) & (lsnr < lo_s)).sum())
        bar = "#" * (count * 40 // max(n_total, 1))
        print(f"    ({hi_str}, {lo_str}]  {count:7d}  {bar}")

    # ------------------------------------------------------------------
    # Reference table
    # ------------------------------------------------------------------
    print("\n  Reference: t  →  log-SNR = log((1-t)/t)")
    refs = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 0.999, 1.0]
    for t_ref in refs:
        if t_ref == 0.0:
            s = "+inf"
        elif t_ref == 1.0:
            s = "-inf"
        else:
            s = f"{log_snr(np.array([t_ref]))[0]:+.4f}"
        print(f"    t={t_ref:.3f}  →  log-SNR={s}")


if __name__ == "__main__":
    main()
