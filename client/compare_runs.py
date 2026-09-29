"""Compare runs side by side from the summary_<timestamp>.json files in the output folder.

Writes comparison.csv next to them and prints the main columns.
"""

import argparse
import csv
import json
from pathlib import Path

COLUMNS = [
    "timestamp", "model", "tag", "images", "failed", "concurrency",
    "images_per_s", "latency_s_p50", "ttft_s_p50", "decode_tps_p50",
    "gpu_mem_baseline_mib", "gpu_mem_peak_mib", "kv_cache_peak_tokens",
    "gpu_util_mean_pct", "gpu_power_mean_w", "energy_j_per_image",
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="/data/output")
    args = p.parse_args()
    out = Path(args.output_dir)

    runs = []
    for path in sorted(out.glob("summary_*.json")):
        s = json.loads(path.read_text())
        ok = s["images"] - s["failed"]
        if s.get("gpu_energy_j") is not None and ok:
            s["energy_j_per_image"] = round(s["gpu_energy_j"] / ok, 1)
        runs.append(s)
    if not runs:
        print(f"No summary_*.json in {out}")
        return

    fields = COLUMNS + sorted({k for s in runs for k in s} - set(COLUMNS))
    with (out / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(runs)

    cells = [[str(s.get(c, "")) for c in COLUMNS] for s in runs]
    widths = [max(len(c), *(len(r[i]) for r in cells)) for i, c in enumerate(COLUMNS)]
    for row in [COLUMNS, *cells]:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))
    print(f"\nFull table: {out / 'comparison.csv'}")


if __name__ == "__main__":
    main()
