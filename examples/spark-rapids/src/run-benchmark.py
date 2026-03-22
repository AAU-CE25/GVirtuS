# run_benchmark.py

"""
Benchmark harness that runs the pipeline across all configurations
and generates comparison reports.

Usage:
  python run_benchmark.py --configs cpu gpu gvirtus_smart --scale-factors 1 10
  python run_benchmark.py --configs all --scale-factors 1
"""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
try:
    from .config import RESULTS_DIR
except ImportError:
    from config import RESULTS_DIR


CONFIGURATIONS = {
    "cpu": {
        "name": "Spark CPU (No GPU)",
        "script": "pipeline_spark.py",
        "args": [],
        "description": "Baseline — pure CPU execution",
    },
    "gpu": {
        "name": "Spark RAPIDS (Local GPU)",
        "script": "pipeline_rapids.py",
        "args": [],
        "description": "RAPIDS with local GPU — best possible performance",
    },
    "gvirtus_tcp": {
        "name": "Spark RAPIDS (GVirtuS TCP)",
        "script": "pipeline_rapids.py",
        "args": ["--gvirtus"],
        "env": {"GVIRTUS_TRANSPORT": "tcp"},
        "description": "RAPIDS with remote vGPU over TCP — worst remote case",
    },
    "gvirtus_rdma": {
        "name": "Spark RAPIDS (GVirtuS RDMA)",
        "script": "pipeline_rapids.py",
        "args": ["--gvirtus"],
        "env": {"GVIRTUS_TRANSPORT": "rdma"},
        "description": "RAPIDS with remote vGPU over RDMA — no GPUDirect",
    },
    "gvirtus_smart": {
        "name": "Spark RAPIDS (GVirtuS Smart GPUDirect)",
        "script": "pipeline_rapids.py",
        "args": ["--gvirtus"],
        "env": {"GVIRTUS_TRANSPORT": "smart_gpudirect"},
        "description": "RAPIDS with remote vGPU — Smart GPUDirect (our contribution)",
    },
}


def run_single_benchmark(config_name: str, scale_factor: int, iteration: int) -> dict:
    """Run a single benchmark configuration."""
    config = CONFIGURATIONS[config_name]
    print(f"\n{'='*70}")
    print(f"  Config: {config['name']}")
    print(f"  Scale Factor: {scale_factor}")
    print(f"  Iteration: {iteration}")
    print(f"{'='*70}")

    # Set environment
    env = os.environ.copy()
    env["SCALE_FACTOR"] = str(scale_factor)
    if "env" in config:
        env.update(config["env"])

    # Run
    start = time.time()
    result = subprocess.run(
        ["python", config["script"]] + config["args"],
        env=env,
        capture_output=True,
        text=True,
    )
    wall_time = time.time() - start

    # Parse results
    results_file = f"results/sf{scale_factor}/{'spark_cpu' if config_name == 'cpu' else 'rapids_gpu'}_results.json"
    stage_timings = {}
    if os.path.exists(results_file):
        with open(results_file) as f:
            data = json.load(f)
            stage_timings = data.get("timings", {})

    return {
        "config": config_name,
        "config_name": config["name"],
        "scale_factor": scale_factor,
        "iteration": iteration,
        "wall_time": wall_time,
        "stage_timings": stage_timings,
        "success": result.returncode == 0,
        "stdout": result.stdout[-2000:] if result.stdout else "",
        "stderr": result.stderr[-2000:] if result.stderr else "",
    }


def generate_report(all_results: list):
    """Generate a comparison report."""
    print(f"\n{'='*70}")
    print(f"  BENCHMARK COMPARISON REPORT")
    print(f"{'='*70}\n")

    # Group by scale factor
    by_sf = {}
    for r in all_results:
        sf = r["scale_factor"]
        if sf not in by_sf:
            by_sf[sf] = {}
        config = r["config"]
        if config not in by_sf[sf]:
            by_sf[sf][config] = []
        by_sf[sf][config].append(r)

    for sf, configs in sorted(by_sf.items()):
        print(f"\n  Scale Factor: {sf} (~{sf}GB)")
        print(f"  {'─'*60}")
        print(f"  {'Configuration':<40} {'Avg Time':>10} {'vs CPU':>10} {'vs Local GPU':>12}")
        print(f"  {'─'*60}")

        cpu_time = None
        gpu_time = None

        for config_name, results in sorted(configs.items()):
            avg_time = sum(r["wall_time"] for r in results) / len(results)

            if config_name == "cpu":
                cpu_time = avg_time
            elif config_name == "gpu":
                gpu_time = avg_time

        for config_name, results in sorted(configs.items()):
            avg_time = sum(r["wall_time"] for r in results) / len(results)
            display_name = CONFIGURATIONS[config_name]["name"]

            vs_cpu = f"{cpu_time / avg_time:.2f}x" if cpu_time else "—"
            vs_gpu = f"{avg_time / gpu_time:.2f}x" if gpu_time else "—"

            marker = " ★" if config_name == "gvirtus_smart" else ""
            print(f"  {display_name:<40} {avg_time:>8.1f}s {vs_cpu:>10} {vs_gpu:>12}{marker}")

        # Stage-level breakdown
        print(f"\n  Stage Breakdown:")
        stages = [
            "1_data_loading", "2_revenue_analytics", "3_customer_360",
            "4_rfm_segmentation", "5_cohort_analysis", "6_funnel_analysis",
            "7_customer_clustering"
        ]

        print(f"  {'Stage':<25}", end="")
        for config_name in sorted(configs.keys()):
            short_name = config_name[:12]
            print(f" {short_name:>12}", end="")
        print()

        for stage in stages:
            print(f"  {stage:<25}", end="")
            for config_name in sorted(configs.keys()):
                results = configs[config_name]
                avg = sum(
                    r["stage_timings"].get(stage, 0) for r in results
                ) / len(results)
                print(f" {avg:>10.1f}s", end="")
            print()

    # Save full report
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = f"{RESULTS_DIR}/benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Full results saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Spark RAPIDS GVirtuS Benchmark")
    parser.add_argument(
        "--configs", nargs="+",
        default=["cpu", "gpu"],
        choices=list(CONFIGURATIONS.keys()) + ["all"],
        help="Configurations to benchmark"
    )
    parser.add_argument(
        "--scale-factors", nargs="+", type=int,
        default=[1],
        help="Scale factors to test (1=~1GB, 10=~10GB, 100=~100GB)"
    )
    parser.add_argument(
        "--iterations", type=int, default=3,
        help="Number of iterations per configuration"
    )
    parser.add_argument(
        "--generate-data", action="store_true",
        help="Generate test data before benchmarking"
    )
    args = parser.parse_args()

    configs = list(CONFIGURATIONS.keys()) if "all" in args.configs else args.configs

    # Generate data if needed
    if args.generate_data:
        for sf in args.scale_factors:
            print(f"\nGenerating data for scale factor {sf}...")
            env = os.environ.copy()
            env["SCALE_FACTOR"] = str(sf)
            subprocess.run(["python", "datagen.py"], env=env)

    # Run benchmarks
    all_results = []
    for sf in args.scale_factors:
        for config in configs:
            for iteration in range(1, args.iterations + 1):
                result = run_single_benchmark(config, sf, iteration)
                all_results.append(result)

    # Report
    generate_report(all_results)


if __name__ == "__main__":
    main()