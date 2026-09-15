"""
Multi-model evaluation script based on the latest test_PCP_cp:
- Supports configurable layer counts and seed lists
- For each sample: independent inference plus the latest PCP-cp MILP per model, then average revenue/time
- Passes an FCP fallback time limit to PCP-cp
- Loads models from --model_dir
"""

from __future__ import annotations

import time
import os
import argparse
from typing import List

import numpy as np
import torch
from tqdm import tqdm

# Reuse the model and pipeline from test_PCP_cp
from test_PCP_cp import (
    EdgeScoringGCN,
    process_data,
    revenue_ratio,
    top_m_selection,
    generate_progressive_bundles,
)


if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([EdgeScoringGCN])


def parse_list_arg(arg: str) -> List[int]:
    return [int(x) for x in arg.split(",") if x.strip()]


def parse_paths_arg(arg: str) -> List[str]:
    """Parse a semicolon-separated path string into a list."""
    return [p.strip() for p in arg.split(";") if p.strip()]


def _save_layer_result(nl: int, m: int, n: int, results_list: list, result_dir: str, test_folder_name: str = None) -> None:
    """Save results for one layer count and problem size to CSV."""
    import pandas as pd

    results_array = np.array(results_list)

    if test_folder_name:
        result_filename = f"test_result_PCP_cp_{nl}layer_{test_folder_name}.csv"
    else:
        result_filename = f"test_result_PCP_cp_{nl}layer_m{m}n{n}.csv"
    result_path = os.path.join(result_dir, result_filename)

    df = pd.DataFrame({
        'method': 'PCP_cp',
        'layers': nl,
        'm_segments': m,
        'n_products': n,
        'revenue_ratio': results_array[:, 0],
        'runtime_ratio': results_array[:, 1],
        'avg_total_time': results_array[:, 2],
        'avg_solve_time': results_array[:, 3],
        'avg_gcn_time': results_array[:, 4],
    })
    df.to_csv(result_path, index=False)
    print(f"✅ Saved: {result_path} ({len(results_array)} samples)")


def _save_seed_averages(nl: int, m: int, n: int, seed_results: dict, result_dir: str, test_folder_name: str = None) -> None:
    """
    Save each seed's average over all samples to CSV.
    """
    import pandas as pd

    seed_avg_data = []
    for seed, results in sorted(seed_results.items()):
        if len(results) > 0:
            results_array = np.array(results)
            seed_avg_data.append({
                'seed': seed,
                'num_samples': len(results),
                'avg_revenue_ratio': float(np.mean(results_array[:, 0])),
                'std_revenue_ratio': float(np.std(results_array[:, 0])),
                'avg_time_ratio': float(np.mean(results_array[:, 1])),
                'std_time_ratio': float(np.std(results_array[:, 1])),
                'avg_total_time': float(np.mean(results_array[:, 2])),
                'avg_solve_time': float(np.mean(results_array[:, 3])),
                'avg_gcn_time': float(np.mean(results_array[:, 4])),
            })

    if not seed_avg_data:
        return

    if test_folder_name:
        result_filename = f"test_result_PCP_cp_{nl}layer_{test_folder_name}_seed_avg.csv"
    else:
        result_filename = f"test_result_PCP_cp_{nl}layer_m{m}n{n}_seed_avg.csv"
    result_path = os.path.join(result_dir, result_filename)

    df = pd.DataFrame(seed_avg_data)
    df.insert(0, 'method', 'PCP_cp')
    df.insert(1, 'layers', nl)
    df.insert(2, 'm_segments', m)
    df.insert(3, 'n_products', n)

    df.to_csv(result_path, index=False)
    print(f"✅ Saved per-seed averages: {result_path} ({len(seed_avg_data)} seeds)")


def _save_seed_sample_results(rows: list, result_dir: str, test_folder_name: str) -> str | None:
    """Save the auditable seed-by-sample long table."""
    if not rows:
        return None

    import pandas as pd

    columns = [
        "method", "layers", "seed", "sample_file", "m_segments", "n_products",
        "revenue_ratio", "runtime_ratio", "total_time", "solve_time", "gcn_time",
    ]
    result_filename = f"test_result_PCP_cp_{rows[0]['layers']}layer_{test_folder_name}_seed_sample.csv"
    result_path = os.path.join(result_dir, result_filename)
    pd.DataFrame(rows, columns=columns).to_csv(result_path, index=False)
    print(f"✅ Saved seed×sample long table: {result_path} ({len(rows)} rows)")
    return result_path


def load_models(
    model_root: str,
    layers: List[int],
    seeds: List[int],
    device: torch.device,
) -> dict:
    """Load models for the requested layers and seeds. Returns {layer: [(seed, model, path), ...]}."""
    loaded = {nl: [] for nl in layers}
    for nl in layers:
        for sd in seeds:
            cand_paths = [
                os.path.join(model_root, f"best_model_edge_{nl}layer_seed{sd}.pt"),
                os.path.join(model_root, f"model_edge_{nl}layer_seed{sd}.pt"),
                os.path.join(model_root, f"model-{nl}layer_seed{sd}.pt"),
            ]
            path = next((p for p in cand_paths if os.path.exists(p)), None)
            if path is None:
                print(f"⚠️ Model not found: layer={nl}, seed={sd}, searched={cand_paths}")
                continue
            try:
                import __main__
                __main__.EdgeScoringGCN = EdgeScoringGCN
                mdl = torch.load(path, map_location=device, weights_only=False)
                mdl.to(device)
                mdl.eval()
                loaded[nl].append((sd, mdl, path))
                print(f"✅ Loaded model: layer={nl}, seed={sd}, path={path}")
            except Exception as e:
                print(f"❌ Failed to load layer={nl}, seed={sd}, path={path}: {e}")
    return loaded


def run_inference_only(mdl, data, n, m_segments):
    """Single-model GNN inference (wall-clock timed); returns sigmoid outputs."""
    inference_start = time.time()
    with torch.no_grad():
        raw_out = mdl(data)

        if isinstance(raw_out, dict):
            if "logit_matrix" in raw_out:
                logits_nm = raw_out["logit_matrix"].detach().cpu().numpy()
            elif "edge_logits" in raw_out:
                s = raw_out["edge_logits"].detach().cpu().numpy()
                logits_nm = s.reshape(n, m_segments)
            else:
                raise ValueError("Unexpected model output keys: " + ",".join(raw_out.keys()))
            sigmoid_output = 1.0 / (1.0 + np.exp(-logits_nm))
        else:
            output = raw_out[:n, :].detach().cpu().numpy()
            sigmoid_output = np.exp(output) / (np.exp(output) + np.exp(1))
    inference_time = time.time() - inference_start
    return sigmoid_output, inference_time


def process_and_solve_milp(
    sigmoid_output,
    n,
    m_segments,
    unit_cs,
    ship_cs,
    unit_us,
    Ns,
    opt_rev,
    stored_cs,
    stored_Rs,
    fcp_fallback_time_limit,
):
    """Process inference output and solve the latest cutting-plane / lazy-cut PCP MILP."""
    selected_products = top_m_selection(sigmoid_output, m=n, threshold=0.5)
    feasible_bundles = generate_progressive_bundles(selected_products, n)

    ratio, solve_time = revenue_ratio(
        n,
        m_segments,
        unit_cs,
        ship_cs,
        unit_us,
        Ns,
        opt_rev,
        feasible_bundles,
        selected_products,
        stored_cs,
        stored_Rs,
        fcp_fallback_time_limit=fcp_fallback_time_limit,
    )
    return ratio, solve_time


def main():
    parser = argparse.ArgumentParser(description="Multi-model average evaluation (based on test_PCP_cp)")
    parser.add_argument("--data_dir", type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    parser.add_argument("--test_subdirs", type=str, default="data/deterministic/test_m10n10_correct_1e_3;data/deterministic/test_m20n10_correct_1e_3;data/deterministic/test_m30n10_correct_1e_3", help="Test data subdirectories (semicolon separated)")
    parser.add_argument("--model_dir", type=str, default="models/main_base_4layer_correct_lr_3", help="Model directory")
    parser.add_argument("--layers", type=str, default="4", help="Layer counts to use, comma separated")
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5,6,7,8,9,10", help="Seeds to use, comma separated")
    parser.add_argument("--result_dir", type=str, default="results/pcp", help="Output directory")
    
    parser.add_argument("--fcp_fallback_time_limit", type=float, default=60.0, help="Time limit (seconds) for the FCP fallback when PCP-cp finds no feasible solution")
    parser.add_argument(
        "--save_result",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save result files",
    )
    args = parser.parse_args()

    dir_path = args.data_dir
    test_subdirs = parse_paths_arg(args.test_subdirs)
    model_root = os.path.join(dir_path, args.model_dir)
    save_result = args.save_result
    result_dir = os.path.join(dir_path, args.result_dir) if save_result else None

    layers = parse_list_arg(args.layers)
    seeds = parse_list_arg(args.seeds)

    if save_result:
        os.makedirs(result_dir, exist_ok=True)

    print(f"📂 data_dir: {dir_path}")
    print(f"🧪 test_subdirs: {test_subdirs}")
    print(f"🗂 model_root: {model_root}")
    print(f"🔢 layers: {layers}")
    print(f"🌱 seeds: {seeds}")
    if save_result:
        print(f"💾 result_dir: {result_dir}")
    else:
        print("💾 Result files will not be saved")
    print(f"⏱ FCP fallback time limit: {args.fcp_fallback_time_limit}s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    models_by_layer = load_models(model_root, layers, seeds, device)
    if not any(models_by_layer.values()):
        print("No models loaded; exiting.")
        return

    for test_subdir in test_subdirs:
        results_by_layer = {nl: {} for nl in layers}
        results_by_seed = {nl: {sd: {} for sd, _, _ in models_by_layer[nl]} for nl in layers}
        seed_sample_rows_by_layer = {nl: [] for nl in layers}
        test_data_path = os.path.join(dir_path, test_subdir)

        if not os.path.exists(test_data_path):
            print(f"⚠️ Test data path does not exist: {test_data_path}; skipping")
            continue

        dir_list = os.listdir(test_data_path)
        test_dataset = []
        misc_dataset = []
        file_names = []

        print(f"\n📊 Reading test set: {test_subdir}")
        for fname in dir_list:
            if fname == ".DS_Store":
                continue
            fpath = os.path.join(test_data_path, fname)
            try:
                dat, misc = process_data(fpath)
                test_dataset.append(dat)
                misc_dataset.append(misc)
                file_names.append(fname)
            except Exception as e:
                print(f"Failed to read {fname}: {e}")
                continue

        sample_num = len(test_dataset)
        print(f"✅ Loaded {sample_num} samples")
        if sample_num == 0:
            continue

        for i in tqdm(range(sample_num), desc=f"Evaluating {test_subdir}"):
            try:
                dat = test_dataset[i].to(device)
                misc = misc_dataset[i]
                n, m_segments, unit_cs, ship_cs, unit_us, Ns, opt_bundles, opt_prices, opt_rev, running_time, gap, stored_cs, stored_Rs = misc

                for nl in layers:
                    if nl not in models_by_layer or not models_by_layer[nl]:
                        continue

                    model_ratios = []
                    model_time_ratios = []
                    model_total_times = []
                    model_solve_times = []
                    model_gcn_times = []

                    for sd, mdl, path in models_by_layer[nl]:
                        try:
                            sigmoid_output, gcn_time = run_inference_only(mdl, dat, n, m_segments)

                            ratio, solve_time = process_and_solve_milp(
                                sigmoid_output, n, m_segments, unit_cs, ship_cs,
                                unit_us, Ns, opt_rev, stored_cs, stored_Rs,
                                args.fcp_fallback_time_limit,
                            )

                            total_time = gcn_time + solve_time
                            time_ratio = total_time / running_time if running_time > 0 else float("inf")

                            model_ratios.append(ratio)
                            model_time_ratios.append(time_ratio)
                            model_total_times.append(total_time)
                            model_solve_times.append(solve_time)
                            model_gcn_times.append(gcn_time)

                            key = (m_segments, n)
                            if key not in results_by_seed[nl][sd]:
                                results_by_seed[nl][sd][key] = []
                            results_by_seed[nl][sd][key].append([ratio, time_ratio, total_time, solve_time, gcn_time])
                            seed_sample_rows_by_layer[nl].append({
                                "method": "PCP_cp", "layers": nl, "seed": sd,
                                "sample_file": file_names[i], "m_segments": m_segments,
                                "n_products": n, "revenue_ratio": float(ratio),
                                "runtime_ratio": float(time_ratio), "total_time": float(total_time),
                                "solve_time": float(solve_time), "gcn_time": float(gcn_time),
                            })

                        except Exception as e_model:
                            print(f"Model failed sample={file_names[i] if i < len(file_names) else i}, layer={nl}, seed={sd}, path={path}: {e_model}")
                            continue

                    if len(model_ratios) > 0:
                        key = (m_segments, n)
                        if key not in results_by_layer[nl]:
                            results_by_layer[nl][key] = []

                        result_entry = [
                            float(np.mean(model_ratios)),
                            float(np.mean(model_time_ratios)),
                            float(np.mean(model_total_times)),
                            float(np.mean(model_solve_times)),
                            float(np.mean(model_gcn_times)),
                        ]
                        results_by_layer[nl][key].append(result_entry)
            except Exception as e:
                print(f"Sample {file_names[i] if i < len(file_names) else i} evaluation failed: {e}")
                continue

        test_folder_name = os.path.basename(test_subdir.rstrip('/'))

        if save_result:
            print(f"\n💾 Saving results for {test_folder_name}...")
            for nl in layers:
                for key, results in results_by_layer[nl].items():
                    if results:
                        m, n = key
                        _save_layer_result(nl, m, n, results, result_dir, test_folder_name)
                        seed_results_for_key = {
                            sd: results_by_seed[nl][sd].get(key, [])
                            for sd in results_by_seed[nl].keys()
                        }
                        _save_seed_averages(nl, m, n, seed_results_for_key, result_dir, test_folder_name)
                        _save_seed_sample_results(seed_sample_rows_by_layer[nl], result_dir, test_folder_name)

        print(f"\n{'='*80}")
        print(f"📊 {test_folder_name} evaluation results")
        print(f"{'='*80}")
        for nl in layers:
            if results_by_layer[nl]:
                total_samples = sum(len(v) for v in results_by_layer[nl].values())
                print(f"\n[Layer {nl}] Samples: {total_samples}")
                for (m, n), results in sorted(results_by_layer[nl].items()):
                    if results:
                        results_array = np.array(results)
                        print(f"  m={m}, n={n}: {len(results_array)} samples")
                        print(f"    Revenue Ratio - Mean: {np.mean(results_array[:, 0]):.4f}, Std: {np.std(results_array[:, 0]):.4f}, "
                              f"Min: {np.min(results_array[:, 0]):.4f}, Max: {np.max(results_array[:, 0]):.4f}")
                        print(f"    Time Ratio - Mean: {np.mean(results_array[:, 1]):.4f}, Std: {np.std(results_array[:, 1]):.4f}, "
                              f"Min: {np.min(results_array[:, 1]):.4f}, Max: {np.max(results_array[:, 1]):.4f}")
                        print(f"    Avg Times - Total: {np.mean(results_array[:, 2]):.4f}s, Solve: {np.mean(results_array[:, 3]):.4f}s, GCN: {np.mean(results_array[:, 4]):.4f}s")
        print(f"{'='*80}\n")

    print("\n" + "="*80)
    print("🎉 All datasets evaluated!")
    print("="*80)
    print(f"Test datasets: {len(test_subdirs)}")
    print(f"Layers: {layers}")
    print(f"Models per layer count: {len(seeds)}")
    if save_result:
        print(f"Output directory: {result_dir}")
    else:
        print("Result files: not saved")
    print("="*80)


if __name__ == "__main__":
    main()
