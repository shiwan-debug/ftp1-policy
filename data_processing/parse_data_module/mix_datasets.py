import os
import argparse
import random
import shutil
import math
import sys
import glob
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# Add path to import common modules
current_dir = os.path.dirname(os.path.abspath(__file__))
# data_processing/parse_data_module -> data_processing -> common
data_processing_dir = os.path.dirname(current_dir)
if data_processing_dir not in sys.path:
    sys.path.insert(0, data_processing_dir)

try:
    from common.replay_buffer import ReplayBuffer
except ImportError:
    print("[Warning] Could not import ReplayBuffer. Zarr stats might fail.", file=sys.stderr)
    ReplayBuffer = None

# ==============================================================================
# CONFIGURATION SECTION
# ==============================================================================
# Define how datasets are grouped, and the temperature 'alpha' for intra-group sampling.
# alpha = 0: Uniform sampling within the group (datasets contribute equal weight).
# alpha = 1: Proportional sampling within the group by total steps (frame count).
# 0 < alpha < 1: Intermediate behavior.
#
# 'datasets' list should contain the folder names of the datasets in the source directory.
DATASET_GROUPS = {
    # Example Group 1: Human Data
    "Human": {
        "datasets": [
            "AetherData_260121_5H_v0211",
            "AetherData_260204_7.5H_v0211",
            "AetherData_20260331",
            "OpenTouch",
            "Paxini-new_v0214",
        ],
        "alpha": 0.4, 
    },
    # Example Group 2: Dexterous Hands
    "DexHand": {
        "datasets": [
            # "SharpaPretrain",
            "sharpa",
            "MotionTrans",
            # "HATO",
            "HumanoidEveryday",
            "DexumiXHand",
            "DexumiInspire",
            "OpenLETDex",
        ],
        "alpha": 0.35, 
    },
    # Example Group 3: Grippers & UMI
    "GripperAndUMI": {
        "datasets": [
            "FreeTacMan",
            "TouchInTheWild",
            "ViTaMIn",
            # "RH20TCfg1OptoForce",
            # "RH20TCfg2ATIAxia",
            # "RH20TCfg3ATIAxia",
            # "RH20TCfg4ATIAxia",
            "RH20TCfg5Franka",
            "RH20TCfg6ATIAxia",
            "RH20TCfg7Tactile",
            "RDP",
            "RDP_Bimanual",
            "VLA_touch",
            "Unit",
            "Unit_Bimanual",
            "exUMI",
            "REASSEMBLE",
            "VisuoTactile_D-WHEEL",
            "VisuoTactile_QINGLOONG",
        ],
        "alpha": 0.2,
    }
}

# Define the target proportion for each group in the final dataset.
# These should sum to 1.0 approx.
GROUP_PROPORTIONS = {
    "Human": 0.2,
    "DexHand": 0.3,
    "GripperAndUMI": 0.5,
}

# ==============================================================================

def find_zarr_files(dataset_path):
    """Find all .zarr directories directly inside the dataset path."""
    if not os.path.exists(dataset_path):
        return []
    
    files = [f for f in os.listdir(dataset_path) if f.endswith('.zarr')]
    full_paths = [os.path.join(dataset_path, f) for f in files]
    return sorted(full_paths)

def get_zarr_stats(zarr_path):
    """Returns (n_episodes, n_steps) for a zarr file."""
    if ReplayBuffer is None:
        return (1, 1) # Fallback if import failed
        
    try:
        # ReplayBuffer.create_from_path is usually efficient, reading mostly metadata
        rb = ReplayBuffer.create_from_path(zarr_path, mode='r')
        return (rb.n_episodes, rb.n_steps)
    except Exception as e:
        # Fallback if corrupted
        # print(f"Error reading {zarr_path}: {e}")
        return (0, 0)

def scan_datasets(source_root, groups_config):
    """
    Scan all datasets and retrieve episode/step counts for each zarr file.
    Returns: dict structure with detailed info.
    {
        dataset_name: {
            'zarrs': [
                {'path': ..., 'episodes': ..., 'steps': ...},
                ...
            ],
            'total_episodes': N,
            'total_steps': M,
            'file_count': K
        }
    }
    """
    results = {}
    print("Scanning datasets for episode stats (this may take a while)...")
    
    all_datasets_in_config = []
    for group_info in groups_config.values():
        all_datasets_in_config.extend(group_info["datasets"])
        
    for ds_name in tqdm(all_datasets_in_config, desc="Scanning Datasets"):
        ds_path = os.path.join(source_root, ds_name)
        info = {
            'zarrs': [],
            'total_episodes': 0,
            'total_steps': 0,
            'file_count': 0
        }
        
        if os.path.exists(ds_path):
            zarr_paths = find_zarr_files(ds_path)
            for z_path in zarr_paths:
                eps, steps = get_zarr_stats(z_path)
                if eps > 0:
                    info['zarrs'].append({
                        'path': z_path,
                        'episodes': eps,
                        'steps': steps,
                        'name': os.path.basename(z_path)
                    })
                    info['total_episodes'] += eps
                    info['total_steps'] += steps
            
            info['file_count'] = len(info['zarrs'])
            
        results[ds_name] = info
        
    return results

def create_dataset_config(output_json_path, mix_plan, dataset_info):
    """
    Create a new data config JSON file based on the mix plan.
    mix_plan: dict ds_name -> list of (zarr_entry, num_copies) -- DEPRECATED structure
    
    New Logic:
    We generate a list of dataset entries for the JSON config.
    For each original dataset, we calculate the total 'scale' required.
    Then we break it down into multiple entries:
      - Integer part: N entries with ratio 1.0
      - Fractional part: 1 entry with ratio = remainder
    """
    
    # Flatten the mix plan to get total scale per dataset
    # The original mix_plan was designed for file-level copying.
    # To adapt to config-based mixing, we need to aggregate the 'copies' back to a dataset-level scale factor.
    # But wait, 'mix_plan' in the previous step was calculated per zarr file.
    # If we want to use 'use_trajectory_ratio', we generally apply it to the whole dataset (all zarrs inside it).
    # Since MultiZarrDataset/ZarrDataset structure usually treats one folder as one domain/dataset.
    
    # We will reconstruct the 'Scale' from the passed mix_plan or better, 
    # we should modify calculate_mix to return dataset-level scales, not file-level assignments.
    # However, to minimize changes, let's aggregate here.
    
    # But wait, MultiZarrDataset usually takes a list of paths.
    # If "Aether" has 10 zarr files inside, MultiZarrDataset usually expects the PATH to "Aether" 
    # and it loads all zarrs inside. 
    # OR does it expect paths to individual zarrs?
    # Let's check finding_zarr_files: The original script scans subfolders.
    # The data_config example shows: "path": ".../Paxini_Check".
    # And ZarrDataset usually iterates zarrs inside that path.
    
    # So we should operate at the DATASET (Folder) level, not individual Zarr level.
    # The previous 'calculate_mix' went down to individual zarr files for stochastic rounding.
    # We can simplify this.
    
    pass

def generate_config_entries(dataset_weights, dataset_info, target_total_steps, group_proportions):
    """
    Re-implementing the core logic to output config entries instead of file copies.
    """
    config_datasets = []

    total_prop = sum(group_proportions.values())
    norm_group_props = {k: v / total_prop for k, v in group_proportions.items()}

    final_stats = []

    for group_name, weight_info in dataset_weights.items():
        g_prop = norm_group_props.get(group_name, 0)
        group_target_total_steps = target_total_steps * g_prop

        for ds in weight_info['valid_list']:
            w = weight_info['datasets'][ds]
            fraction = w / weight_info['denom'] if weight_info['denom'] > 0 else 0
            ds_target_steps = group_target_total_steps * fraction

            ds_orig_steps = dataset_info[ds]['total_steps']
            ds_orig_eps = dataset_info[ds]['total_episodes']
            scaling_factor = ds_target_steps / ds_orig_steps if ds_orig_steps > 0 else 0

            # Now generate config entries
            # 1. Integer parts (Full copies)
            full_copies = int(scaling_factor)
            remainder = scaling_factor - full_copies

            # To avoid creating too many entries for huge upsampling (e.g. 100x),
            # we might want a 'sample_weight' support in DataLoader, but currently we rely on duplication.

            ds_path = os.path.dirname(dataset_info[ds]['zarrs'][0]['path']) # Heuristic to get dataset root
            # Verify ds_path matches the source root logic
            # modifying scan_datasets to store root path would be cleaner, but this works if zarrs are found.

            # Optimization: If scaling_factor is very close to integer, round it
            if remainder > 0.99:
                full_copies += 1
                remainder = 0.0
            if remainder < 0.01:
                remainder = 0.0

            entries_created = 0

            # Determine base name
            base_name = ds

            # Add full copies
            for i in range(full_copies):
                entry_name = f"{base_name}_copy{i}" if (full_copies > 1 or remainder > 0) else base_name
                config_datasets.append({
                    "name": entry_name,
                    "path": ds_path,
                    "use_trajectory_ratio": 1.0,
                    "norm_stats_domain_name": base_name,
                    "enabled": True
                })
                entries_created += 1

            # Add remainder
            if remainder > 0:
                entry_name = f"{base_name}_partial"
                # If we have copies, ensure distinct name
                if full_copies > 0:
                    entry_name = f"{base_name}_copy{full_copies}_partial"

                config_datasets.append({
                    "name": entry_name,
                    "path": ds_path,
                    "use_trajectory_ratio": float(f"{remainder:.4f}"), # rounding for clean json
                    "norm_stats_domain_name": base_name,
                    "enabled": True
                })
                entries_created += 1

            final_stats.append({
                'Group': group_name,
                'Dataset': ds,
                'Alpha': weight_info.get('alpha', '?'),
                'Orig_Eps': ds_orig_eps,
                'Orig_Steps': ds_orig_steps,
                'Orig_Files': 1, # Dataset level
                'Target_Steps_Alloc': ds_target_steps,
                'Scale': scaling_factor,
                'Final_Eps': ds_orig_eps * scaling_factor,
                'Final_Steps': ds_orig_steps * scaling_factor,
                'Final_Files': entries_created
            })

    return config_datasets, final_stats

def sort_stats_within_group_by_steps(final_stats):
    """Keep group order, and sort rows by Final_Steps in each group (descending)."""
    grouped_stats = defaultdict(list)
    group_order = []

    for item in final_stats:
        group_name = item['Group']
        if group_name not in grouped_stats:
            group_order.append(group_name)
        grouped_stats[group_name].append(item)

    sorted_stats = []
    for group_name in group_order:
        sorted_stats.extend(
            sorted(grouped_stats[group_name], key=lambda x: x['Final_Steps'], reverse=True)
        )

    return sorted_stats


def print_preview(final_stats, target_total_steps, output_stats_path=None):
    final_stats = sort_stats_within_group_by_steps(final_stats)
    HEADER = (
        f"{'Group':<15} | {'Dataset':<35} | {'OrgEps':>8} | {'Scale':>5} | {'FinFiles':>8} | "
        f"{'FinEps':>9} | {'FinSteps':>10} | {'%Eps':>6} | {'%Steps':>7}"
    )
    table_width = len(HEADER)
    stats_lines = []
    stats_lines.append("\n" + "=" * table_width)
    stats_lines.append(HEADER)
    stats_lines.append("-" * table_width)

    print("\n" + "=" * table_width)
    print(HEADER)
    print("-" * table_width)

    total_final_eps = sum(x['Final_Eps'] for x in final_stats)
    total_final_steps = sum(x['Final_Steps'] for x in final_stats)

    current_group = ""
    for s in final_stats:
        if s['Group'] != current_group:
            print("-" * table_width)
            stats_lines.append("-" * table_width)
            current_group = s['Group']

        share_eps = 100.0 * s['Final_Eps'] / total_final_eps if total_final_eps > 0 else 0
        share_steps = 100.0 * s['Final_Steps'] / total_final_steps if total_final_steps > 0 else 0
        line = (
            f"{s['Group']:<15} | {s['Dataset']:<35} | {s['Orig_Eps']:>8.00f} | {s['Scale']:>5.2f} | "
            f"{s['Final_Files']:>8} | {s['Final_Eps']:>9.00f} | {s['Final_Steps']:>10.00f} | "
            f"{share_eps:>5.1f}% | {share_steps:>6.1f}%"
        )
        print(line)
        stats_lines.append(line)

    print("-" * table_width)
    stats_lines.append("-" * table_width)
    footer = f"{'TOTAL':<53} | {total_final_eps:>9.00f} | {total_final_steps:>10.00f} steps"
    print(footer)
    stats_lines.append(footer)
    print("=" * table_width + "\n")
    stats_lines.append("=" * table_width + "\n")

    if output_stats_path:
        with open(output_stats_path, 'w') as f:
            f.write("\n".join(stats_lines))
        print(f"Saved statistics to: {output_stats_path}")

def save_data_config(config_datasets, output_path, description="Generated by mix_datasets.py"):
    out_data = {
        "datasets": config_datasets,
        "default_use_trajectory_ratio": 1.0,
        "description": description
    }
    with open(output_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved data config to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Mix datasets based on groups and temperature (step-based intra-group weighting and scaling)."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Root directory containing source datasets (subfolders).")
    parser.add_argument("--output_path", type=str, required=True, help="Path for the output data_config.json file.")
    parser.add_argument("--total_steps", type=int, default=None, help="Target total STEPS (frames) of the new dataset.")
    parser.add_argument(
        "--total_episodes",
        type=int,
        default=None,
        help="[Deprecated] Target total EPISODES; converted to steps using global avg steps/episode.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    input_dir = os.path.expanduser(args.input_dir)
    output_path = os.path.expanduser(args.output_path)

    print(f"Source Root: {input_dir}")
    print(f"Output Config: {output_path}")

    # 1. Scan datasets
    dataset_info = scan_datasets(input_dir, DATASET_GROUPS)
    total_orig_eps = sum(d['total_episodes'] for d in dataset_info.values())
    total_orig_steps = sum(d['total_steps'] for d in dataset_info.values())

    # 1.5 Calculate Weights (step-based intra-group temperature sampling)
    dataset_weights = {}
    for group_name, group_info in DATASET_GROUPS.items():
        datasets = group_info["datasets"]
        alpha = group_info["alpha"]
        valid_datasets = [d for d in datasets if d in dataset_info and dataset_info[d]['total_steps'] > 0]
        if not valid_datasets:
            continue

        group_denom = 0
        local_weights = {}
        for ds in valid_datasets:
            n_steps = dataset_info[ds]['total_steps']
            w = 1.0 if alpha == 0 else float(n_steps) ** alpha
            local_weights[ds] = w
            group_denom += w
        dataset_weights[group_name] = {
            'datasets': local_weights,
            'denom': group_denom,
            'valid_list': valid_datasets,
            'alpha': alpha,
        }

    # 2. Determine Target Total Steps
    if args.total_steps is not None and args.total_episodes is not None:
        print("[Info] Both --total_steps and --total_episodes are set; using --total_steps.")

    if args.total_steps is not None:
        target_total_steps = args.total_steps
        print(f"Target Total Steps (CLI): {target_total_steps}")
    elif args.total_episodes is not None:
        if total_orig_eps <= 0:
            raise ValueError("No valid episodes found; cannot convert --total_episodes to steps.")
        avg_steps_per_episode = total_orig_steps / total_orig_eps
        target_total_steps = int(round(args.total_episodes * avg_steps_per_episode))
        print(
            f"Target Total Steps (from --total_episodes={args.total_episodes}, "
            f"avg_steps_per_episode={avg_steps_per_episode:.2f}): {target_total_steps}"
        )
    else:
        target_total_steps = total_orig_steps
        print(f"Target Total Steps (Default): {target_total_steps}")

    # 3. Generate Config Entries
    config_datasets, final_stats = generate_config_entries(
        dataset_weights,
        dataset_info,
        target_total_steps,
        GROUP_PROPORTIONS,
    )

    # 4. Preview and Save Stats
    stats_path = os.path.splitext(output_path)[0] + "_stats.txt"
    print_preview(final_stats, target_total_steps, output_stats_path=stats_path)

    # 5. Save Config
    save_data_config(config_datasets, output_path)

if __name__ == "__main__":
    main()


