# The code in this file is responsible for creating the input datasets (for Problem 1), elaborating all the scenarios from 17 to 21.
# The code creates and saves to disk separately the processed data for each of the 10 different Tps (1 to 10).
# The code also splits the data into train and test sets, with a 70-30 ratio, and saves them in the corresponding folders, for each Tp.
# The code is structured in three main phases:
#       1. Reading Phase:   in this phase, the code caches all the labels from the text files and all the mmWave data into RAM, for faster
#                           access during sample creation.
#       2. Sample Identification Phase: in this phase, the code identifies the indices of the positive and negative samples for all scenarios,
#                                       guaranteeing that all of them are linked to samples that are considered "valid" for all the Tps (we
#                                       ensure it by checking the validity of the samples for the Tp = 10: a sample which is valid for Tp = 10
#                                       is also valid for all other Tps); in this way, we will have training and test sets with the same
#                                       dimensions across all Tps).
#       3. Sample Creation Phase:   in this phase, the code uses the pre-identified indices to create the samples for all scenarios and each Tp,
#                                   also balancing the positive and negative samples, shuffling them, splitting them into train and test sets,
#                                   and finally saving them to disk in compressed .npz format.


import os
import numpy as np
from pathlib import Path
from tqdm import tqdm


SEED = 1234
SPLIT_RATIOS = [0.6, 0.2]

BASE_PATH = f"./scenarios17_21_clean_neg_samples_60_20_20"
SCENARIOS = [17, 18, 19, 20, 21]

TPS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

TO_WINDOW_WIDTH = 16
TP_WINDOW_WIDTH = 10


def find_sample_indices_by_scenario(labels, scenario, sample_type="pos"):
    indices = []
    for t in tqdm(range(TO_WINDOW_WIDTH - 1, len(labels) - TP_WINDOW_WIDTH), total=len(labels), desc=f"Finding {sample_type} sample indices for Scenario {scenario}"):
        # t is the index of the last time step in the observation window
        tp_10_obs_window = labels[(t + 1 - TO_WINDOW_WIDTH) : (t + 1)]
        tp_10_pred_window = labels[(t + 1) : (t + 1 + TP_WINDOW_WIDTH)]
        next_16_time_samples = labels[(t + 1) : (t + 1 + TO_WINDOW_WIDTH)]
        prev_sample = labels[t + 1 - TO_WINDOW_WIDTH - 1]

        if sample_type == "pos":
            if "1" not in tp_10_obs_window and "1" not in tp_10_pred_window[0:(TP_WINDOW_WIDTH-1)] and tp_10_pred_window[TP_WINDOW_WIDTH-1] == "1":
                # We append the index of the last time step in the prediction window with length = 10 (i.e., the index of
                # the first label "1" in the prediction window, that is the first time step in which the blockage occurs).
                indices.append(t + TP_WINDOW_WIDTH)
        else:
            if prev_sample == "1" and "1" not in tp_10_obs_window and "1" not in next_16_time_samples:
                # We append the index of the first time step in the observation window found with a prediction window long 10.
                indices.append((t + 1 - TO_WINDOW_WIDTH))
    return indices


def create_power_heatmap(start_idx, end_idx, scenario, cached_mmwaves):
    power_heatmap = []
    for idx in range(start_idx, end_idx):
        power_heatmap.append(cached_mmwaves[scenario][idx])
    return power_heatmap


def make_samples_by_tp_by_scenario(indices, tp, scenario, cached_mmwaves, sample_type="pos"):
    samples = []

    # The indices we receive here are the indices of the last time step in the prediction window
    # (i.e., the index of the first "1" in the prediction window for the positive samples)
    for t in tqdm(range(len(indices)), desc=f"Processing {sample_type} samples for Scenario {scenario}"):
        start_obs_idx = None
        end_obs_idx = None

        if sample_type == "pos":
            # For the pos samples, every value in "indices" is the index of the last time step in the prediction window (i.e.,
            # the index of the first label "1" in the prediction window, that is the first time step in which the blockage occurs).
            start_obs_idx = (indices[t] + 1) - tp - TO_WINDOW_WIDTH
            end_obs_idx = (indices[t] + 1) - tp
        else:
            # For the neg samples, every value in "indices" is the index of the first time step in the observation window of a neg
            # sample valid for Tp = 10, so also for all the other Tps.
            start_obs_idx = indices[t]
            end_obs_idx = indices[t] + TO_WINDOW_WIDTH

        samples.append(create_power_heatmap(start_obs_idx, end_obs_idx, scenario, cached_mmwaves))
    return samples


def prune_samples(samples_list, target_count):
    indices = np.random.choice(len(samples_list), size=target_count, replace=False)
    return [samples_list[i] for i in indices]


if __name__ == "__main__":
    os.makedirs(BASE_PATH, exist_ok=True)
    os.makedirs(os.path.join(BASE_PATH, "train"), exist_ok=True)
    os.makedirs(os.path.join(BASE_PATH, "val"), exist_ok=True)
    os.makedirs(os.path.join(BASE_PATH, "test"), exist_ok=True)

    print("--- READING PHASE ---")
    all_scenario_labels = {}
    all_cached_mmwaves = {scenario: {} for scenario in SCENARIOS}

    for scenario in SCENARIOS:
        label_dir = Path(f"./scenario{scenario}/unit1/label_data") if scenario in [17, 18, 19, 20] else Path(f"./scenario{scenario}/unit1/label")
        num_files = len(list(label_dir.glob("label_*.txt")))

        scenario_labels = []
        for i in tqdm(range(1, num_files + 1), desc=f"Processing labels for Scenario {scenario}"):
            logical_i = i

            # We apply a correction to the scenario 20, because it has a gap in the label data between the files label_44256.txt and label_50838.txt (both included).
            if scenario == 20 and logical_i > 44255:
                logical_i = logical_i + 6583

            file_path = label_dir / f"label_{logical_i}.txt"
            if file_path.exists():
                with open(file_path, "r") as f:
                    line = f.readline()
                    scenario_labels.append(line.strip())
        all_scenario_labels[scenario] = scenario_labels

        cached_file_path = f"./scenario{scenario}/unit1/mmWave_data_unified.npz"
        all_cached_mmwaves[scenario] = np.load(cached_file_path)["mmwaves"]
    print("---------------------------------------")

    print("\n\n--- SAMPLE IDENTIFICATION PHASE ---")
    pos_samples_indices = {scenario: [] for scenario in SCENARIOS}
    neg_samples_indices = {scenario: [] for scenario in SCENARIOS}
    for scenario in SCENARIOS:
        scenario_labels = all_scenario_labels[scenario]
        pos_samples_indices[scenario] = find_sample_indices_by_scenario(scenario_labels, scenario, sample_type="pos")
        neg_samples_indices[scenario] = find_sample_indices_by_scenario(scenario_labels, scenario, sample_type="neg")

        # Saving indices to disk in a .txt file
        with open(os.path.join(BASE_PATH, f"scenario{scenario}_pos_indices.txt"), "w") as f:
            for idx in pos_samples_indices[scenario]:
                f.write(f"{idx}\n")
        with open(os.path.join(BASE_PATH, f"scenario{scenario}_neg_indices.txt"), "w") as f:
            for idx in neg_samples_indices[scenario]:
                f.write(f"{idx}\n")
    print("---------------------------------------")

    print("\n\n--- SAMPLE CREATION PHASE ---")
    lengths = []
    for tp in TPS:
        print(f"Processing samples for Tp={tp}...")
        samples = []
        labels = []

        # Creating samples
        for scenario in SCENARIOS:
            pos_samples = make_samples_by_tp_by_scenario(pos_samples_indices[scenario], tp, scenario, all_cached_mmwaves, sample_type="pos")
            neg_samples = make_samples_by_tp_by_scenario(neg_samples_indices[scenario], tp, scenario, all_cached_mmwaves, sample_type="neg")

            print(f"Scenario {scenario} - Tp={tp}: found {len(pos_samples)} pos samples and {len(neg_samples)} neg samples before balancing.")
            np.random.seed(SEED)    # We set the seed before pruning to ensure the same indices are selected across all scenarios and Tps.
            if len(neg_samples) > len(pos_samples):
                print(" --> pruning neg samples...")
                neg_samples = prune_samples(neg_samples, len(pos_samples))
            else:
                print(" --> pruning pos samples...")
                pos_samples = prune_samples(pos_samples, len(neg_samples))
            print(f"Scenario {scenario} - Tp={tp}: balanced to {len(pos_samples)} pos and {len(neg_samples)} neg (total: {len(pos_samples) + len(neg_samples)}).\n")
            
            samples.extend(pos_samples)
            samples.extend(neg_samples)
            labels.extend([1] * len(pos_samples))
            labels.extend([0] * len(neg_samples))
        
        samples_np = np.array(samples)
        labels_np = np.array(labels)
        lengths.append(len(samples_np))

        # Shuffling samples
        np.random.seed(SEED)    # We set the seed before pruning to ensure the same ordering is done across all scenarios and Tps.
        indices = np.random.permutation(len(samples_np))
        samples_np = samples_np[indices]
        labels_np = labels_np[indices]

        # Splitting samples
        train_end_idx = int(SPLIT_RATIOS[0] * len(samples_np))
        val_end_idx = int((SPLIT_RATIOS[0] + SPLIT_RATIOS[1]) * len(samples_np))
        train_samples = samples_np[:train_end_idx]
        train_labels = labels_np[:train_end_idx]
        val_samples = samples_np[train_end_idx:val_end_idx]
        val_labels = labels_np[train_end_idx:val_end_idx]
        test_samples = samples_np[val_end_idx:]
        test_labels = labels_np[val_end_idx:]

        # Saving samples
        train_save_path = os.path.join(BASE_PATH, "train", f"train_Tp{tp}.npz")
        val_save_path = os.path.join(BASE_PATH, "val", f"val_Tp{tp}.npz")
        test_save_path = os.path.join(BASE_PATH, "test", f"test_Tp{tp}.npz")
        np.savez_compressed(train_save_path, features=train_samples, labels=train_labels)
        np.savez_compressed(val_save_path, features=val_samples, labels=val_labels)
        np.savez_compressed(test_save_path, features=test_samples, labels=test_labels)

        print(f" --> Saved train, val, and test sets for Tp={tp}.\n     Train shape: {train_samples.shape}\n     Val shape: {val_samples.shape}\n     Test shape: {test_samples.shape}\n\n")
    print("---------------------------------------")
