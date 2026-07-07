# This script processes mmWave data for specified scenarios, caching the results into compressed binary files for efficient storage and retrieval.

import numpy as np
from pathlib import Path
from tqdm import tqdm

SCENARIOS = [17, 18, 19, 20, 21]

if __name__ == "__main__":
    for scenario in SCENARIOS:
        mmwave_dir = Path(f"./scenario{scenario}/unit1/mmWave_data")

        num_files = len(list(mmwave_dir.glob("mmWave_power_*.txt")))
        scenario_mmwaves = []

        scenario_mmwaves = []
        for idx in tqdm(range(num_files), desc=f"Caching mmWaves into RAM for Scenario {scenario}"):
            logical_idx = idx
            
            # We apply a correction to the scenario 20, because it has a gap in the mmWave data between the indices 44255 and 50837 (both included)
            if scenario == 20 and logical_idx >= 44255:
                logical_idx = logical_idx + 6583
            
            # We apply a correction to the scenario 21, because it has a different relationship between the labels and the mmWave files:
            # the mmWave files are shifted down by 1 position compared to the labels (i.e., the mmWave_power_0.txt file corresponds to
            # the label_1.txt file).
            file_idx = logical_idx + 1 if scenario in [17, 18, 19, 20] else logical_idx
            mmWave_file_path = f"./scenario{scenario}/unit1/mmWave_data/mmWave_power_{file_idx}.txt"
            
            power_floats = []
            with open(mmWave_file_path, "r") as f:
                power_floats = [float(line) for line in f.read().splitlines()]
                # power_cut = power_floats[:27] + power_floats[37:]
            
            scenario_mmwaves.append(power_floats)

        save_path = f"./scenario{scenario}/unit1/mmWave_data_unified.npz"
        np.savez_compressed(save_path, mmwaves=np.array(scenario_mmwaves))

        print(f"Unified mmWave data for Scenario {scenario}, saved to {save_path}.")
