# This script checks the correspondence between label files and mmWave power files in each scenario by accessing the single label_k.txt
# and mmWave_power_k.txt files in order (directory by directory, i.e. scenario by scenario):
#   - if a label file does not have a corresponding mmWave power file, we delete the label file;
#   - if a mmWave power file does not have a corresponding label file, we delete the mmWave power file.

# After the first round of deletion, we check again if there are still label files without corresponding mmWave power files and vice versa,
# and we report the number of such files before and after deletion. The goal is to ensure a perfect correspondence between label files and
# mmWave power files, which is crucial for the integrity of the dataset (and is strictly required by the script that processes the raw
# dataset to create the positive and the negative samples).


from pathlib import Path
from tqdm import tqdm


SCENARIOS = [17, 18, 19, 20, 21]


if __name__ == "__main__":
    counters = []
    for scenario in SCENARIOS:
        print(f"Processing labels of Scenario {scenario}...")
        counter = 0

        label_dir = Path(f"./scenario{scenario}/unit1/label_data") if scenario in [17, 18, 19, 20] else Path(f"./scenario{scenario}/unit1/label")
        num_files = len(list(label_dir.glob("label_*.txt")))

        for i in tqdm(range(1, num_files + 1), desc=f"Processing label files of Scenario {scenario}"):
            file_path = label_dir / f"label_{i}.txt"
            if file_path.exists():
                power_dir = Path(f"./scenario{scenario}/unit1/mmWave_data")
                power_file_path = power_dir / f"mmWave_power_{i}.txt" if scenario in [17, 18, 19, 20] else power_dir / f"mmWave_power_{i-1}.txt"
                if not power_file_path.exists():
                    counter += 1

                    # delete the label file
                    file_path.unlink()
        
        counters.append(counter)

    # check again if all label files have a corresponding mmWave power file
    counters_after_deletion = []
    for scenario in SCENARIOS:
        counter_after_deletion = 0
        print(f"Re-checking labels of Scenario {scenario}...")
        label_dir = Path(f"./scenario{scenario}/unit1/label_data") if scenario in [17, 18, 19, 20] else Path(f"./scenario{scenario}/unit1/label")
        num_files = len(list(label_dir.glob("label_*.txt")))

        for i in tqdm(range(1, num_files + 1), desc=f"Re-checking label files of Scenario {scenario}"):
            file_path = label_dir / f"label_{i}.txt"
            if file_path.exists():
                power_dir = Path(f"./scenario{scenario}/unit1/mmWave_data")
                power_file_path = power_dir / f"mmWave_power_{i}.txt" if scenario in [17, 18, 19, 20] else power_dir / f"mmWave_power_{i-1}.txt"
                if not power_file_path.exists():
                    counter_after_deletion = counter_after_deletion + 1

        counters_after_deletion.append(counter_after_deletion)

    print(f"counters: {counters}")
    print(f"counters after deletion: {counters_after_deletion}")


    # now, we do the same procedure in the opposite "direction"
    power_counters = []
    for scenario in SCENARIOS:
        print(f"Processing powers of Scenario {scenario}...")
        power_counter = 0

        power_dir = Path(f"./scenario{scenario}/unit1/mmWave_data")
        power_num_files = len(list(power_dir.glob("mmWave_power_*.txt")))

        for i in tqdm(range(1, power_num_files + 1 + 7000), desc=f"Processing power files of Scenario {scenario}"):
            power_file_path = power_dir / f"mmWave_power_{i}.txt"
            if power_file_path.exists():
                label_dir = Path(f"./scenario{scenario}/unit1/label_data") if scenario in [17, 18, 19, 20] else Path(f"./scenario{scenario}/unit1/label")
                label_file_path = label_dir / f"label_{i}.txt" if scenario in [17, 18, 19, 20] else label_dir / f"label_{i+1}.txt"
                if not label_file_path.exists():
                    power_counter += 1

                    # delete the power file
                    power_file_path.unlink()
        
        power_counters.append(power_counter)
    
    # check again if all mmWave power files have a corresponding label file
    power_counters_after_deletion = []
    for scenario in SCENARIOS:
        print(f"Re-checking powers of Scenario {scenario}...")
        power_counter_after_deletion = 0

        power_dir = Path(f"./scenario{scenario}/unit1/mmWave_data")
        power_num_files = len(list(power_dir.glob("mmWave_power_*.txt")))

        for i in tqdm(range(1, power_num_files + 1), desc=f"Re-checking power files of Scenario {scenario}"):
            power_file_path = power_dir / f"mmWave_power_{i}.txt"
            if power_file_path.exists():
                label_dir = Path(f"./scenario{scenario}/unit1/label_data") if scenario in [17, 18, 19, 20] else Path(f"./scenario{scenario}/unit1/label")
                label_file_path = label_dir / f"label_{i}.txt" if scenario in [17, 18, 19, 20] else label_dir / f"label_{i+1}.txt"
                if not label_file_path.exists():
                    power_counter_after_deletion += 1
        
        power_counters_after_deletion.append(power_counter_after_deletion)
    
    print(f"power_counters: {power_counters}")
    print(f"power_counters after deletion: {power_counters_after_deletion}")
