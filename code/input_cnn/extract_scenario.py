# This script accesses to the zip folder of a specific scenario and extracts only the label files, the mmWave power files,
# and the csv file, ignoring the rest of the files in the zip folder (GPS location and camera files).

# In this way, we heavily save processing time and disk space, since the GPS and camera files are not needed in further processing steps.


import zipfile
from tqdm import tqdm


SCENARIOS = [17, 18, 19, 20, 21]


def is_valid_path(file_path):
    for valid_path in paths_to_keep:
        if file_path.startswith(valid_path):
            return True
    return False


if __name__ == "__main__":
    for scenario in SCENARIOS:
        zip_path = f"./scenario{scenario}.zip"

        extract_path = f"./scenario{scenario}/"

        paths_to_keep = [
            f"scenario{scenario}.csv",
            "unit1/mmWave_data/",
            "unit1/label_data/" if scenario in [17, 18, 19, 20] else "unit1/label/"
        ]

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                all_files = zip_ref.namelist()

                files_to_extract = [file for file in all_files if is_valid_path(file)]

                for file in tqdm(files_to_extract, desc="Extraction", unit="file"):
                    zip_ref.extract(file, extract_path)

                print(f"\nExtraction of Scenario {scenario} completed successfully.")
        except Exception as e:
            print(f"An error occurred: {e}")
