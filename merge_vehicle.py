import json
import glob
import os
import re

def merge_yearly_data_with_years(input_pattern, output_path):
    final_data = []
    total_files = 0
    total_entries = 0

    # Find all files matching data/vehicles_processed_{year}.json
    file_list = glob.glob(input_pattern)

    # Sort files to keep output somewhat organized
    file_list.sort()

    for file_path in file_list:
        # Extract the year from the filename using regex (e.g., _2025.json -> 2025)
        year_match = re.search(r'_(\d{4})\.json$', file_path)
        file_year = year_match.group(1) if year_match else "Unknown"

        print(f"📦 Processing Year {file_year} from: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                vehicles = json.load(f)
                total_files += 1
                for v in vehicles:
                    # Inject the year into the record
                    v["year"] = file_year
                    final_data.append(v)
                    total_entries += 1

            except (json.JSONDecodeError, KeyError) as e:
                print(f"  ⚠️ Error parsing {file_path}: {e}")

    # Save the complete list to a single master file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=4)

    print("\n" + "="*40)
    print(f"📊 FULL MERGE REPORT")
    print(f"Years Merged:       {total_files}")
    print(f"Total Records:      {total_entries}")
    print(f"Saved to:           {output_path}")
    print("="*40)

if __name__ == "__main__":
    # Path configuration
    INPUT_PATTERN = "data/vehicles_processed_*.json"
    OUTPUT_FILE = "data/master_vehicles_database.json"

    merge_yearly_data_with_years(INPUT_PATTERN, OUTPUT_FILE)