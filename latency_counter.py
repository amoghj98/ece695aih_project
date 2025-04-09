## computes cumulative latency from latency metrics

#!/usr/bin/env python3
import os
import glob
import re
import sys

def convert_time_to_seconds(time_str):
    """
    Convert a string with a time value and unit (e.g., "7.258s", "854.394ms")
    to seconds.
    """
    # Regular expression to capture the numerical part and the unit
    match = re.match(r"([0-9.]+)([a-zA-Z]+)", time_str)
    if not match:
        raise ValueError(f"Invalid time format: {time_str}")
    
    value, unit = match.groups()
    value = float(value)
    
    # Convert based on unit
    if unit == "s":
        return value
    elif unit == "ms":
        return value / 1000.0
    elif unit == "us":
        return value / 1e6
    elif unit == "ns":
        return value / 1e9
    else:
        raise ValueError(f"Unknown unit: {unit}")

def process_file(filepath):
    """
    Read a file and return the Self CUDA time (in seconds) found on the line
    that starts with "Self CUDA time total:".
    """
    total = 0.0
    with open(filepath, 'r') as f:
        for line in f:
            # We check if the line has the summary for Self CUDA time
            if line.startswith("Self CUDA time total:"):
                # Example expected line: "Self CUDA time total: 7.258s"
                parts = line.strip().split("Self CUDA time total:")
                if len(parts) > 1:
                    time_str = parts[1].strip()
                    try:
                        total = convert_time_to_seconds(time_str)
                    except ValueError as e:
                        print(f"Error in file {filepath}: {e}")
                break  # We found the summary line; no need to continue
    return total

def main(folder_path):
    # Create a pattern to locate all .csv files in the provided folder
    pattern = os.path.join(folder_path, "*.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print("No CSV files found in the specified folder.")
        return
    
    accumulated_total = 0.0
    for filepath in csv_files:
        file_total = process_file(filepath)
        print(f"File '{os.path.basename(filepath)}': Self CUDA time total = {file_total} s")
        accumulated_total += file_total

    print(f"\nAccumulated total Self CUDA time across files: {accumulated_total} s")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python latency_counter.py <folder_path>")
        sys.exit(1)

    folder = sys.argv[1]
    main(folder)
