# calculates cumulative timestamp time

import os

# Define the folder containing the CSV files.
data_folder = 'data_50_beamSearch_llama'

# Get all file names that match the basic naming convention.
csv_files = [f for f in os.listdir(data_folder) if f.startswith('forward_pass_') and f.endswith('.csv')]

timestamps = []

# Extract the timestamp from the file names.
for filename in csv_files:
    try:
        # Remove the '.csv' extension and split on the last underscore.
        base_name = filename[:-4]
        # The timestamp is expected to be the part after the last underscore.
        ts_str = base_name.rsplit('_', 1)[-1]
        ts = float(ts_str)
        timestamps.append(ts)
    except ValueError:
        print(f"Warning: Could not parse a timestamp from {filename}")

# Ensure there are at least two timestamps to calculate a difference.
if len(timestamps) < 2:
    print("Not enough timestamped files to calculate a cumulative time difference.")
else:
    # Sort the timestamps in ascending order.
    timestamps.sort()
    
    # Calculate the cumulative time difference.
    cumulative_time = sum(t2 - t1 for t1, t2 in zip(timestamps, timestamps[1:]))
    print(f"Cumulative time between forward passes (from timestamps): {cumulative_time:.3f} seconds")
