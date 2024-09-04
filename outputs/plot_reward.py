import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the base path containing the results folders
base_folder_path = r'E:\Eco-Light-bachelor\outputs'

# List of folders to include, excluding 'fixed'
folders = ['A2C_low_peak', 'DQN_low_peak', 'SAC_low_peak']  # Removed 'SAC_low_flat'

# Initialize a dictionary to store results for each folder
results = {}

# Iterate through each folder
for folder in folders:
    folder_path = os.path.join(base_folder_path, folder)
    data = []

    # Iterate through all CSV files in the current folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            # Extract run number from the file name
            run_number = int(file_name.split('run')[-1].split('.')[0])

            # Load the CSV file
            file_path = os.path.join(folder_path, file_name)

            # Check if the file is empty
            if os.stat(file_path).st_size == 0:
                print(f"Skipping empty file: {file_path}")
                continue

            # Try to read the CSV file
            try:
                df = pd.read_csv(file_path)
            except pd.errors.EmptyDataError:
                print(f"No columns to parse in file: {file_path}")
                continue

            # Sum up the 'Rewards' column for other methods
            total_waiting_time = df['reward'].sum()

            # Store the result along with the run number
            data.append((run_number, total_waiting_time))

    # Sort data by run number and convert to DataFrame
    data.sort(key=lambda x: x[0])
    df_result = pd.DataFrame(data, columns=['Run', 'Rewards'])

    # Update the run number to timestep values (each run corresponds to 3600 seconds)
    timestep_labels = {i: i * 3600 for i in range(1, 20)}

    df_result['Timesteps'] = df_result['Run'].map(timestep_labels)

    # Store results in the dictionary
    results[folder] = df_result

# Plot the results from all folders in the same figure
plt.figure(figsize=(12, 8))

for folder, df in results.items():
    plt.plot(df['Timesteps'], df['Rewards'], marker='o', label=folder)



# Generate x values based on timesteps used in the plot (e.g., using A2C_low_peak timesteps for x-axis consistency)
x_values = list(results['A2C_low_peak']['Timesteps'])  # Using A2C_low_peak timesteps for x-axis consistency



# Set Y-axis range (optional)
# plt.ylim(0, 200000)

#plt.title('Comparison of Rewards by Timesteps Across Different Methods')
plt.xlabel('Timesteps')
plt.ylabel('Rewards')
plt.legend()
plt.grid(True)
plt.show()
