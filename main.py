import os
import platform

# Determine the platform and set the correct file path
current_platform = platform.system()

if current_platform == "Windows":
    csv_filename = r'C:\Repos\02B-01\adc.csv'
elif current_platform == "Linux":
    csv_filename = '/home/pi/02B-01/adc.csv'
else:
    raise OSError(f"Unsupported platform: {current_platform}")

# Ensure the directory exists before trying to open the file
csv_dir = os.path.dirname(csv_filename)
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

# Now proceed with the file operations
try:
    with open(csv_filename, mode='w', newline='') as file:
        # Your code for writing to the CSV file
        pass  # Replace with your code to generate the CSV content
except Exception as e:
    print(f"An error occurred: {e}")


import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import pandas as pd
    print("Pandas is already installed.")
except ImportError:
    print("Pandas is not installed. Installing now...")
    install("pandas")
    try:
        import pandas as pd
        print("Pandas has been successfully installed.")
    except ImportError:
        print("Failed to install Pandas.")

try:
    import matplotlib
    print("Matplotlib is already installed.")
except ImportError:
    print("Matplotlib is not installed. Installing now...")
    install("matplotlib")
    try:
        import matplotlib
        print("Matplotlib has been successfully installed.")
    except ImportError:
        print("Failed to install Matplotlib.")

import csv
import random
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the range for ADC values centered around 2.500 with deviations up to 1.000
ADC_CENTER = 2.500
ADC_DEVIATION = 1.000

# Number of simulated readings to display
NUM_READINGS = 100

# Initialize lists to store data for plotting (only keep last NUM_READINGS points)
indexes = []
q1_values = []
q2_values = []
q3_values = []
q4_values = []

# Function to generate a random ADC value centered around 2.500 with deviation up to 1.000
def generate_random_adc_value():
    return round(random.uniform(ADC_CENTER - ADC_DEVIATION, ADC_CENTER + ADC_DEVIATION), 3)

# Function to update the plot
def update(frame):
    # Generate 1 new reading in each update (10 updates per second)
    # Generate current datetime
    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Generate random ADC readings for each quadrant
    index = len(indexes) + 1 if len(indexes) == 0 else indexes[-1] + 1
    q1 = generate_random_adc_value()
    q2 = generate_random_adc_value()
    q3 = generate_random_adc_value()
    q4 = generate_random_adc_value()

    # Append new data to lists
    indexes.append(index)
    q1_values.append(q1)
    q2_values.append(q2)
    q3_values.append(q3)
    q4_values.append(q4)

    # Append data to CSV file
    with open(csv_filename, mode='a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([index, current_datetime, q1, q2, q3, q4])

    # Keep only the last NUM_READINGS points (in place modification)
    if len(indexes) > NUM_READINGS:
        del indexes[:-NUM_READINGS]
        del q1_values[:-NUM_READINGS]
        del q2_values[:-NUM_READINGS]
        del q3_values[:-NUM_READINGS]
        del q4_values[:-NUM_READINGS]

    # Clear previous plots
    plt.cla()

    # Plot the updated data
    plt.plot(indexes, q1_values, label='Q1')
    plt.plot(indexes, q2_values, label='Q2')
    plt.plot(indexes, q3_values, label='Q3')
    plt.plot(indexes, q4_values, label='Q4')

    # Add labels and legend
    plt.xlabel('Index')
    plt.ylabel('ADC Reading')
    plt.title('Real-Time Simulated ADC Readings (Last 100 Readings)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)  # Fixed legend below the chart

# Create a CSV file and write the header
with open(csv_filename, mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(['Index', 'Datetime', 'Q1', 'Q2', 'Q3', 'Q4'])

# Set up the figure and axis for the plot with increased width
plt.figure(figsize=(12, 6))  # Increased width from 10 to 12 inches

# Use FuncAnimation to update the plot in real-time every 100 ms (10 updates per second)
ani = FuncAnimation(plt.gcf(), update, interval=100, cache_frame_data=False)  # Update every 100 ms

# Adjust layout to make space for the legend
plt.subplots_adjust(bottom=0.2)

# Show the plot
plt.show()
