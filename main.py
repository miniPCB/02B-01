import os
import platform
import smbus
import time

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

# Create an SMBus instance (e.g., bus number 1 for Raspberry Pi)
bus = smbus.SMBus(1)

# TLA2024 I2C address (modify if different)
address = 0x48

def read_adc_channel(channel, address, bus):
    """
    Reads the voltage from a specified channel on the TLA2024 ADC.

    Parameters:
    - channel (int): ADC channel to read (0-3).
    - address (int): I2C address of the TLA2024 device.
    - bus (smbus.SMBus): SMBus instance for I2C communication.

    Returns:
    - float: Measured voltage on the specified channel.
    """
    # Validate the channel number
    if channel < 0 or channel > 3:
        raise ValueError('Invalid channel: must be 0-3')

    # Map the channel to the MUX configuration bits for single-ended input
    mux = 0b100 + channel  # Channels 0-3 correspond to MUX settings 100-111

    # Configuration Register settings
    # OS = 1 (start a single conversion)
    # MUX[2:0] = mux (select the input channel)
    # PGA[2:0] = 0b010 (gain ±2.048V)
    # MODE = 1 (single-shot mode)
    config_upper = (1 << 15) | (mux << 12) | (0b010 << 9) | (1 << 8)

    # DR[2:0] = 0b100 (1600 samples per second)
    # COMP_MODE = 0 (traditional comparator)
    # COMP_POL = 0 (active low)
    # COMP_LAT = 0 (non-latching)
    # COMP_QUE[1:0] = 0b11 (disable comparator)
    config_lower = (0b100 << 5) | 0x03  # Last two bits set COMP_QUE[1:0] = 0b11

    # Combine upper and lower bytes of the configuration
    config = config_upper | config_lower

    # Split the 16-bit configuration into two 8-bit bytes
    config_MSB = (config >> 8) & 0xFF
    config_LSB = config & 0xFF

    # Write configuration to the ADC's Configuration Register (register 0x01)
    bus.write_i2c_block_data(address, 0x01, [config_MSB, config_LSB])

    # Wait for the conversion to complete (conversion time depends on data rate)
    time.sleep(0.001)  # Wait 1ms for conversion (safe for 1600SPS data rate)

    # Read the conversion result from the Conversion Register (register 0x00)
    data = bus.read_i2c_block_data(address, 0x00, 2)
    result = (data[0] << 8) | data[1]

    # Right-shift to align the 12-bit result (TLA2024 outputs data in bits [15:4])
    raw_adc = result >> 4

    # Convert to signed 12-bit integer
    if raw_adc > 0x7FF:
        raw_adc -= 0x1000

    # Calculate the voltage based on the ADC's full-scale range (±2.048V)
    voltage = raw_adc * 0.001  # Each LSB represents 1mV

    return voltage



# Function to update the plot
def update(frame):
    # Generate 1 new reading in each update (10 updates per second)
    # Generate current datetime
    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Generate random ADC readings for each quadrant
    index = len(indexes) + 1 if len(indexes) == 0 else indexes[-1] + 1
    q1 = read_adc_channel(0, address, bus)
    q2 = read_adc_channel(1, address, bus)
    q3 = read_adc_channel(2, address, bus)
    q4 = read_adc_channel(3, address, bus)

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
