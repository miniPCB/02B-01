import os
import platform
import time
import csv
from datetime import datetime
import subprocess
import sys

# Function to install missing packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check and import smbus
try:
    import smbus
    print("smbus is already installed.")
except ImportError:
    print("smbus is not installed. Installing now...")
    install("smbus")
    try:
        import smbus
        print("smbus has been successfully installed.")
    except ImportError:
        print("Failed to install smbus.")

# Check and import matplotlib
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    print("matplotlib is already installed.")
except ImportError:
    print("matplotlib is not installed. Installing now...")
    install("matplotlib")
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        print("matplotlib has been successfully installed.")
    except ImportError:
        print("Failed to install matplotlib.")

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

import smbus
import time
import csv
from datetime import datetime

# Number of readings to display
NUM_READINGS = 100

# Initialize lists to store data for plotting (only keep last NUM_READINGS points)
indexes = []
q1_values = []
q2_values = []
q3_values = []
q4_values = []

# Create an SMBus instance (e.g., bus number 1 for Raspberry Pi)
bus = smbus.SMBus(1)

# TLA2024 I2C address (modify if different)
address = 0x48

def read_adc_channel(channel, address, bus):
    # Validate the channel number
    if channel < 0 or channel > 3:
        raise ValueError('Invalid channel: must be 0-3')

    # Map the channel to the MUX configuration bits for single-ended input
    mux = 0b100 + channel  # Channels 0-3 correspond to MUX settings 100-111

    # Start with a configuration value of 0
    config = 0x0000

    # Set the OS bit (bit 15) to 1 to start a single conversion
    config |= (1 << 15)

    # Set MUX bits [14:12] to select the channel
    config |= (mux & 0x7) << 12

    # Set PGA bits [11:9] to 0b010 for ±2.048V range
    config |= (0b010 & 0x7) << 9

    # Set MODE bit (bit 8) to 1 for single-shot mode
    config |= (1 << 8)

    # Set DR bits [7:5] to 0b100 for 1600 samples per second
    config |= (0b100 & 0x7) << 5

    # Disable the comparator by setting COMP_MODE, COMP_POL, COMP_LAT to 0 and COMP_QUE[1:0] to 0b11
    config |= (0x03)  # COMP_QUE[1:0] bits

    # Debugging: Print the configuration register value
    print(f"Channel {channel}: Configuration Register: 0x{config:04X}")

    # Split the 16-bit configuration into two 8-bit bytes
    config_MSB = (config >> 8) & 0xFF
    config_LSB = config & 0xFF

    # Write configuration to the ADC's Configuration Register (register 0x01)
    bus.write_i2c_block_data(address, 0x01, [config_MSB, config_LSB])

    # Wait until conversion is complete by checking the OS bit
    while True:
        config_status = bus.read_i2c_block_data(address, 0x01, 2)
        status = (config_status[0] << 8) | config_status[1]
        if status & (1 << 15):  # Check if OS bit is set (conversion complete)
            break
        time.sleep(0.001)  # Wait 1ms before checking again

    # Read the conversion result from the Conversion Register (register 0x00)
    data = bus.read_i2c_block_data(address, 0x00, 2)
    result = (data[0] << 8) | data[1]

    # Debugging output
    print(f"Channel {channel}: Raw I2C Data Bytes: [{hex(data[0])}, {hex(data[1])}]")
    print(f"Channel {channel}: Combined Result (before shift): {hex(result)}")

    # Right-shift to align the 12-bit result (TLA2024 outputs data in bits [15:4])
    raw_adc = result >> 4

    print(f"Channel {channel}: Raw ADC Value (after shift): {raw_adc}")

    # Convert to signed 12-bit integer (if necessary)
    if raw_adc > 0x7FF:
        raw_adc -= 0x1000
        print(f"Channel {channel}: Adjusted Raw ADC Value (signed): {raw_adc}")

    # Calculate the voltage based on the ADC's full-scale range (±2.048V)
    voltage = raw_adc * 0.001  # LSB size is 1 mV

    print(f"Channel {channel}: Voltage: {voltage} V\n")

    return voltage

# Function to update the plot
def update(frame):
    # Generate current datetime
    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Read ADC readings for each channel
    index = len(indexes) + 1 if len(indexes) == 0 else indexes[-1] + 1
    try:
        q1 = read_adc_channel(0, address, bus)
        q2 = read_adc_channel(1, address, bus)
        q3 = read_adc_channel(2, address, bus)
        q4 = read_adc_channel(3, address, bus)
    except Exception as e:
        print(f"Error reading ADC channels: {e}")
        q1 = q2 = q3 = q4 = None

    # Debugging printouts
    print(f"Index: {index}, Time: {current_datetime}, Q1: {q1}, Q2: {q2}, Q3: {q3}, Q4: {q4}")

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

    # Keep only the last NUM_READINGS points
    if len(indexes) > NUM_READINGS:
        indexes.pop(0)
        q1_values.pop(0)
        q2_values.pop(0)
        q3_values.pop(0)
        q4_values.pop(0)

    # Clear previous plots
    plt.cla()

    # Plot the updated data
    plt.plot(indexes, q1_values, label='Q1')
    plt.plot(indexes, q2_values, label='Q2')
    plt.plot(indexes, q3_values, label='Q3')
    plt.plot(indexes, q4_values, label='Q4')

    # Add labels and legend
    plt.xlabel('Index')
    plt.ylabel('ADC Reading (V)')
    plt.title('Real-Time ADC Readings (Last 100 Readings)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)

# Create a CSV file and write the header
with open(csv_filename, mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(['Index', 'Datetime', 'Q1', 'Q2', 'Q3', 'Q4'])

# Set up the figure and axis for the plot
plt.figure(figsize=(12, 6))

# Use FuncAnimation to update the plot in real-time every 100 ms
ani = FuncAnimation(plt.gcf(), update, interval=100, cache_frame_data=False)

# Adjust layout to make space for the legend
plt.subplots_adjust(bottom=0.2)

# Show the plot
plt.show()
