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
avg_values = []

# Create an SMBus instance (e.g., bus number 1 for Raspberry Pi)
bus = smbus.SMBus(1)

# TLA2024 I2C address (modify if different)
tla2024_address = 0x48

# MCP4018 I2C address updated to 0x2F as per your request
mcp4018_address = 0x2F

# Initialize wiper position
wiper_position = 32

def set_wiper_position(bus, address, position):
    # Ensure position is within 0-127
    position = max(0, min(position, 127))
    # Send the position byte to the MCP4018
    try:
        bus.write_byte(address, position)
        print(f"Set wiper position to {position}")
    except Exception as e:
        print(f"Failed to set wiper position: {e}")

# Set initial wiper position
set_wiper_position(bus, mcp4018_address, wiper_position)

def read_adc_channel(channel, tla2024_address, bus):
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
    bus.write_i2c_block_data(tla2024_address, 0x01, [config_MSB, config_LSB])

    # Wait until conversion is complete by checking the OS bit
    while True:
        config_status = bus.read_i2c_block_data(tla2024_address, 0x01, 2)
        status = (config_status[0] << 8) | config_status[1]
        if status & (1 << 15):  # Check if OS bit is set (conversion complete)
            break
        time.sleep(0.001)  # Wait 1ms before checking again

    # Read the conversion result from the Conversion Register (register 0x00)
    data = bus.read_i2c_block_data(tla2024_address, 0x00, 2)
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

# Threshold voltages
LOW_THRESHOLD = 1.50  # Voltage below which we increment the wiper position
HIGH_THRESHOLD = 1.75  # Voltage above which we decrement the wiper position
CONSECUTIVE_COUNT = 5  # Number of consecutive readings required to adjust the wiper

# Initialize counters
above_threshold_counter = 0
below_threshold_counter = 0

# Function to update the plot
def update(frame):
    global wiper_position
    global above_threshold_counter, below_threshold_counter

    # Generate current datetime
    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Read ADC readings for each channel
    index = len(indexes) + 1 if len(indexes) == 0 else indexes[-1] + 1
    try:
        q1 = read_adc_channel(0, tla2024_address, bus)
        q2 = read_adc_channel(1, tla2024_address, bus)
        q3 = read_adc_channel(2, tla2024_address, bus)
        q4 = read_adc_channel(3, tla2024_address, bus)
    except Exception as e:
        print(f"Error reading ADC channels: {e}")
        q1 = q2 = q3 = q4 = None

    # Calculate the average of Q1, Q2, Q3, and Q4
    voltages = [v for v in [q1, q2, q3, q4] if v is not None]
    if voltages:
        avg_voltage = sum(voltages) / len(voltages)
    else:
        avg_voltage = None

    # Adjust wiper position based on the average voltage and counters
    if avg_voltage is not None:
        if avg_voltage < LOW_THRESHOLD:
            below_threshold_counter += 1
            above_threshold_counter = 0
            print(f"Average voltage {avg_voltage:.2f}V < {LOW_THRESHOLD}V: below_threshold_counter = {below_threshold_counter}")
        elif avg_voltage > HIGH_THRESHOLD:
            above_threshold_counter += 1
            below_threshold_counter = 0
            print(f"Average voltage {avg_voltage:.2f}V > {HIGH_THRESHOLD}V: above_threshold_counter = {above_threshold_counter}")
        else:
            above_threshold_counter = 0
            below_threshold_counter = 0
            print(f"Average voltage {avg_voltage:.2f}V within thresholds.")

        # Check if the counters have reached the required consecutive count
        if below_threshold_counter >= CONSECUTIVE_COUNT:
            wiper_position += 1
            below_threshold_counter = 0  # Reset counter after adjustment
            print(f"Average voltage consistently below {LOW_THRESHOLD}V: Incrementing wiper position to {wiper_position}")
            # Ensure wiper_position is within 0-127
            wiper_position = max(0, min(wiper_position, 127))
            set_wiper_position(bus, mcp4018_address, wiper_position)
        elif above_threshold_counter >= CONSECUTIVE_COUNT:
            wiper_position -= 1
            above_threshold_counter = 0  # Reset counter after adjustment
            print(f"Average voltage consistently above {HIGH_THRESHOLD}V: Decrementing wiper position to {wiper_position}")
            # Ensure wiper_position is within 0-127
            wiper_position = max(0, min(wiper_position, 127))
            set_wiper_position(bus, mcp4018_address, wiper_position)
    else:
        print("Average voltage is None, skipping wiper adjustment")

    # Debugging printouts
    print(f"Index: {index}, Time: {current_datetime}, Q1: {q1}, Q2: {q2}, Q3: {q3}, Q4: {q4}, Average: {avg_voltage}")

    # Append new data to lists
    indexes.append(index)
    q1_values.append(q1)
    q2_values.append(q2)
    q3_values.append(q3)
    q4_values.append(q4)
    avg_values.append(avg_voltage)

    # Append data to CSV file
    with open(csv_filename, mode='a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([index, current_datetime, q1, q2, q3, q4, avg_voltage])

    # Keep only the last NUM_READINGS points
    if len(indexes) > NUM_READINGS:
        indexes.pop(0)
        q1_values.pop(0)
        q2_values.pop(0)
        q3_values.pop(0)
        q4_values.pop(0)
        avg_values.pop(0)

    # Clear previous plots
    plt.cla()

    # Plot the updated data
    plt.plot(indexes, q1_values, label='Q1')
    plt.plot(indexes, q2_values, label='Q2')
    plt.plot(indexes, q3_values, label='Q3')
    plt.plot(indexes, q4_values, label='Q4')
    plt.plot(indexes, avg_values, label='Average', linestyle='--', color='black')

    plt.ylim(0, 3.3)

    # Add labels and legend
    plt.xlabel('Index')
    plt.ylabel('ADC Reading (V)')
    plt.title('Real-Time ADC Readings (Last 100 Readings)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)

# Create a CSV file and write the header
with open(csv_filename, mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(['Index', 'Datetime', 'Q1', 'Q2', 'Q3', 'Q4', 'Average'])

# Set up the figure and axis for the plot
plt.figure(figsize=(12, 6))

# Use FuncAnimation to update the plot in real-time every 100 ms
ani = FuncAnimation(plt.gcf(), update, interval=100, cache_frame_data=False)

# Adjust layout to make space for the legend
plt.subplots_adjust(bottom=0.25)

# Show the plot
plt.show()
