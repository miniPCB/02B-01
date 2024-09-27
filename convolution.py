import os
import platform
import time
import csv
from datetime import datetime
import subprocess
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

# Adjust NUM_READINGS to be at least 2 to allow convolution
NUM_READINGS = 100
NUM_READINGS = max(NUM_READINGS, 2)

# Initialize lists to store data for plotting (only keep last NUM_READINGS points)
indexes = []

# Raw data lists
q1_raw_values = []
q2_raw_values = []
q3_raw_values = []
q4_raw_values = []
avg_raw_values = []

# Filtered data lists
q1_values = []
q2_values = []
q3_values = []
q4_values = []
avg_values = []

# Initialize list to store wiper positions
wiper_positions = []

# Initialize list to store convolution results
convolution_results = []

# Create an SMBus instance (e.g., bus number 1 for Raspberry Pi)
bus = smbus.SMBus(1)

# TLA2024 I2C address (modify if different)
tla2024_address = 0x48

# MCP4018 I2C address updated to 0x2F as per your request
mcp4018_address = 0x2F

# Initialize wiper position
wiper_position = 100

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

    # Set DR bits [7:5] to 0b110 for 3300 samples per second
    config |= (0b110 & 0x7) << 5

    # Disable the comparator by setting COMP_MODE, COMP_POL, COMP_LAT to 0 and COMP_QUE[1:0] to 0b11
    config |= (0x03)  # COMP_QUE[1:0] bits

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

    # Right-shift to align the 12-bit result (TLA2024 outputs data in bits [15:4])
    raw_adc = result >> 4

    # Convert to signed 12-bit integer (if necessary)
    if raw_adc > 0x7FF:
        raw_adc -= 0x1000

    # Calculate the voltage based on the ADC's full-scale range (±2.048V)
    voltage = raw_adc * 0.001  # LSB size is 1 mV

    return voltage

# Threshold voltages
LOW_THRESHOLD = 1.5  # Voltage below which we increment the wiper position
HIGH_THRESHOLD = 1.5  # Voltage above which we decrement the wiper position
CONSECUTIVE_COUNT = 15  # Number of consecutive readings required to adjust the wiper

# Initialize counters
above_threshold_counter = 0
below_threshold_counter = 0

# Kalman filter variables for each channel
kalman_vars = {
    'q1': {'x_est': 0.0, 'P': 1.0},
    'q2': {'x_est': 0.0, 'P': 1.0},
    'q3': {'x_est': 0.0, 'P': 1.0},
    'q4': {'x_est': 0.0, 'P': 1.0},
    'avg': {'x_est': 0.0, 'P': 1.0}
}

# Initialize measurement windows for dynamic R estimation
window_size = 20  # Number of recent measurements to consider
measurement_windows = {
    'q1': [],
    'q2': [],
    'q3': [],
    'q4': [],
    'avg': []
}

# Minimum R value to prevent division by zero or too small R
MIN_R = 1e-5

# Scaling factor for Q adjustment
q_scale = 1e-2  # Adjust based on system dynamics

def kalman_filter(z, x_est_prev, P_prev, Q, R):
    # Prediction step
    x_pred = x_est_prev
    P_pred = P_prev + Q

    # Update step
    K = P_pred / (P_pred + R)
    x_est = x_pred + K * (z - x_pred)
    P = (1 - K) * P_pred

    return x_est, P

# Set up the figure and axis for the plot before defining the update function
plt.figure(figsize=(12, 6))

# Initialize variables for sampling interval measurement
last_update_time = None

# Function to update the plot
def update(frame):
    try:
        global wiper_position
        global above_threshold_counter, below_threshold_counter, kalman_vars, measurement_windows
        global last_update_time

        # Initialize adjusted_wiper_positions
        adjusted_wiper_positions = []

        # Generate current datetime
        current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Measure sampling interval
        current_time = time.time()
        if last_update_time is not None:
            sampling_interval = current_time - last_update_time
        else:
            sampling_interval = 0.1  # Default value

        last_update_time = current_time

        # Read and average ADC readings for each channel
        index = indexes[-1] + 1 if indexes else 1
        sample_count = 10  # Number of samples to average

        q1_sum = 0.0
        q2_sum = 0.0
        q3_sum = 0.0
        q4_sum = 0.0

        try:
            for _ in range(sample_count):
                q1_sample = read_adc_channel(0, tla2024_address, bus)
                q2_sample = read_adc_channel(1, tla2024_address, bus)
                q3_sample = read_adc_channel(2, tla2024_address, bus)
                q4_sample = read_adc_channel(3, tla2024_address, bus)

                q1_sum += q1_sample
                q2_sum += q2_sample
                q3_sum += q3_sample
                q4_sum += q4_sample
        except Exception as e:
            print(f"Error reading ADC channels: {e}")
            q1 = q2 = q3 = q4 = None
        else:
            q1 = q1_sum / sample_count
            q2 = q2_sum / sample_count
            q3 = q3_sum / sample_count
            q4 = q4_sum / sample_count

        # Append raw data to lists
        q1_raw_values.append(q1)
        q2_raw_values.append(q2)
        q3_raw_values.append(q3)
        q4_raw_values.append(q4)

        # Update measurement windows
        for key, value in zip(['q1', 'q2', 'q3', 'q4'], [q1, q2, q3, q4]):
            if value is not None:
                measurement_windows[key].append(value)
                if len(measurement_windows[key]) > window_size:
                    measurement_windows[key].pop(0)

        # Dynamic estimation of R for each channel
        R_values = {}
        for key in ['q1', 'q2', 'q3', 'q4']:
            window = measurement_windows[key]
            if len(window) >= 2:
                R_est = np.var(window)
                R_values[key] = max(R_est, MIN_R)
            else:
                R_values[key] = 1e-2  # Default value if not enough data

        # Apply Kalman filter to each reading with dynamic R
        q_filtered = {}
        for key, value in zip(['q1', 'q2', 'q3', 'q4'], [q1, q2, q3, q4]):
            if value is not None:
                # Dynamic adjustment of Q based on the rate of change
                delta = abs(value - kalman_vars[key]['x_est'])
                Q_dynamic = q_scale * delta
                kalman_vars[key]['x_est'], kalman_vars[key]['P'] = kalman_filter(
                    value, kalman_vars[key]['x_est'], kalman_vars[key]['P'], Q_dynamic, R_values[key]
                )
                q_filtered[key] = kalman_vars[key]['x_est']
            else:
                q_filtered[key] = None

        q1_filtered = q_filtered['q1']
        q2_filtered = q_filtered['q2']
        q3_filtered = q_filtered['q3']
        q4_filtered = q_filtered['q4']

        # Calculate the average of the filtered voltages
        filtered_voltages = [v for v in [q1_filtered, q2_filtered, q3_filtered, q4_filtered] if v is not None]
        if filtered_voltages:
            avg_filtered_voltage = sum(filtered_voltages) / len(filtered_voltages)
            # Update measurement window for avg
            measurement_windows['avg'].append(avg_filtered_voltage)
            if len(measurement_windows['avg']) > window_size:
                measurement_windows['avg'].pop(0)
            # Dynamic estimation of R for avg
            if len(measurement_windows['avg']) >= 2:
                R_avg_est = np.var(measurement_windows['avg'])
                R_avg = max(R_avg_est, MIN_R)
            else:
                R_avg = 1e-2  # Default value
            # Dynamic adjustment of Q for avg
            delta_avg = abs(avg_filtered_voltage - kalman_vars['avg']['x_est'])
            Q_avg_dynamic = q_scale * delta_avg
            # Apply Kalman filter to the average
            kalman_vars['avg']['x_est'], kalman_vars['avg']['P'] = kalman_filter(
                avg_filtered_voltage, kalman_vars['avg']['x_est'], kalman_vars['avg']['P'], Q_avg_dynamic, R_avg
            )
            avg_voltage_filtered = kalman_vars['avg']['x_est']
        else:
            avg_voltage_filtered = None

        # Use the filtered average voltage for control logic
        avg_raw_voltage = sum([v for v in [q1, q2, q3, q4] if v is not None]) / len([v for v in [q1, q2, q3, q4] if v is not None])
        avg_raw_values.append(avg_raw_voltage)

        if avg_voltage_filtered is not None:
            avg_voltage = avg_voltage_filtered  # Replace the raw average with the filtered one
        else:
            avg_voltage = None  # Handle as needed

        # Adjust wiper position based on the filtered average voltage and counters
        if avg_voltage is not None:
            if avg_voltage < LOW_THRESHOLD:
                below_threshold_counter += 1
                above_threshold_counter = 0
                print(f"Filtered average voltage {avg_voltage:.2f}V < {LOW_THRESHOLD}V: below_threshold_counter = {below_threshold_counter}")
            elif avg_voltage > HIGH_THRESHOLD:
                above_threshold_counter += 1
                below_threshold_counter = 0
                print(f"Filtered average voltage {avg_voltage:.2f}V > {HIGH_THRESHOLD}V: above_threshold_counter = {above_threshold_counter}")
            else:
                above_threshold_counter = 0
                below_threshold_counter = 0
                print(f"Filtered average voltage {avg_voltage:.2f}V within thresholds.")

            # Check if the counters have reached the required consecutive count
            if below_threshold_counter >= CONSECUTIVE_COUNT:
                wiper_position += 1
                below_threshold_counter = 0  # Reset counter after adjustment
                print(f"Filtered average voltage consistently below {LOW_THRESHOLD}V: Incrementing wiper position to {wiper_position}")
                # Ensure wiper_position is within 0-127
                wiper_position = max(0, min(wiper_position, 127))
                set_wiper_position(bus, mcp4018_address, wiper_position)
            elif above_threshold_counter >= CONSECUTIVE_COUNT:
                wiper_position -= 1
                above_threshold_counter = 0  # Reset counter after adjustment
                print(f"Filtered average voltage consistently above {HIGH_THRESHOLD}V: Decrementing wiper position to {wiper_position}")
                # Ensure wiper_position is within 0-127
                wiper_position = max(0, min(wiper_position, 127))
                set_wiper_position(bus, mcp4018_address, wiper_position)
        else:
            print("Filtered average voltage is None, skipping wiper adjustment")

        # Debugging printouts
        print(f"Index: {index}, Wiper position: {wiper_position}, Time: {current_datetime}")
        print(f"Raw Voltages - Q1: {q1}, Q2: {q2}, Q3: {q3}, Q4: {q4}, Average: {avg_raw_voltage}")
        print(f"Filtered Voltages - Q1: {q1_filtered}, Q2: {q2_filtered}, Q3: {q3_filtered}, Q4: {q4_filtered}, Average: {avg_voltage_filtered}")

        # Keep only the last NUM_READINGS - 1 points BEFORE appending new data
        if len(indexes) >= NUM_READINGS:
            indexes.pop(0)
            q1_values.pop(0)
            q2_values.pop(0)
            q3_values.pop(0)
            q4_values.pop(0)
            avg_values.pop(0)
            q1_raw_values.pop(0)
            q2_raw_values.pop(0)
            q3_raw_values.pop(0)
            q4_raw_values.pop(0)
            avg_raw_values.pop(0)
            wiper_positions.pop(0)
            if convolution_results:
                convolution_results.pop(0)
            if adjusted_wiper_positions:
                adjusted_wiper_positions.pop(0)

        # Append filtered data to lists for plotting
        indexes.append(index)
        q1_values.append(q1_filtered)
        q2_values.append(q2_filtered)
        q3_values.append(q3_filtered)
        q4_values.append(q4_filtered)
        avg_values.append(avg_voltage_filtered)
        wiper_positions.append(wiper_position)

        # Determine the number of samples corresponding to 0.5 seconds
        window_duration = 0.5    # Desired window duration in seconds
        window_size = int(window_duration / sampling_interval)  # Number of samples in the window

        # Ensure both lists have sufficient data for the window
        if len(avg_values) >= window_size and len(wiper_positions) >= window_size:
            # Extract the data for the window
            avg_values_window = avg_values[-window_size:]
            wiper_positions_window = wiper_positions[-window_size:]

            # Subtract 95 from wiper positions
            adjusted_wiper_positions_window = [wp - 95 for wp in wiper_positions_window]

            # Normalize the adjusted wiper positions
            max_adjusted_wiper = max(map(abs, adjusted_wiper_positions_window)) or 1  # Avoid division by zero
            normalized_wiper = [wp / max_adjusted_wiper for wp in adjusted_wiper_positions_window]

            # Perform convolution
            convolution_window = np.convolve(avg_values_window, normalized_wiper, mode='same')

            # Create a convolution result that aligns with the full data length
            padding_length = len(indexes) - len(convolution_window)
            convolution_results = [0] * padding_length + convolution_window.tolist()

            # Update adjusted_wiper_positions for plotting
            adjusted_wiper_positions = [0] * padding_length + adjusted_wiper_positions_window
        else:
            convolution_results = [0] * len(indexes)
            adjusted_wiper_positions = [0] * len(wiper_positions)

        # Append data to CSV file
        with open(csv_filename, mode='a', newline='') as file:
            csv_writer = csv.writer(file)
            conv_result = convolution_results[-1] if convolution_results else ''
            csv_writer.writerow([
                index, current_datetime,
                q1, q2, q3, q4, avg_raw_voltage,  # Raw values
                q1_filtered, q2_filtered, q3_filtered, q4_filtered, avg_voltage_filtered,  # Filtered values
                wiper_position,
                conv_result
            ])

        # Clear previous plots
        plt.cla()

        # Ensure data lists are the same length for plotting
        min_length = min(len(indexes), len(avg_values), len(wiper_positions), len(convolution_results))
        indexes_plot = indexes[-min_length:]
        avg_values_plot = avg_values[-min_length:]
        wiper_positions_plot = wiper_positions[-min_length:]
        adjusted_wiper_positions_plot = adjusted_wiper_positions[-min_length:]
        convolution_results_plot = convolution_results[-min_length:]

        # Plot the convolution result
        plt.plot(indexes_plot, convolution_results_plot, label='Convolution', color='purple')

        # Plot the filtered average voltage
        plt.plot(indexes_plot, avg_values_plot, label='Average Filtered', linestyle='--', color='black')

        # Plot the adjusted wiper positions (scaled)
        max_avg_voltage = max(avg_values_plot) if avg_values_plot else 1
        max_adjusted_wiper = max(map(abs, adjusted_wiper_positions_plot)) or 1
        wiper_positions_scaled = [wp * (max_avg_voltage / max_adjusted_wiper) for wp in adjusted_wiper_positions_plot]
        plt.plot(indexes_plot, wiper_positions_scaled, label='Adjusted Wiper Position (scaled)', linestyle=':', color='green')

        # Adjust plot labels and legend
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title(f'Real-Time Data (Last {NUM_READINGS} Readings)\nWiper Position: {wiper_position}', fontsize=14)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

        # Set x-limits to show the last NUM_READINGS readings
        plt.xlim(indexes_plot[0], indexes_plot[-1])

        # Optionally, set y-limits
        combined_data = avg_values_plot + convolution_results_plot + wiper_positions_scaled
        if combined_data:
            plt.ylim(min(combined_data), max(combined_data))
        else:
            plt.ylim(0, 3.3)
    except Exception as e:
        print(f"An error occurred in the update function: {e}")
        import traceback
        traceback.print_exc()

# Create a CSV file and write the header
with open(csv_filename, mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow([
        'Index', 'Datetime',
        'Q1_raw', 'Q2_raw', 'Q3_raw', 'Q4_raw', 'Average_raw',
        'Q1_filtered', 'Q2_filtered', 'Q3_filtered', 'Q4_filtered', 'Average_filtered',
        'Wiper_Position', 'Convolution_Result'
    ])

# Use FuncAnimation to update the plot in real-time every 100 ms
ani = FuncAnimation(plt.gcf(), update, interval=100, cache_frame_data=False)

# Adjust layout to make space for the legend
plt.subplots_adjust(bottom=0.25)

# Show the plot
plt.show()
