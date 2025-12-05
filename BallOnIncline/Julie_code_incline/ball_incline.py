import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.signal import find_peaks

class DistanceAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.offset = None
        self.sensor_distances = None
        self.relative_distances = None
        self.fit_results = []

    def load_data(self):
        # Use tab delimiter to read the file
        self.data = pd.read_csv(self.file_path, delimiter='\t', header=None)
        self.offset = self.data.iloc[:, 0]
        self.sensor_distances = self.data.iloc[:, 1:]

    def calculate_relative_distances(self):
        self.relative_distances = self.sensor_distances.sub(self.sensor_distances.iloc[:, 0], axis=0)

    def perform_linear_fits(self):
        self.fit_results = []  # Clear previous results
        for col in range(1, self.relative_distances.shape[1]):
            y = self.relative_distances.iloc[:, col].values
            x = np.arange(1, len(self.relative_distances) + 1)  # Experiment numbers as row indices + 1

            # Remove NaN values (from diff calculation)
            valid_indices = ~np.isnan(x)
            x_valid = x[valid_indices]
            y_valid = y[valid_indices]

            # Perform linear regression
            slope, intercept, _, _, std_err = linregress(x_valid, y_valid)

            # Store results for each fit
            self.fit_results.append({
                'sensor_pair': f"Sensor {col} to Sensor {col + 1}",
                'slope': slope,
                'intercept': intercept,
                'std_err': std_err
            })

    def print_fit_results(self):
        for result in self.fit_results:
            print(f"{result['sensor_pair']}:")
            print(f"  Slope: {result['slope']}")
            print(f"  Intercept: {result['intercept']}")
            print(f"  Standard Error: {result['std_err']}")

    def plot_results(self):
        for col in range(1, self.relative_distances.shape[1]):
            y = self.relative_distances.iloc[:, col].values
            x = np.arange(1, len(self.relative_distances) + 1)  # Experiment numbers as row indices + 1

            # Remove NaN values (from diff calculation)
            valid_indices = ~np.isnan(x)
            x_valid = x[valid_indices]
            y_valid = y[valid_indices]

            # Calculate fitted values
            slope = self.fit_results[col - 1]['slope']
            intercept = self.fit_results[col - 1]['intercept']
            fitted_values = slope * x_valid + intercept

            # Calculate mean and std for the legend
            mean_value = np.mean(y_valid)
            std_value = np.std(y_valid)
            # Save mean and std to a CSV file
            with open("mean_std_sensors.csv", "a") as file:
                if file.tell() == 0:  # Check if the file is empty to write the header
                    file.write("Experiment,Mean,Std\n")
                file.write(f"{col},{mean_value:.2f},{std_value:.2f}\n")
            # Plot
            plt.scatter(x_valid, y_valid, label=f"Experiment {col}: Mean={mean_value:.2f}, Std={std_value:.2f}", alpha=0.7)
            plt.plot(x_valid, fitted_values, label=f"Fit for Experiment {col}", color='red')

        plt.ylabel('Relative Distance to Offset (cm)')
        plt.xlabel('Experiment Number')
        plt.legend()
        plt.title('Experiment vs Relative Distance')
        plt.show()





class PeakAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.filtered_data = None
        self.peak_centers = []

    def load_data(self):
        # Load the data assuming a CSV format
        self.data = pd.read_csv(self.file_path, header=16, delimiter = ',')
        self.data.columns = ['x', 'y']  # Assuming two columns: x (time/position) and y (voltage)

    def plot_raw_data(self):
        plt.plot(self.data['x'], self.data['y'], label='Raw Data')
        plt.xlabel('X (Time/Position)')
        plt.ylabel('Y (Voltage)')
        plt.title('Raw Data')
        plt.legend()
        plt.show()

    def filter_data(self, threshold=1):
        # Remove points where y (voltage) is close to zero
        self.filtered_data = self.data[self.data['y'].abs() > threshold]

    def fit_step_functions(self):

        # Find peaks in the filtered data
        peaks, _ = find_peaks(self.filtered_data['y'], height=0)
        self.peak_centers = self.filtered_data.iloc[peaks]['x'].values

    def plot_filtered_data_and_peaks(self):
        plt.plot(self.filtered_data['x'], self.filtered_data['y'], label='Filtered Data', marker='.')
        plt.scatter(self.peak_centers, [self.filtered_data[self.filtered_data['x'] == x]['y'].values[0] for x in self.peak_centers],
                    color='red', label='Peak Centers')
        plt.xlabel('X (Time/Position)')
        plt.ylabel('Y (Voltage)')
        plt.title('Filtered Data with Peaks')
        plt.legend()
        plt.show()

    def get_peak_centers(self):
        return self.peak_centers
    def calculate_acceleration(self):
        # Load the relative distances from the file
        mean_std_data = pd.read_csv("mean_std_sensors.csv")
        accelerations = []
        uncertainties = []

        # Calculate time intervals as differences between peak centers
        time_intervals = np.diff(self.peak_centers)

        for i in range(len(mean_std_data) - 1):
            # Calculate relative distance between consecutive sensors
            distance = mean_std_data['Mean'][i + 1] - mean_std_data['Mean'][i]
            time = time_intervals[i]  # Time interval between sensors

            # Calculate acceleration using the formula: a = 2 * distance / time^2
            acceleration = 2 * distance / (time ** 2)
            accelerations.append(acceleration)

        # Calculate weighted mean acceleration and its uncertainty
        weights = []
        weighted_accelerations = []

        for i in range(len(mean_std_data) - 1):
            # Calculate relative distance between consecutive sensors
            distance = mean_std_data['Mean'][i + 1] - mean_std_data['Mean'][i]
            time = time_intervals[i]  # Time interval between sensors

            # Calculate acceleration using the formula: a = 2 * distance / time^2
            acceleration = 2 * distance / (time ** 2)

            # Calculate weight based on uncertainties
            std1 = mean_std_data['Std'][i]
            std2 = mean_std_data['Std'][i + 1]
            weight = 1 / (std1 ** 2 + std2 ** 2)

            # Store weighted acceleration and weight
            weighted_accelerations.append(acceleration * weight)
            weights.append(weight)

        # Calculate weighted mean acceleration
        weighted_mean_acceleration = sum(weighted_accelerations) / sum(weights)

        # Calculate uncertainty on the weighted mean acceleration
        weighted_uncertainty = np.sqrt(1 / sum(weights))

        # Store results in a DataFrame
        results = pd.DataFrame({
            'Weighted Mean Acceleration': [weighted_mean_acceleration],
            'Uncertainty': [weighted_uncertainty]
        })

        # Save results to a CSV file
        results.to_csv("weighted_acceleration_results.csv", index=False)

        # Print results
        print(results)

