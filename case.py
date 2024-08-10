import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples to generate
n_samples = 10000

# Generate synthetic data for each feature
data = {
    "PassengerCount": np.random.randint(50, 300, size=n_samples),
    "PassengerLoadFactor": np.random.uniform(0.5, 1.0, size=n_samples),
    "AverageAge": np.random.randint(18, 70, size=n_samples),
    "LoyaltyTier": np.random.choice([1, 2, 3, 4], size=n_samples),
    "PriorityBoardingCount": np.random.randint(0, 50, size=n_samples),
    "NumberOfBoardingGroups": np.random.randint(1, 5, size=n_samples),
    "AircraftType": np.random.choice([1, 2, 3], size=n_samples),
    "GateProximity": np.random.choice([1, 2, 3], size=n_samples),
    "FlightTime": np.random.choice([1, 2, 3, 4], size=n_samples),
    "StaffCount": np.random.randint(3, 10, size=n_samples),
    "GroundCrewEfficiency": np.random.uniform(0.8, 1.2, size=n_samples),  # Efficiency multiplier
    "GateType": np.random.choice([1, 2], size=n_samples),  # 1: Aerobridge, 2: Remote Stand
    "WeatherConditions": np.random.choice([1, 2, 3], size=n_samples),  # 1: Clear, 2: Rainy, 3: Stormy
    "PassengerDemographics": np.random.choice([1, 2, 3], size=n_samples)  # 1: Mixed, 2: Families, 3: Elderly
}

# Calculate TotalBoardingTime based on features
data["TotalBoardingTime"] = (
    0.05 * data["PassengerCount"] +
    0.2 * (4 - data["LoyaltyTier"]) +
    0.1 * (50 - data["PriorityBoardingCount"]) +
    3 * data["NumberOfBoardingGroups"] +
    5 * data["AircraftType"] +
    2 * data["GateProximity"] +
    10 * (1 - data["GroundCrewEfficiency"]) +  # Efficiency reduces time
    3 * data["GateType"] +  # Remote stands take longer
    5 * data["WeatherConditions"] +  # Bad weather increases time
    2 * (3 - data["PassengerDemographics"]) +  # Families/Elderly increase time
    np.random.normal(0, 2, size=n_samples)  # Random noise
)

# Generate AirportFees based on TotalBoardingTime (e.g., linear relationship)
data["AirportFees"] = (
    10 * data["TotalBoardingTime"] +  # Fee based on time
    1000 +  # Base fee
    np.random.normal(0, 50, size=n_samples)  # Random noise
)

# Create a DataFrame
df_synthetic = pd.DataFrame(data)

# Display the first few rows of the synthetic dataset
df_synthetic.head()
