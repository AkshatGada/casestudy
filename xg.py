from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import pickle

# Split the data into features and target
X = df_synthetic.drop(columns=["AirportFees"])
y = df_synthetic["AirportFees"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost regressor
model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the trained model to a .pkl file
model_filename = "/mnt/data/xgboost_model.pkl"
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

model_filename
