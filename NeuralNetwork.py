# %%
import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# %%
# Load the training dataset
file_path_train = "/Users/huaijinsun/Downloads/training.xlsx"
df_train = pd.read_excel(file_path_train)

# Load the testing dataset
file_path_test = "/Users/huaijinsun/Downloads/scoring.xlsx"
df_test = pd.read_excel(file_path_test)

# %%
# Tidy data

# Remove the 'Region' column
df_train = df_train.drop(columns=['Region'])
df_test = df_test.drop(columns=['Region'])

# Convert all values in 'GVWR Class' column to strings
df_train['GVWR Class'] = df_train['GVWR Class'].astype(str)
df_test['GVWR Class'] = df_test['GVWR Class'].astype(str)

# Convert all values in 'Number of Vehicles Registered at the Same Address' column to strings
df_train['Number of Vehicles Registered at the Same Address'] = df_train['Number of Vehicles Registered at the Same Address'].astype(str)
df_test['Number of Vehicles Registered at the Same Address'] = df_test['Number of Vehicles Registered at the Same Address'].astype(str)

print(df_train.head())
print(df_test.head())

# %%
# Transform Date, GVWR Class Fuel Type, Model Year, Fuel Technology, Electric Mile Range, and Number of Vehicles Registered at the Same Address columns
# Initialize the LabelEncoder
label_encoders = {}

# List of columns to encode
columns_to_encode = ['Date', 'Vehicle Category', 'GVWR Class', 'Fuel Type', 'Model Year', 'Fuel Technology', 'Electric Mile Range', 'Number of Vehicles Registered at the Same Address']

# Fit and transform the training data, and transform the test data
for column in columns_to_encode:
	label_encoders[column] = LabelEncoder()
	df_train[column] = label_encoders[column].fit_transform(df_train[column])
	df_test[column] = df_test[column].map(lambda s: '<unknown>' if s not in label_encoders[column].classes_ else s)
	label_encoders[column].classes_ = np.append(label_encoders[column].classes_, '<unknown>')
	df_test[column] = label_encoders[column].transform(df_test[column])

# Display the first few rows of the updated dataset
print(df_train.head())
print(df_test.head())

# %%
X_train = df_train.drop(columns=['Vehicle Population']) 
y_train = df_train['Vehicle Population']  
X_test = df_test.drop(columns=['Vehicle Population'])  
y_test = df_test['Vehicle Population']  

# Build the neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error', optimizer=optimizer)

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# %%
rms = math.sqrt(mse)
print('rms:', rms)

# %%
# Define the target and features
X_train = df_train.drop(columns=['Vehicle Population']) 
y_train = df_train['Vehicle Population']  
X_test = df_test.drop(columns=['Vehicle Population'])  
y_test = df_test['Vehicle Population']  

# Define the K-fold Cross Validator
kf = KFold(n_splits = 5, shuffle = True, random_state = 42)

# Initialize lists to store results
mse_scores = []

# K-fold Cross Validation model evaluation
for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
    # Build the neural network model
    model = Sequential()
    model.add(Dense(64, input_dim=X_train_fold.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    
    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    # Train the model
    model.fit(X_train_fold, y_train_fold, epochs=100, batch_size=32, verbose=0)

    # Evaluate the model on the validation fold
    y_val_pred = model.predict(X_val_fold)
    mse = mean_squared_error(y_val_fold, y_val_pred)
    mse_scores.append(mse)

# Calculate the average MSE from cross-validation
average_mse_cv = np.mean(mse_scores)
print(f"Cross-Validation Mean Squared Error: {average_mse_cv}")

# Evaluate the model on the testing dataset
y_test_pred = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_test_pred)
print(f"Testing Mean Squared Error: {mse_test}")

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Define the model
model = RandomForestRegressor()

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model on the validation set
y_pred = best_model.predict(X_val)
rms_error = np.sqrt(mean_squared_error(y_val, y_pred))
print(f'RMS Error on CV Set: {rms_error}')

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse}")


# %%
print(f"BEST PARAM: {best_params}")
print(f"BEST MODEL: {best_model}")

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Define the model
model = RandomForestRegressor()

# Define the parameter grid
param_grid = {
    'n_estimators': [450, 500, 550, 600],
    'max_depth': [None, 25, 30, 35],
    'min_samples_split': [1, 2, 3, 4],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# %%
#Mean Squared Error: 32218414.343950592
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Define the model
model = RandomForestRegressor()

# Define the parameter grid
param_grid = {
    'n_estimators': [500],
    'max_depth': [None, 20, 30, 40],
    'min_samples_split': [1, 2, 4],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Define the model
model = RandomForestRegressor()

# Define the parameter grid
param_grid = {
    'n_estimators': [500],
    'max_depth': [None, 30],
    'min_samples_split': [1, 2],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# %%
print(f"BEST PARAM: {best_params}")
print(f"BEST MODEL: {best_model}")

# %%
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Define the model
model = RandomForestRegressor()

# Define the parameter grid
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'bootstrap': [True, False]
}

# Perform random search with cross-validation
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = random_search.best_params_
best_model = random_search.best_estimator_

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# %%
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Define the model
model = RandomForestRegressor()

# Define the parameter grid
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'bootstrap': [True, False]
}

# Perform random search with cross-validation
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = random_search.best_params_
best_model = random_search.best_estimator_

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# %%
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np

X_train = df_train.drop(columns=['Vehicle Population']) 
y_train = df_train['Vehicle Population']  
X_test = df_test.drop(columns=['Vehicle Population'])  
y_test = df_test['Vehicle Population']  

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define the model
xgb_model = XGBRegressor()

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model on the validation set
y_pred = best_model.predict(X_val)
rms_error = np.sqrt(mean_squared_error(y_val, y_pred))
print(f'RMS Error on CV Set: {rms_error}')

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define base models
base_models = [
    ('rf', RandomForestRegressor(max_depth=30, n_estimators=500)),
    ('xgb', XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8))
]

# Define the meta-model
meta_model = RidgeCV()

# Create the stacking model
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# Train the stacking model
stacking_model.fit(X_train, y_train)

# Evaluate the stacking model on the validation set
y_pred = stacking_model.predict(X_val)
rms_error = np.sqrt(mean_squared_error(y_val, y_pred))
print(f'RMS Error on CV Set: {rms_error}')

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# %%
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet

# Define base models
base_models = [
    ('rf', RandomForestRegressor(max_depth=30, n_estimators=500)),
    ('xgb', XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8)),
    ('gbm', GradientBoostingRegressor(n_estimators=200, max_depth=5)),
    ('svr', SVR(C=1.0, epsilon=0.2)),
    ('knn', KNeighborsRegressor(n_neighbors=5)),
    ('linear', LinearRegression()),
    ('lasso', Lasso(alpha=0.1)),
    ('elasticnet', ElasticNet(alpha=0.1))
]

# Define the meta-model
meta_model = RidgeCV()

# Create the stacking model
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# Train the stacking model
stacking_model.fit(X_train, y_train)

# Evaluate the stacking model on the validation set
y_pred = stacking_model.predict(X_val)
rms_error = np.sqrt(mean_squared_error(y_val, y_pred))
print(f'RMS Error on CV Set: {rms_error}')

# Evaluate the best model on the test set
y_pred = stacking_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# %%
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

X_train = df_train.drop(columns=['Vehicle Population']) 
y_train = df_train['Vehicle Population']  
X_test = df_test.drop(columns=['Vehicle Population'])  
y_test = df_test['Vehicle Population']  

# Assuming X and y are your features and target variable
X_train1, X_val, y_train1, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define the Gradient Boosting model
gbm_model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1)

# Train the model
gbm_model.fit(X_train1, y_train1)

# Evaluate the model on the validation set
y_pred = gbm_model.predict(X_val)
rms_error = np.sqrt(mean_squared_error(y_val, y_pred))
print(f'RMS Error on CV Set: {rms_error}')

# Evaluate the model on the test set
y_test_pred = gbm_model.predict(X_test)
test_rms_error = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(f'RMS Error on Test Set: {test_rms_error}')

# %%
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

# Initialize the model
gbm_model = GradientBoostingRegressor()

# Perform Grid Search
grid_search = GridSearchCV(estimator=gbm_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train1, y_train1)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate the best model on the validation set
y_pred = best_model.predict(X_val)
rms_error = np.sqrt(mean_squared_error(y_val, y_pred))
print(f'RMS Error on CV Set: {rms_error}')
print(f'Best Parameters: {grid_search.best_params_}')



# %%
y_test_pred = best_model.predict(X_test)
test_rms_error = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(f'RMS Error on Test Set: {test_rms_error}')
