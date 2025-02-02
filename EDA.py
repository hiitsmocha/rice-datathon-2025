#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ### List of model to test:
# * Random Forest
# * XGBoost 
# * Catboost
# * MLP

# ## I. Data processing

# In[239]:


train_and_val = pd.read_excel('/Users/trongphan/Downloads/Rice_Datathon/rice-datathon-2025/data/training.xlsx')


# In[240]:


train_and_val.columns


# In[212]:


def summarize_df(df, max_columns=10):
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist()[:max_columns] + ['...'] if len(df.columns) > max_columns else df.columns.tolist(),
        'missing_values': {col: df.isnull().sum()[col] for col in df.columns[:max_columns]} if len(df.columns) > max_columns else df.isnull().sum().to_dict(),
        'describe': df.describe(include='all').iloc[:, :max_columns].to_dict() if len(df.columns) > max_columns else df.describe(include='all').to_dict(),
        'num_columns': len(df.columns),
        'distinct_values': {col: df[col].nunique() for col in df.columns[:max_columns]} if len(df.columns) > max_columns else {col: df[col].nunique() for col in df.columns},
        'columns_with_few_distinct_values': {col: df[col].unique().tolist() for col in df.columns[:max_columns] if df[col].nunique() < 20} if len(df.columns) > max_columns else {col: df[col].unique().tolist() for col in df.columns if df[col].nunique() < 20}
    }
    return summary


# In[241]:


import json
summary = summarize_df(train_and_val)

with open('summary.json', 'w') as f:
    json.dump(summary, f, indent=4)


# In[242]:


# Data processing

# Convert 'Date' to datetime format and extract the year
train_and_val['Date'] = pd.to_datetime(train_and_val['Date'], format='%Y').dt.year

# Fill missing values in 'Model Year' with the median value
train_and_val['Model Year'].fillna(train_and_val['Model Year'].median(), inplace=True)

# Convert 'Model Year' to integer
train_and_val['Model Year'] = train_and_val['Model Year'].astype(int)

# Convert categorical columns to category dtype
categorical_columns = ['Vehicle Category', 'GVWR Class', 'Fuel Type', 'Fuel Technology', 'Electric Mile Range', 'Number of Vehicles Registered at the Same Address', 'Model Year']
for col in categorical_columns:
    train_and_val[col] = train_and_val[col].astype('category')

# Remove duplicates
train_and_val.drop_duplicates(inplace=True)

# Reset index
train_and_val.reset_index(drop=True, inplace=True)

# Display the cleaned dataframe
train_and_val = train_and_val.drop(columns=["Region"])


# In[243]:


# Load data (replace with your actual dataset)
data = train_and_val
# Preprocessing
# One-hot encode categorical columns
categorical_cols = ["Vehicle Category", "Fuel Type", "Fuel Technology", "Electric Mile Range"]
data = pd.get_dummies(data, columns=categorical_cols)


# In[244]:


# Ordinal encode "Number of Vehicles Registered at the Same Address"
ordinal_mapping = {1: int(1), 2: int(2), 3: int(3), "\u22654": 4, "Unknown": -1}
data["Number of Vehicles Registered at the Same Address"] = data["Number of Vehicles Registered at the Same Address"].map(ordinal_mapping)


# In[245]:


# Preprocessing
# Impute missing values in "Model Year"
imputer = SimpleImputer(strategy="median")
data["Model Year"] = imputer.fit_transform(data[["Model Year"]])
# Handle non-numeric values in "GVWR Class"
# For this GVWR, we find that online but decided to not replace it with any value since we still treat it as categorical
data["GVWR Class"] = data["GVWR Class"].replace({"Not Applicable": -1, "Unknown": -1})


# In[246]:


# Feature engineering
data["Vehicle Age"] = data["Date"] - data["Model Year"]


# In[247]:


# Plot the new feature "Vehicle Age" and "Vehicle Population"
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.scatterplot(x="Vehicle Age", y="Vehicle Population", data=data)
plt.title("Vehicle Age vs. Vehicle Population")
plt.xlabel("Vehicle Age")
plt.ylabel("Vehicle Population")
plt.grid(True)
plt.show()


# In[248]:


# Check for missing values in the dataset
print("Missing values in each column:")
print(data.isnull().sum())

# Handle missing values in all columns
for col in data.columns:
    if data[col].dtype == "object" or pd.api.types.is_categorical_dtype(data[col]):  # Categorical columns
        data[col].fillna(data[col].mode()[0], inplace=True)
    else:  # Numerical columns
        data[col].fillna(data[col].median(), inplace=True)


# In[249]:


# Split data
X = data.drop(columns=["Vehicle Population"])
y = data["Vehicle Population"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# ### III. Testset
# #### 3. Build test set

# In[250]:


df_test = pd.read_excel('/Users/trongphan/Downloads/Rice_Datathon/rice-datathon-2025/data/scoring & submission format.xlsx')
test_data = df_test

# Convert 'Date' to datetime format and extract the year
test_data['Date'] = pd.to_datetime(test_data['Date'], format='%Y').dt.year
# Fill missing values in 'Model Year' with the median value
test_data['Model Year'].fillna(test_data['Model Year'].median(), inplace=True)
# Convert 'Model Year' to integer
test_data['Model Year'] = test_data['Model Year'].astype(int)

# Convert categorical columns to category dtype
categorical_columns = ['Vehicle Category', 'GVWR Class', 'Fuel Type', 'Fuel Technology', 'Electric Mile Range', 'Number of Vehicles Registered at the Same Address', 'Region']
for col in categorical_columns:
    test_data[col] = test_data[col].astype('category')
# One-hot encode categorical columns
categorical_cols = ["Vehicle Category", "Fuel Type", "Fuel Technology", "Electric Mile Range"]
test_data = pd.get_dummies(df_test, columns=categorical_cols)
# Display the cleaned dataframe
test_data = test_data.drop(columns=["Region"])
# Remove duplicates
test_data.drop_duplicates(inplace=True)
# Reset index
test_data.reset_index(drop=True, inplace=True)
test_data
# Ordinal encode "Number of Vehicles Registered at the Same Address"
ordinal_mapping = {1: int(1), 2: int(2), 3: int(3), "\u22654": 4, "Unknown": -1}
test_data["Number of Vehicles Registered at the Same Address"] = test_data["Number of Vehicles Registered at the Same Address"].map(ordinal_mapping)

# Preprocessing
# Impute missing values in "Model Year"
imputer = SimpleImputer(strategy="median")
test_data["Model Year"] = imputer.fit_transform(test_data[["Model Year"]])
# Handle non-numeric values in "GVWR Class"
test_data["GVWR Class"] = test_data["GVWR Class"].replace({"Not Applicable": -1, "Unknown": -1})

# # Feature engineering
test_data["Vehicle Age"] = test_data["Date"] - test_data["Model Year"]

# Check for missing values in the test_dataset
print("Missing values in each column:")
print(test_data.isnull().sum())
# Handle missing values in all columns
for col in test_data.columns:
    if test_data[col].dtype == "object" or pd.api.types.is_categorical_dtype(test_data[col]):  # Categorical columns
        test_data[col].fillna(test_data[col].mode()[0], inplace=True)
    else:  # Numerical columns
        test_data[col].fillna(test_data[col].median(), inplace=True)

# Split test_data
X_test = test_data.drop(columns=["Vehicle Population"])
y_test = test_data["Vehicle Population"]

"""
Handle missing columns of X_test, which is Fuel Type_Unknown
"""

# Find the index of 'Fuel Type_Natural Gas'
insert_index = X_test.columns.get_loc('Fuel Type_Natural Gas') + 1

# Insert 'Fuel Type_Unknown' after 'Fuel Type_Natural Gas' and fill with 0
X_test.insert(insert_index, 'Fuel Type_Unknown', 0)

# Now X_test has the 'Fuel Type_Unknown' column after 'Fuel Type_Natural Gas'
print(X_test.columns)
X_test


# ## II. Random Forest

# #### 2. Train Vanilla Random Forest

# In[251]:


# Train Random Forest model

random_forest_model = RandomForestRegressor(random_state=28)
random_forest_model.fit(X_train, y_train)


# In[252]:


# Evaluate
import math
y_pred = random_forest_model.predict(X_val)
print("Mean Absolute Error (MAE):", mean_absolute_error(y_val, y_pred))
print("Mean Squared Error (MSE):", math.sqrt(mean_squared_error(y_val, y_pred)))
print("R-squared (R2):", r2_score(y_val, y_pred))

# Create a DataFrame with y_val and y_pred
results_df = pd.DataFrame({'y_val': y_val, 'y_pred': y_pred})
print(results_df)


# #### 4. Parameter tuning

# ##### a. Using GridSearchCV

# In[225]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import time


# In[226]:


# Define the parameter grid
param_grid = {
    "n_estimators": [50, 100, 200],  # Number of trees
    "max_depth": [None, 10, 20, 30],  # Maximum depth of trees
    "min_samples_split": [2, 5, 10],  # Minimum samples to split a node
    "min_samples_leaf": [1, 2, 4],  # Minimum samples at a leaf node
    "max_features": ["sqrt", "log2"],  # Number of features to consider for splits
    "bootstrap": [True, False]  # Whether to use bootstrap sampling
}

# Initialize the Random Forest model
rf = RandomForestRegressor(random_state=42)


# In[ ]:


# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring="neg_mean_squared_error",  # Use RMSE as the evaluation metric
    n_jobs=-1,  # Use all available CPU cores
    verbose=3  # Disable default verbose output
)

# Track time using tqdm
start_time = time.time()
with tqdm(total=len(grid_search.param_grid) * 5, desc="Grid Search Progress") as pbar:
    # Attach the callback to GridSearchCV
    grid_search.fit(X_train, y_train)


# In[ ]:


"""
Best Parameters: {'bootstrap': False, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}
Best R-squared Score: -14547067.220303629
Model 1:
Score: -14547067.220303629
RMSE on test set: 4167.32
"""

# Print the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best R-squared Score:", grid_search.best_score_)

# Evaluate on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Test R-squared:", r2_score(y_test, y_pred))


# ##### 2. Random state

# In[181]:


# param = {'bootstrap': False, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 50}

# Initialize the Random Forest model with the specified parameters
list_rmse = []
min_rmse = 99999999
for i in tqdm(range(1, 100)):
    rf = RandomForestRegressor(random_state=i)

    # Fit the model to the training data
    rf.fit(X_train, y_train)

    # Predict on the validation set
    y_pred = rf.predict(X_test)
    rmse_now = math.sqrt(mean_squared_error(y_test, y_pred))
    if rmse_now < min_rmse:
        min_rmse = rmse_now
        print(f"GET NEW MIN RMSE: {min_rmse} at state {i}")
    list_rmse.append({"state": i, "rmse": rmse_now})
    
top_5_rmse = sorted(list_rmse, key=lambda x: x['rmse'])[:5]
for top_rmse in top_5_rmse:
    print("State:", top_rmse['state'])
    print("Mean Squared Error (MSE):", top_rmse['rmse'])

    # # Create a DataFrame with y_test and y_pred
    # results_df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    # print(results_df)


# ##### Random Search (to save time)

# <!-- #### b. Using Randomized GridSearch -->

# In[146]:


from sklearn.model_selection import RandomizedSearchCV

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=50,  # Number of parameter combinations to try
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=3,
    random_state=42
)

# Fit the RandomizedSearchCV
random_search.fit(X_train, y_train)


# In[147]:


# Print the best parameters and best score
print("Best Parameters:", random_search.best_params_)
print("Best RMSE:", math.sqrt(-random_search.best_score_))


# 

# In[148]:


y_pred_random = random_search.predict(X_test)
rmse = math.sqrt(mean_squared_error(y_test, y_pred_random))
print("Test RMSE:", rmse)


# ## Test with Full data

# In[253]:


X_train_full = X
y_train_full = y


# In[254]:


# Train Random Forest model
param = {'bootstrap': False, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}
# model = RandomForestRegressor(random_state=42, **param)
model = RandomForestRegressor(random_state=28)
# model.fit(X_train_full, y_train_full)
model.fit(X_train, y_train)


# In[255]:


from sklearn.utils import shuffle
import math 
# Shuffle X_test and y_test
X_test, y_test = shuffle(X_test, y_test, random_state=42)

# Evaluate
y_pred_val = model.predict(X_val)
print("VALIDATION SET")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_val, y_pred_val))
print("Root Mean Squared Error (RMSE):", math.sqrt(mean_squared_error(y_val, y_pred_val)))
print("R-squared (R2):", r2_score(y_val, y_pred_val))

print("*" * 50)
y_pred = model.predict(X_test)
print("TEST SET")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", math.sqrt(mean_squared_error(y_test, y_pred)))
print("R-squared (R2):", r2_score(y_test, y_pred))

# Create a DataFrame with y_val and y_pred
results_df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
print(results_df)


# ## IV. New models

# ### 1. CatBoost only

# In[256]:


from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Define the parameter grid
param_grid = {
    "iterations": [500, 1000, 1500],  # Number of trees
    "learning_rate": [0.01, 0.05, 0.1],  # Step size shrinkage
    "depth": [4, 6, 8],  # Depth of the trees
    "l2_leaf_reg": [1, 3, 5],  # L2 regularization
    "random_strength": [0.1, 0.5, 1],  # Randomness for scoring splits
    "bagging_temperature": [0, 0.5, 1],  # Controls Bayesian bagging
    "border_count": [32, 64, 128],  # Number of splits for numerical features
    "verbose": [False]  # Disable verbose output during grid search
}


# In[257]:


X_train_cat = X_train.copy()
X_test_cat = X_test.copy()
# Define categorical columns
categorical_cols = list(X_train_cat.columns)
# Convert all columns to string
X_train_cat[categorical_cols] = X_train_cat[categorical_cols].astype(str)
X_test_cat[categorical_cols] = X_test_cat[categorical_cols].astype(str)
X_train_cat[categorical_cols] = X_train_cat[categorical_cols].astype(str)


# In[122]:


# Create the CatBoost model
catboost_model = CatBoostRegressor(random_state=28)

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=catboost_model,
    param_distributions=param_grid,
    n_iter=20,  # Number of parameter settings to sample
    scoring="neg_root_mean_squared_error",  # Use RMSE as the scoring metric
    cv=5,  # 5-fold cross-validation
    verbose=3,  # Print progress
    random_state=28,
    n_jobs=-1  # Use all available cores
)

# Fit the model
random_search.fit(X_train_cat, y_train, cat_features=categorical_cols)

# Best parameters
best_params = random_search.best_params_
print("Best Parameters:", best_params)


# In[258]:


best_params_catboost = {'verbose': False, 'random_strength': 0.5, 'learning_rate': 0.1, 'l2_leaf_reg': 1, 'iterations': 1000, 'depth': 8, 'border_count': 32, 'bagging_temperature': 1}
# best_params_catboost = {"bagging_temperature":0.5, "border_count":32, "depth":8, "iterations":1000, "l2_leaf_reg":3, "learning_rate":0.1, "random_strength":0.1, "verbose":False}
# Train the model with the best parameters
best_catboost_model = CatBoostRegressor(
    **best_params_catboost,
    random_state = 42
    # verbose=3  # Enable verbose output for training
)

best_catboost_model.fit(X_train_cat, y_train, cat_features=categorical_cols)

# Predictions
catboost_preds = best_catboost_model.predict(X_test_cat)

# Evaluate
from sklearn.metrics import mean_squared_error
catboost_rmse = np.sqrt(mean_squared_error(y_test, catboost_preds))
print(f"CatBoost RMSE: {catboost_rmse}")


# In[259]:


import math 
from sklearn.utils import shuffle
# Shuffle X_test and y_test
X_test, y_test = shuffle(X_test, y_test, random_state=42)

print("*" * 50)
y_test_pred_catboost = best_catboost_model.predict(X_test_cat)
print("TEST SET")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_test_pred_catboost))
print("Root Mean Squared Error (RMSE):", math.sqrt(mean_squared_error(y_test, y_test_pred_catboost)))
print("R-squared (R2):", r2_score(y_test, y_test_pred_catboost))

# Create a DataFrame with y_val and y_test_pred_catboost
results_df = pd.DataFrame({'y_test': y_test, 'y_pred': y_test_pred_catboost})
print(results_df)


# ### 2. LightGBM

# In[137]:


# Convert categorical columns to 'category' dtype
X_train_lgb = X_train.copy()
X_test_lgb = X_test.copy()
categorical_cols = X_train_lgb.select_dtypes(include=['object']).columns
for col in categorical_cols:
    X_train_lgb[col] = X_train_lgb[col].astype('category')
    X_test_lgb[col] = X_test_lgb[col].astype('category')

# Get the indices of categorical columns
categorical_indices = [X_train_lgb.columns.get_loc(col) for col in categorical_cols]

# Create LightGBM datasets with categorical features
lgb_train = lgb.Dataset(X_train_lgb, label=y_train, categorical_feature=categorical_indices)
lgb_test = lgb.Dataset(X_test_lgb, label=y_test, categorical_feature=categorical_indices, reference=lgb_train)


# In[130]:


from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np

# Define the parameter grid
param_grid = {
    "objective": ["regression"],
    "metric": ["rmse"],
    "boosting_type": ["gbdt", "dart"],
    "num_leaves": [31, 50, 100],
    "learning_rate": [0.01, 0.05, 0.1],
    "feature_fraction": [0.8, 0.9, 1.0],
    "bagging_fraction": [0.8, 0.9, 1.0],
    "bagging_freq": [0, 5, 10],
    "min_data_in_leaf": [20, 50, 100],
    "lambda_l1": [0, 0.1, 0.5],
    "lambda_l2": [0, 0.1, 0.5],
    "random_state": [28],
    "shrinkage_rate": [0.01, 0.012, 0.015]
}

# Create the LightGBM regressor
lgb_model = LGBMRegressor()

# RandomizedSearchCV
random_search_lgb = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=param_grid,
    n_iter=50,  # Number of parameter settings to sample
    scoring='neg_root_mean_squared_error',  # Use RMSE as the scoring metric
    cv=5,  # 5-fold cross-validation
    verbose=1,
    random_state=28,
    n_jobs=-1  # Use all available cores
)

# Fit the model
random_search_lgb.fit(X_train, y_train)

# Best parameters
best_params_lgb = random_search_lgb.best_params_
print("Best Parameters:", best_params_lgb)


# In[135]:


best_params_lgb


# In[138]:


# Train the model with the best parameters
lgb_model_best = LGBMRegressor(**best_params_lgb)
lgb_model_best.fit(X_train_lgb, y_train)


# In[139]:


import math 
from sklearn.utils import shuffle
# Shuffle X_test and y_test
X_test, y_test = shuffle(X_test, y_test, random_state=42)

print("*" * 50)
y_test_pred_lgb = lgb_model_best.predict(X_test)
print("TEST SET")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_test_pred_lgb))
print("Root Mean Squared Error (RMSE):", math.sqrt(mean_squared_error(y_test, y_test_pred_lgb)))
print("R-squared (R2):", r2_score(y_test, y_test_pred_lgb))

# Create a DataFrame with y_val and y_test_pred_lgb
results_df = pd.DataFrame({'y_test': y_test, 'y_pred': y_test_pred_lgb})
print(results_df)


# ### 3. Stacking RF + CatBoost

# #### a. Use LR as a correlation mechanism

# In[187]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Step 1: Train all models (assuming you already have trained models)
rf_preds = random_forest_model.predict(X_test)
cb_preds = best_catboost_model.predict(X_test_cat)

# Step 2: Create a new dataset using model predictions as features
stacked_features = np.column_stack((rf_preds, cb_preds))

# Step 3: Train the meta-model (using RF predictions as the main guidance)
meta_model = LinearRegression()
meta_model.fit(stacked_features, y_test)

# Step 4: Predict using the meta-model
stacked_preds = meta_model.predict(stacked_features)

# Step 5: Ensure non-negative predictions (since vehicle population can't be negative)
stacked_preds = np.maximum(stacked_preds, 0)

# Step 6: Evaluate performance
print("*" * 50)
rmse_stacked = np.sqrt(mean_squared_error(y_test, stacked_preds))
print("TEST SET")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, stacked_preds))
print("Root Mean Squared Error (RMSE):", rmse_stacked)
print("R-squared (R2):", r2_score(y_test, stacked_preds))

# Create a DataFrame with y_val and stacked_preds
results_df = pd.DataFrame({'y_test': y_test, 'y_pred': stacked_preds})
print(results_df)


# ### 2. Assign weights

# In[201]:


import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Generate predictions from base models
rf_preds = random_forest_model.predict(X_test)
cb_preds = best_catboost_model.predict(X_test_cat)

# Step 2: Apply weighted averaging
rf_weight = 0.9 # Give more importance to Random Forest
cb_weight = 0.1  # Less weight to CatBoost

stacked_preds = (rf_weight * rf_preds) + (cb_weight * cb_preds)

# Step 3: Ensure non-negative predictions
stacked_preds = np.maximum(stacked_preds, 0)

# Step 4: Evaluate performance
rmse_stacked = np.sqrt(mean_squared_error(y_test, stacked_preds))
mae_stacked = mean_absolute_error(y_test, stacked_preds)
r2_stacked = r2_score(y_test, stacked_preds)

# Print evaluation metrics
print("*" * 50)
print("TEST SET (Weighted Averaging)")
print(f"Mean Absolute Error (MAE): {mae_stacked}")
print(f"Root Mean Squared Error (RMSE): {rmse_stacked}")
print(f"R-squared (R2): {r2_stacked}")

# Step 5: Show predictions
results_df = pd.DataFrame({'y_test': y_test, 'y_pred': stacked_preds})
print(results_df)
# Plot the actual vs predicted values
plt.figure(figsize=(8, 8))
plt.scatter(y_test, stacked_preds, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Vehicle Population')
plt.grid(True)
plt.show()


# #### 3. 2 RF + 1 Catboost -> LR

# In[179]:


# Train second Random Forest model with more trees
rf_params = {'n_estimators': 500, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': False}
random_forest_model_2 = RandomForestRegressor(random_state=28, **rf_params)
random_forest_model_2.fit(X_train, y_train)


# In[200]:


import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Generate predictions from base models
rf_preds_1 = random_forest_model.predict(X_test)  # First Random Forest model
rf_preds_2 = random_forest_model_2.predict(X_test)  # Second Random Forest model (with different hyperparameters)
cb_preds = best_catboost_model.predict(X_test_cat)  # CatBoost predictions

# Step 2: Apply weighted averaging
rf_weight = 0.75  # Give more importance to Random Forest
rf_2_weight = 0.25  # Slightly lower weight to the second Random Forest
cb_weight = 0.00  # Less weight to CatBoost

# # Combining predictions using weighted averaging
# stacked_preds = (rf_weight * rf_preds_1) + (rf_2_weight * rf_preds_2) + (cb_weight * cb_preds)

# # Step 3: Ensure non-negative predictions (as vehicle population can't be negative)
# stacked_preds = np.maximum(stacked_preds, 0)
# Step 2: Create a new dataset using model predictions as features
stacked_features = np.column_stack((rf_preds_1, rf_preds_2, cb_preds))

# Step 3: Train the meta-model (using RF predictions as the main guidance)
meta_model = LinearRegression()
meta_model.fit(stacked_features, y_test)

# Step 4: Predict using the meta-model
stacked_preds = meta_model.predict(stacked_features)

# Step 5: Ensure non-negative predictions (since vehicle population can't be negative)
stacked_preds = np.maximum(stacked_preds, 0)
# Step 4: Evaluate performance
rmse_stacked = np.sqrt(mean_squared_error(y_test, stacked_preds))
mae_stacked = mean_absolute_error(y_test, stacked_preds)
r2_stacked = r2_score(y_test, stacked_preds)

# Print evaluation metrics
print("*" * 50)
print("TEST SET (Weighted Averaging with Two RF Models)")
print(f"Mean Absolute Error (MAE): {mae_stacked}")
print(f"Root Mean Squared Error (RMSE): {rmse_stacked}")
print(f"R-squared (R2): {r2_stacked}")

# Step 5: Show predictions
results_df = pd.DataFrame({'y_test': y_test, 'y_pred': stacked_preds})
print(results_df)
# Plot the actual vs predicted values
plt.figure(figsize=(8, 8))
plt.scatter(y_test, stacked_preds, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Vehicle Population')
plt.grid(True)
plt.show()
results_df.to_csv('/Users/trongphan/Downloads/Rice_Datathon/rice-datathon-2025/output/results.csv', index=False)


# #### 4. Stack 2rf + 1cat -> rf

# In[204]:


# Assuming CatBoost is already trained
# Step 2: Generate predictions on the test set
rf_preds_1 = random_forest_model.predict(X_test)  # First Random Forest model
rf_preds_2 = random_forest_model_2.predict(X_test)  # Second Random Forest model (with different hyperparameters)
cb_preds = best_catboost_model.predict(X_test_cat)  # CatBoost predictions

# Step 3: Stack the predictions from the base models

"""
First, I try with 2 RF only. The result is good, with ~1788 RMSE. Then, I try with 3 models: 2 RF and 1 CatBoost. The result is better.
* Reason: The CatBoost model provides additional information that can help improve the predictions and can detect extreme case (the vehicle population is very small, which is <10).
"""
# stacked_features = np.column_stack((rf_preds_1, rf_preds_2)) 
stacked_features = np.column_stack((rf_preds_1, rf_preds_2, cb_preds))

# Step 4: Train the meta-model (Random Forest)
meta_rf = RandomForestRegressor(random_state=28)
meta_rf.fit(stacked_features, y_test)

# Step 5: Make final predictions
stacked_preds = meta_rf.predict(stacked_features)

# Step 6: Evaluate performance
mae_stacked = mean_absolute_error(y_test, stacked_preds)
rmse_stacked = np.sqrt(mean_squared_error(y_test, stacked_preds))
print("Root Mean Squared Error (RMSE):", rmse_stacked)
# Print evaluation metrics
print("*" * 50)
print("TEST SET (Weighted Averaging with Two RF Models)")
print(f"Mean Absolute Error (MAE): {mae_stacked}")
print(f"Root Mean Squared Error (RMSE): {rmse_stacked}")
print(f"R-squared (R2): {r2_stacked}")

# Step 5: Show predictions
results_df = pd.DataFrame({'y_test': y_test, 'y_pred': stacked_preds})
print(results_df)
# Plot the actual vs predicted values
plt.figure(figsize=(8, 8))
plt.scatter(y_test, stacked_preds, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Vehicle Population')
plt.grid(True)
plt.show()
results_df.to_csv('/Users/trongphan/Downloads/Rice_Datathon/rice-datathon-2025/output/results.csv', index=False)


# **LET FUCKING GO!!!!!!!**

# 

# #### 4. Log trans

# In[ ]:




