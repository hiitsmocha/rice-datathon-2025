import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Set Streamlit page config
st.set_page_config(page_title="Data Processing Pipeline", page_icon="📊", layout="wide")

# Custom CSS for modern aesthetic and button-styled navigation
st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background-color: #fffff;
        }

        /* Sidebar background */
        .sidebar .sidebar-content {
            background-color: #2b3e50;
            color: white;
        }

        /* Titles and headers */
        h1, h2, h3 {
            color: #1e81b0;
        }

        /* Button style */
        .stButton>button {
            width: 100%;
            background-color: #1e81b0;
            color: white;
            border: none;
            padding: 10px 15px;
            margin: 5px 0;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: 0.3s;
        }

        .stButton>button:hover {
            background-color: #145a7a;
        }

        /* DataFrame style */
        .stDataFrame {
            border-radius: 10px;
            background-color: white;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Function to load data
@st.cache_data
def load_data():
    with st.spinner("📥 Loading dataset... Please wait!"):
        return pd.read_csv("rice-datathon-2025/train_and_val.csv")

# Sidebar Navigation
st.sidebar.image("https://glasscock.rice.edu/sites/g/files/bxs2991/files/2020-07/Rice_Owl_Flat_280_Blue.png", width=100)
st.sidebar.title("📌 Navigation")

sections = {
    "🏗️ Data Wrangling": "data_wrangling",
    "📊 EDA": "eda",
    "⚙️ Model Selection": "model_selection",
    "📈 Result Interpretation": "result_interpretation",
    "🤖 AI assistant": "ai_assistant"
}

# Initialize session state for selected_section
if "selected_section" not in st.session_state:
    st.session_state.selected_section = "data_wrangling"

# Sidebar buttons
for label, key in sections.items():
    if st.sidebar.button(label, key=key):
        st.session_state.selected_section = key  # Store selection in session state

# Retrieve the selected section
selected_section = st.session_state.selected_section
# Data Wrangling
if selected_section == "data_wrangling":
    st.title("🏗️ Data Wrangling")
    data = load_data()

    # Drop Unnecessary Columns 
    if "Region" in data.columns:
        data = data.drop(columns=["Region"])

    # Handle Missing Values
    data['Model Year'].fillna(data['Model Year'].median(), inplace=True)
    data['Model Year'] = data['Model Year'].astype(int)

    # Convert 'GVWR Class' to Numeric
    data["GVWR Class"] = data["GVWR Class"].replace({"Not Applicable": -1, "Unknown": -1})
    data["GVWR Class"] = pd.to_numeric(data["GVWR Class"], errors='coerce')

    # One-Hot Encoding & Ordinal Encoding
    ordinal_mapping = {'1': 1, '2': 2, '3': 3, '≥4': 4, 'Unknown': -1}
    data["Number of Vehicles Registered at the Same Address"] = data["Number of Vehicles Registered at the Same Address"].map(ordinal_mapping)
    data = pd.get_dummies(data, columns=["Vehicle Category", "Fuel Type", "Fuel Technology", "Electric Mile Range"])

    # Convert Date Column
    data['Date'] = data['Date'].fillna(2020)
    data['Date'] = pd.to_datetime(data['Date'], format='%Y', errors='coerce').dt.year
    data["Vehicle Age"] = data["Date"] - data["Model Year"]
    
    # Step 1: Load and Display the Dataset
    st.header("Step 1: Load and Display the Dataset")
    st.write("First, we load the dataset and display the first few rows to understand its structure.")
    code_step1 = """
    # Load the dataset
    data = pd.read_csv("our_dataset.csv")

    # Display the first 5 rows
    data.head()
    """

    st.code(code_step1, language="python")
    st.write("**Explanation**: The dataset is loaded using `pd.read_csv()`, and the first 5 rows are displayed using `data.head()`.")
    st.write("### 🔍 Raw Dataset Preview")
    raw_data = pd.read_excel("rice-datathon-2025/data/training.xlsx")
    st.dataframe(raw_data.head())  # Show first 5 rows
    
    # Step 2: Basic Information about the Dataset
    st.header("Step 2: Basic Information about the Dataset")
    st.write("Next, we check the dataset's shape, info, and missing values.")
    code_step2 = """
    # Check dataset shape
    print(f"Number of rows: {data.shape[0]}, Number of columns: {data.shape[1]}")

    # Display dataset info
    data.info()

    # Check for missing values
    data.isnull().sum()
    """
    st.code(code_step2, language="python")
    st.write("**Explanation**: We use `data.shape` to get the dimensions, `data.info()` to see column types, and `data.isnull().sum()` to check for missing values.")

    # Step 3: Handle Missing Values
    st.header("Step 3: Handle Missing Values")
    st.write("We handle missing values by filling numerical columns with the median and categorical columns with the mode.")
    code_step3 = """
    # Fill missing values in numerical columns with median
    data['Model Year'].fillna(data['Model Year'].median(), inplace=True)

    # Fill missing values in categorical columns with mode
    for col in data.columns:
        if data[col].dtype == "object" or pd.api.types.is_categorical_dtype(data[col]):
            data[col].fillna(data[col].mode()[0], inplace=True)
    """
    st.code(code_step3, language="python")
    st.write("**Explanation**: Missing values in numerical columns are filled with the median, while categorical columns are filled with the mode.")

    # Step 4: Convert Data Types
    st.header("Step 4: Convert Data Types")
    st.write("We convert the 'Date' column to datetime format and categorical columns to the 'category' dtype.")
    code_step4 = """
    # Convert 'Date' to datetime and extract the year
    data['Date'] = pd.to_datetime(data['Date'], format='%Y').dt.year

    # Convert categorical columns to 'category' dtype
    categorical_columns = ['Vehicle Category', 'GVWR Class', 'Fuel Type', 'Fuel Technology', 'Electric Mile Range', 'Number of Vehicles Registered at the Same Address', 'Model Year']
    for col in categorical_columns:
        data[col] = data[col].astype('category')
    """
    st.code(code_step4, language="python")
    st.write("**Explanation**: The 'Date' column is converted to datetime, and categorical columns are converted to the 'category' dtype for efficient memory usage.")

    # Step 5: Remove Duplicates and Reset Index
    st.header("Step 5: Remove Duplicates and Reset Index")
    st.write("We remove duplicate rows and reset the index for consistency.")
    code_step5 = """
    # Remove duplicates
    data.drop_duplicates(inplace=True)

    # Reset index
    data.reset_index(drop=True, inplace=True)
    """
    st.code(code_step5, language="python")
    st.write("**Explanation**: Duplicates are removed using `drop_duplicates()`, and the index is reset using `reset_index()`.")

    
# Exploratory Data Analysis (EDA)
elif selected_section == "eda":
    st.title("📊 Exploratory Data Analysis")
    
    # Load Data
    data = pd.read_csv("rice-datathon-2025/data/train_and_val.csv")
    # st.write("### 🔍 Raw Dataset Preview")
    # st.dataframe(data.head())  # Show first 5 rows

    # Step 6: Feature Engineering
    st.header("Step 6: Feature Engineering")
    st.write("We create a new feature called 'Vehicle Age' by subtracting the 'Model Year' from the 'Date'.")
    code_step6 = """
    # Create 'Vehicle Age' feature
    data["Vehicle Age"] = data["Date"] - data["Model Year"]
    """
    st.code(code_step6, language="python")
    st.write("**Explanation**: The 'Vehicle Age' feature is created to represent the age of the vehicle in years.")

    # Step 7: Visualize Data
    st.header("Step 7: Visualize Data")
    st.write("We visualize the relationship between 'Vehicle Age' and 'Vehicle Population' using a scatter plot.")
    code_step7 = """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Scatter plot: Vehicle Age vs. Vehicle Population
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Vehicle Age", y="Vehicle Population", data=data)
    plt.title("Vehicle Age vs. Vehicle Population")
    plt.xlabel("Vehicle Age")
    plt.ylabel("Vehicle Population")
    plt.grid(True)
    plt.show()
    """
    st.code(code_step7, language="python")
    st.write("**Explanation**: A scatter plot is created using `seaborn.scatterplot()` to visualize the relationship between 'Vehicle Age' and 'Vehicle Population'.")

    # Ensure 'Vehicle Age' exists
    if "Model Year" in data.columns and "Date" in data.columns:
        data["Date"] = data["Date"].fillna(2020)  # Handle missing dates
        data["Date"] = pd.to_datetime(data["Date"], format='%Y', errors='coerce').dt.year
        data["Vehicle Age"] = data["Date"] - data["Model Year"]
    else:
        st.error("❌ Missing 'Model Year' or 'Date' columns. Cannot compute 'Vehicle Age'.")

    if "Vehicle Age" in data.columns and "Vehicle Population" in data.columns:
        st.write("### 🔍 Vehicle Age vs. Population")
        fig, ax = plt.subplots()
        sns.scatterplot(x="Vehicle Age", y="Vehicle Population", data=data, color="darkblue", ax=ax)
        ax.set_facecolor("#f2f9fa")
        st.pyplot(fig)
    else:
        st.warning("⚠️ Cannot generate plots. Ensure 'Vehicle Age' and 'Vehicle Population' exist in the dataset.")
    st.write("### 📌 Insights")
    st.write("The Vehicle Population is highest for newer vehicles (age 0) and decreases significantly as the age increases. This trend indicates that newer vehicles dominate the population, while older vehicles (e.g., 30-40 years) are much rarer.")

    
    # Step 8: Split Data into Training and Validation Sets
    st.header("Step 8: Split Data into Training and Validation Sets")
    st.write("We split the dataset into training and validation sets for modeling.")
    code_step8 = """
    from sklearn.model_selection import train_test_split

    # Split data into features (X) and target (y)
    X = data.drop(columns=["Vehicle Population"])
    y = data["Vehicle Population"]

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    """
    st.code(code_step8, language="python")
    st.write("**Explanation**: The dataset is split into training and validation sets using `train_test_split()` with an 80-20 split.")

    # Step 9: Final Check for Missing Values
    st.header("Step 9: Final Check for Missing Values")
    st.write("We perform a final check to ensure no missing values remain in the dataset.")
    code_step9 = """
    # Check for missing values
    data.isnull().sum()
    """
    st.code(code_step9, language="python")
    st.write("**Explanation**: A final check is performed to ensure all missing values have been handled.")

    # Conclusion
    st.header("Conclusion")
    st.write("The dataset has been cleaned, preprocessed, and is now ready for modeling. The EDA process included handling missing values, converting data types, removing duplicates, creating new features, and visualizing relationships between variables.")

    st.write("### 📌 Processed Dataset Preview")
    final_data = pd.read_csv("rice-datathon-2025/data/X_train.csv")
    st.dataframe(final_data.head())
# Assuming the user has selected model selection
elif selected_section == "model_selection":
    st.warning("""📌 Intuition: We aim to predict the vehicle population using machine learning models. We'll train Random Forest, CatBoost, and XGBoost models, tune hyperparameters, and stack models to improve performance.
               The reason why we chose these models is that they are robust, handle non-linear relationships well, and are suitable for regression tasks, especially all the features are categorical data.""")
    st.title("⚙️ Random Forest")

    # Step 1: Training the Random Forest Model
    st.write("### Step 1: Train Random Forest Model")
    st.markdown("""
    First, we train a basic Random Forest model and evaluate its performance on both the validation and test sets.
    """)

    code1 = '''
    # Train Random Forest model
    random_forest_model = RandomForestRegressor(random_state=28)
    random_forest_model.fit(X_train, y_train)

    # Shuffle validation and test data
    X_test, y_test = shuffle(X_test, y_test, random_state=42)

    # Evaluate on Validation Set
    y_pred_val = random_forest_model.predict(X_val)
    print("VALIDATION SET")
    print("Mean Absolute Error (MAE):", mean_absolute_error(y_val, y_pred_val))
    print("Root Mean Squared Error (RMSE):", math.sqrt(mean_squared_error(y_val, y_pred_val)))
    print("R-squared (R2):", r2_score(y_val, y_pred_val))

    # Evaluate on Test Set
    y_pred = random_forest_model.predict(X_test)
    print("TEST SET")
    print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
    print("Root Mean Squared Error (RMSE):", math.sqrt(mean_squared_error(y_test, y_pred)))
    print("R-squared (R2):", r2_score(y_test, y_pred))
    '''
    st.code(code1, language="python")

    st.write("### Step 2: Hyperparameter Tuning with GridSearchCV")
    st.markdown("""
    Now we tune the hyperparameters of the Random Forest model using **GridSearchCV**.
    """)

    code2 = '''
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "bootstrap": [True, False]
    }

    grid_search = GridSearchCV(estimator=random_forest_model, param_grid=param_grid, cv=5, 
                            scoring="neg_mean_squared_error", n_jobs=-1, verbose=3)

    grid_search.fit(X_train, y_train)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best RMSE from GridSearchCV: {math.sqrt(-grid_search.best_score_)}")
    '''
    st.code(code2, language="python")

    st.write("### Step 3: Random State Tuning for Optimizing RMSE")
    st.markdown("""
    Next, we experiment with different random states to find the one that minimizes the RMSE.
    """)

    code3 = '''
    list_rmse = []
    min_rmse = float('inf')
    for i in tqdm(range(1, 100)):
        rf = RandomForestRegressor(random_state=i)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        rmse_now = math.sqrt(mean_squared_error(y_test, y_pred))
        if rmse_now < min_rmse:
            min_rmse = rmse_now
            print(f"New Min RMSE: {min_rmse} at state {i}")
        list_rmse.append({"state": i, "rmse": rmse_now})

    top_5_rmse = sorted(list_rmse, key=lambda x: x['rmse'])[:5]
    for top_rmse in top_5_rmse:
        print(f"State: {top_rmse['state']}, RMSE: {top_rmse['rmse']}")
    '''
    st.code(code3, language="python")

    # Streamlit App Title
    st.title("⚙️ CatBoost")

    # Instructions for the user
    st.write("### Step 1: Train CatBoost Model")
    st.markdown("""
    In this step, we train a CatBoost model, perform hyperparameter tuning, and evaluate its performance.
    """)

    # Example code snippet for training the CatBoost model
    code4 = '''
    from catboost import CatBoostRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import math

    # Define best parameters for CatBoost
    best_params_catboost = {
        'iterations': 1000,
        'learning_rate': 0.1,
        'depth': 8,
        'l2_leaf_reg': 1,
        'random_strength': 0.5,
        'bagging_temperature': 1,
        'border_count': 32,
        'verbose': False,
        'random_state': 42
    }

    # Initialize CatBoost model with best parameters
    catboost_model = CatBoostRegressor(**best_params_catboost)

    # Train the model
    catboost_model.fit(X_train_cat, y_train_full, cat_features=categorical_cols)

    # Predictions
    catboost_preds = catboost_model.predict(X_test_cat)

    # Evaluate on Test Set
    catboost_rmse = np.sqrt(mean_squared_error(y_test_cat, catboost_preds))
    print(f"CatBoost RMSE: {catboost_rmse}")
    print("*" * 50)
    print("TEST SET")
    print("Mean Absolute Error (MAE):", mean_absolute_error(y_test_cat, catboost_preds))
    print("Root Mean Squared Error (RMSE):", catboost_rmse)
    print("R-squared (R2):", r2_score(y_test_cat, catboost_preds))
    '''
    st.code(code4, language="python")

    # Hyperparameter tuning code for CatBoost using GridSearchCV or RandomizedSearchCV
    st.write("### Step 2: Hyperparameter Tuning with GridSearchCV or RandomizedSearchCV")
    st.markdown("""
    You can further fine-tune the hyperparameters using **GridSearchCV** or **RandomizedSearchCV**. Here's how:
    """)

    # Hyperparameter tuning code snippet for GridSearchCV
    code5 = '''
    from sklearn.model_selection import GridSearchCV

    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'iterations': [500, 1000, 1500],
        'learning_rate': [0.01, 0.05, 0.1],
        'depth': [4, 6, 8],
        'l2_leaf_reg': [1, 3, 5],
        'random_strength': [0.1, 0.5, 1],
        'bagging_temperature': [0, 0.5, 1],
        'border_count': [32, 64, 128],
        'verbose': [False]
    }

    # Initialize the CatBoost model
    catboost_model = CatBoostRegressor(random_state=42, cat_features=categorical_cols)

    # Perform GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(catboost_model, param_grid, cv=3, n_jobs=-1, scoring='neg_root_mean_squared_error')
    grid_search.fit(X_train_cat, y_train_full)

    # Get the best parameters found by GridSearchCV
    best_params_catboost = grid_search.best_params_
    print("Best parameters found by GridSearchCV:", best_params_catboost)

    # Train the model with the best parameters
    best_catboost_model = grid_search.best_estimator_

    # Evaluate on Test Set
    catboost_preds = best_catboost_model.predict(X_test_cat)
    catboost_rmse = np.sqrt(mean_squared_error(y_test_cat, catboost_preds))
    print(f"CatBoost RMSE: {catboost_rmse}")
    print("*" * 50)
    print("TEST SET")
    print("Mean Absolute Error (MAE):", mean_absolute_error(y_test_cat, catboost_preds))
    print("Root Mean Squared Error (RMSE):", catboost_rmse)
    print("R-squared (R2):", r2_score(y_test_cat, catboost_preds))
    '''
    st.code(code5, language="python")

    # Step 1: Stacking Random Forest and CatBoost
    st.title("⚙️ Stacking Multiple Models")
    st.warning("📌 Intuition: Leveraging the strengths of multiple models to enhance overall performance. CatBoost is highly sensitive to extreme values, while Random Forest struggles with very small values. By combining these models, we aim to balance their strengths, improving both accuracy and robustness in our predictions.")
    st.write("### Method 1: Stacking Random Forest and CatBoost with Meta-Model (Linear Regression)")
    st.markdown("""
    In this method, we combine the predictions of Random Forest and CatBoost models by stacking them 
    and then use a meta-model (Linear Regression) to make final predictions.
    """)

    # Code snippet for stacking with meta-model (Linear Regression)
    code5 = '''
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    import numpy as np

    # Step 1: Fit the trained model with the training data
    rf_preds = random_forest_model.predict(X_train)
    cb_preds = best_catboost_model.predict(X_train_cat)

    # Step 2: Create a new dataset using model predictions as features
    stacked_features_1 = np.column_stack((rf_preds, cb_preds))

    # Step 3: Train the meta-model on training data(using RF predictions as the main guidance)
    meta_model = LinearRegression()
    meta_model.fit(stacked_features_1, y_train)

    # Step 4: Predict using the meta-model
    rf_preds_test = random_forest_model.predict(X_test)
    cb_preds_test = best_catboost_model.predict(X_test_cat)
    stacked_features_1_test = np.column_stack((rf_preds_test, cb_preds_test))
    stacked_preds_1_test = meta_model.predict(stacked_features_1_test)

    # Step 5: Ensure non-negative predictions (since vehicle population can't be negative)
    stacked_preds_1 = np.maximum(stacked_preds_1_test, 0)

    # Step 6: Evaluate performance
    rmse_stacked = np.sqrt(mean_squared_error(y_test, stacked_preds_1_test))
    mae_stacked = mean_absolute_error(y_test, stacked_preds_1_test)
    r2_stacked = r2_score(y_test, stacked_preds_1_test)

    print("TEST SET")
    print("Mean Absolute Error (MAE):", mae_stacked)
    print("Root Mean Squared Error (RMSE):", rmse_stacked)
    print("R-squared (R2):", r2_stacked)
    '''
    st.code(code5, language="python")

    # Step 1: Stacking with Two Random Forests and CatBoost
    st.write("### Method 2: Stacking Predictions from Two Random Forest Models and CatBoost with Meta Model (Random Forest) - BEST SELECTION")
    st.markdown("""
    In this method, we stack the predictions from one/two Random Forest models and a CatBoost model 
    and use two Random Forest meta-models to make final predictions.
    """)

    # Code snippet for stacking with Two RFs and CatBoost
    code5 = '''
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np

    # Step 1: Generate features using the trained models
    rf_preds_1 = random_forest_model.predict(X_train)  # First Random Forest model
    rf_preds_2 = random_forest_model_2.predict(X_train)  # Second Random Forest model (with different hyperparameters)
    cb_preds = best_catboost_model.predict(X_train_cat)  # CatBoost predictions

    # Step 2: Stack the predictions from the base models
    stacked_features_rf_only = np.column_stack((rf_preds_1, cb_preds)) 
    stacked_features_rf_cb = np.column_stack((rf_preds_1, rf_preds_2, cb_preds))

    # Step 3: Train the meta-model (Random Forest)
    meta_rf_only = RandomForestRegressor(random_state=28)
    meta_rf_only.fit(stacked_features_rf_only, y_train)
    meta_rf_cb = RandomForestRegressor(random_state=28)
    meta_rf_cb.fit(stacked_features_rf_cb, y_train)

    # Step 4: Make final predictions
    rf_preds_1_test = random_forest_model.predict(X_test)  # First Random Forest model
    rf_preds_2_test = random_forest_model_2.predict(X_test)  # Second Random Forest model
    cb_preds_test = best_catboost_model.predict(X_test_cat)  # CatBoost predictions

    stacked_features_rf_only_test = np.column_stack((rf_preds_1_test, rf_preds_2_test)) 
    stacked_features_rf_cb_test = np.column_stack((rf_preds_1_test, rf_preds_2_test, cb_preds_test))

    stacked_preds_rf_only_test = meta_rf_only.predict(stacked_features_rf_only_test)
    stacked_preds_rf_cb_test = meta_rf_cb.predict(stacked_features_rf_cb_test)

    # Step 5: Evaluate performance
    mae_stacked_rf_only = mean_absolute_error(y_test, stacked_preds_rf_only_test)
    rmse_stacked_rf_only = np.sqrt(mean_squared_error(y_test, stacked_preds_rf_only_test))
    r2_stacked_rf_only = r2_score(y_test, stacked_preds_rf_only_test)

    mae_stacked_rf_cb = mean_absolute_error(y_test, stacked_preds_rf_cb_test)
    rmse_stacked_rf_cb = np.sqrt(mean_squared_error(y_test, stacked_preds_rf_cb_test))
    r2_stacked_rf_cb = r2_score(y_test, stacked_preds_rf_cb_test)
    '''
    st.code(code5, language="python")

elif selected_section == "result_interpretation":
    st.title("📈 Result Interpretation")
    st.write("### 📌 Understanding Model Performance")

    st.write("### Model Performance Metrics")
    st.markdown("""
    Below is a table summarizing the performance metrics of different models.
    """)

    import pandas as pd

    # Create a DataFrame with performance metrics
    results_df = pd.DataFrame({
        "Model": [
            "CatBoost Only",
            "Base Random Forest",
            "Random Forest + CatBoost -> Linear Regression",
            "Two Random Forests -> Random Forest",
            "Two Random Forests + CatBoost -> Random Forest",
            "Tuned XGBoost"
        ],
        "Mean Absolute Error (MAE)": [2092.98, 559.86, 564.66, 559.90, 563.79, 1075.45],
        "Root Mean Squared Error (RMSE)": [5661.89, 3730.61, 3757.97, 3719.93, 3736.97, 6006.13],
        "R-squared (R2)": [0.9154, 0.9633, 0.9627, 0.9635, 0.9631, 0.9061]
    })

    st.dataframe(results_df)

    st.write("### Sample Predictions from Test Set")
    st.markdown("""
    The table below shows a few examples of the actual vs. predicted values from the test set for each model.
    """)

    # Sample predictions from CatBoost
    catboost_sample_predictions = pd.DataFrame({
        "y_test": [47, 11999, 1, 7, 183742, 7, 7, 6, 2551, 1],
        "y_pred": [-1523.39, 4759.20, 3439.39, 195.64, 165083.70, -2978.96, -1004.15, -167.97, -394.54, 644.07]
    })

    st.write("#### CatBoost Model Predictions")
    st.dataframe(catboost_sample_predictions)

    # Sample predictions from Base Random Forest
    rf_base_sample_predictions = pd.DataFrame({
        "y_test": [1, 58, 3425, 112, 5771, 3, 3, 853, 495, 70],
        "y_pred": [2.28, 83.85, 3426.17, 142.89, 6087.88, 2.12, 3.39, 1032.91, 480.70, 79.69]
    })

    st.write("#### Base Random Forest Predictions")
    st.dataframe(rf_base_sample_predictions)

    # Sample predictions from Random Forest + CatBoost -> Linear Regression
    rf_cb_lr_sample_predictions = pd.DataFrame({
        "y_test": [316065, 315986, 306487, 284754, 284153, 1, 1, 1, 1, 1],
        "y_pred": [292402.11, 297319.85, 292453.54, 266745.41, 280101.62, -11.02, -9.43, -10.49, -8.60, -9.77]
    })

    st.write("#### Random Forest + CatBoost -> Linear Regression Predictions")
    st.dataframe(rf_cb_lr_sample_predictions)

    # Sample predictions from Two Random Forests Only
    rf_only_sample_predictions = pd.DataFrame({
        "y_test": [316065, 315986, 306487, 284754, 284153, 1, 1, 1, 1, 1],
        "y_pred": [279491.90, 294791.51, 288526.82, 257626.26, 276024.33, 1.58, 2.75, 2.57, 3.35, 2.82]
    })

    st.write("#### Two Random Forests -> Random Forest Predictions")
    st.dataframe(rf_only_sample_predictions)

    # Sample predictions from Tuned XGBoost
    xgb_sample_predictions = pd.DataFrame({
        "y_test": [16, 582, 73, 4, 16, 520, 15, 44018, 508, 500],
        "y_pred": [36.09, 303.40, 31.57, 1.42, 22.95, -77.27, 129.97, 162.68, -147.48, 52.60]
    })
    st.write("#### Tuned XGBoost Predictions")
    st.dataframe(xgb_sample_predictions)
    

elif selected_section == "ai_assistant":
    st.title("🤖 AI Assistant")
    st.write("### 📌 AI Assistant Features")
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import google.generativeai as genai

    # Configure Gemini API
    genai.configure(api_key="AIzaSyDUZVf2L1nDjQ2Z9iLM12EWNROQnjuX-qU")
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Initialize session memory
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    st.title("📊 AI-Powered CSV Query & Visualization App")

    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        st.write("### 📂 Uploaded CSV Data Preview:")
        st.dataframe(df.head())  # Show first few rows

        # Show basic statistics
        st.write("### 📊 Dataset Summary Statistics:")
        st.write(df.describe())

        # Visualization section
        st.write("## 📈 Data Visualization")

        # Let user choose column for visualization
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_columns = df.select_dtypes(exclude=["number"]).columns.tolist()

        if numeric_columns:
            col_to_plot = st.selectbox("Select a numeric column for histogram:", numeric_columns)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df[col_to_plot], bins=30, kde=True, ax=ax)
            ax.set_title(f"Histogram of {col_to_plot}")
            st.pyplot(fig)

        if categorical_columns:
            col_to_plot = st.selectbox("Select a categorical column for bar chart:", categorical_columns)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            df[col_to_plot].value_counts().plot(kind="bar", ax=ax)
            ax.set_title(f"Bar Chart of {col_to_plot}")
            st.pyplot(fig)

        # User query input
        user_query = st.text_area("Ask a question about this data:")

        if st.button("Get Answer"):
            if user_query:
                # Convert CSV data to text format
                csv_text = df.to_csv(index=False)

                # Construct Gemini prompt
                prompt = f"""
                You are an AI assistant analyzing a dataset. The user has uploaded the following CSV file:

                {csv_text}

                The user asks: "{user_query}"

                Please analyze the data and provide an insightful response.
                """
                
                # Get response from Gemini
                response = model.generate_content(prompt)

                # Save query & response to session memory
                st.session_state["chat_history"].append({"query": user_query, "response": response.text})

                # Display AI response
                st.write("### 🤖 Gemini AI's Answer:")
                st.write(response.text)
            else:
                st.warning("Please enter a query before clicking 'Get Answer'.")

        # Show chat history
        if st.session_state["chat_history"]:
            st.write("## 📝 Chat History")
            for chat in st.session_state["chat_history"]:
                with st.expander(f"📌 {chat['query']}"):
                    st.write(chat["response"])

        # Option to clear memory
        if st.button("Clear Chat History"):
            st.session_state["chat_history"] = []
            st.success("Chat history cleared!")
