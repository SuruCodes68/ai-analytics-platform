import streamlit as st
import pandas as pd
import io
import re
import contextlib
import google.generativeai as genai
import numpy as np

# --- Imports for our hard-coded baseline model ---
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

# --- IMPORTS FOR THE AI'S SANDBOX ---
# We import everything the AI might need, so it doesn't have to.
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.compose import make_column_selector
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor


# --- Page Configuration ---
st.set_page_config(page_title="My Data Analytics Platform", layout="wide")
st.title("🚀 AI-Powered Data Analytics Platform")

# --- Function to Extract Python Code ---
def extract_python_code(markdown_text):
    """
    Finds the first Python code block in a markdown string.
    """
    match = re.search(r'```python\n(.*?)```', markdown_text, re.DOTALL)
    if match:
        return match.group(1)
    return None

# --- Function to Run the Extracted Code (UPDATED) ---
def run_model_code(code_to_run, df, X_train, X_test, y_train, y_test):
    """
    Executes the AI's code to get y_pred.
    Returns the y_pred array or an error string.
    """
    try:
        # Define the local environment for the exec() call
        # We pass in *all* the tools the AI is allowed to use.
        local_vars = {
            'df': df,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'pd': pd,
            'np': np,
            'Pipeline': Pipeline,
            'SimpleImputer': SimpleImputer,
            'StandardScaler': StandardScaler,
            'OneHotEncoder': OneHotEncoder,
            'ColumnTransformer': ColumnTransformer,
            'TransformedTargetRegressor': TransformedTargetRegressor,
            'make_column_selector': make_column_selector, 
            'XGBRegressor': XGBRegressor, 
            'RandomForestRegressor': RandomForestRegressor 
        }
        
        # This exec call is now powerful enough to run custom cleaning functions
        # and then the model pipeline, as long as it defines 'y_pred' at the end.
        exec(code_to_run, local_vars, local_vars)
        
        y_pred = local_vars.get('y_pred')
        
        if y_pred is None:
            return None, "Error: The AI's code ran but did not define a 'y_pred' variable."
            
        return y_pred, "Success"
        
    except Exception as e:
        return None, f"An error occurred while running the AI's code:\n{e}"

# --- Hard-Coded Baseline Model Function ---
# This function will NOT be changed. Its failure on this dataset is
# the *trigger* for the AI's advanced cleaning.
def run_baseline_model(df):
    """
    Runs a robust Ridge Regression pipeline and returns the metrics.
    This serves as our "Baseline Model" for every run.
    """
    st.info("Running Baseline Model (Ridge Regression)...")
    
    try:
        # Try to intelligently find the target column
        potential_targets = ['charges', 'price', 'saleprice', 'revenue', 'score', 'target']
        target_col = [col for col in df.columns if col.lower() in potential_targets]
        
        if not target_col:
            # Fallback: use the last numeric column
            numeric_cols = df.select_dtypes(include=np.number).columns
            # Attempt to drop 'id' columns
            if 'id' in numeric_cols.str.lower():
                numeric_cols = numeric_cols.drop(numeric_cols[numeric_cols.str.lower() == 'id'][0])
                
            if len(numeric_cols) > 0:
                target_col = numeric_cols[-1]
            else:
                return "Error: Could not identify a numeric target variable.", None, None
        else:
            target_col = target_col[0] # Use the first match
        
        # --- NEW: Handle the target column for the Used Car dataset ---
        # The 'price' column is also dirty (e.g., "1,50,000")
        if target_col.lower() == 'price' and df[target_col].dtype == 'object':
            y = pd.to_numeric(df[target_col].str.replace(r'[,\D]', '', regex=True), errors='coerce')
            y = y.fillna(y.median())
        else:
            y = df[target_col]

        X = df.drop(target_col, axis=1)
        
        st.write(f"Inferred Target Variable: `{target_col}`")
        
        # Automatically find numeric and categorical features
        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
        
        # Create robust preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine pipelines into a single preprocessor
        # THIS PREPROCESSOR WILL FAIL on "1248 CC" by treating it as a category.
        # This is the intended behavior for our "dumb" baseline.
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', Ridge(alpha=1.0)) 
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_pipeline.fit(X_train, y_train)

        y_pred = model_pipeline.predict(X_test)
        
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results = { "model": "Baseline (Ridge Regression)", "target": target_col, "rmse": rmse, "r2": r2 }
        results_str = f"Target: {target_col}\nRMSE: ${rmse:,.2f}\nR-squared: {r2:.4f}"
        
        return results_str, results, (X_train, X_test, y_train, y_test)

    except Exception as e:
        return f"Error running baseline model: {e}", None, None

# --- AI Function (Phase 13 Upgrade!) ---
def get_ai_suggestion(df, baseline_results_str, target_col):
    """
    Sends the data summary AND baseline results to Gemini.
    Asks for a *better* model that includes CUSTOM DATA CLEANING.
    """
    st.info("AI is analyzing baseline and finding a 'Champion' model...")
    
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    # --- NEW: Add a hint about dirty columns for the AI ---
    dirty_cols_examples = []
    for col in df.select_dtypes(include='object').columns:
        try:
            # Find a value with a space, like "1248 CC"
            example_value = df[col].dropna().str.split().str[0].iloc[0]
            if pd.to_numeric(example_value, errors='coerce') is not np.nan:
                 dirty_cols_examples.append(f"'{col}' (e.g., \"{df[col].dropna().iloc[0]}\")")
        except:
            pass # Ignore errors

    dirty_col_hint = "No obvious dirty text-numeric columns found."
    if dirty_cols_examples:
        dirty_col_hint = f"Hint: The baseline may have failed due to 'dirty' text columns that need cleaning (e.g., {', '.join(dirty_cols_examples)})."

    summary = f"""
    Here is a summary of the dataset:
    1. DataFrame.info():
    {info_str}
    2. Potential 'Dirty' Columns (Text that might need cleaning):
    {dirty_col_hint}
    """
    
    system_prompt = f"""
    You are an expert Data Scientist. Your job is to analyze the following dataset summary and beat my baseline model.

    **My Baseline Model (Ridge Regression) Results:**
    {baseline_results_str} 
    
    **CRITICAL ANALYSIS:** The baseline R-squared is very low (or negative). This is because the baseline's simple preprocessor cannot handle 'dirty' text-numeric columns (like a column 'engine' having values '1248 CC' or 'mileage' having '23.4 kmpl'). The baseline model is treating these as categorical features, which is wrong.

    **Your Task: Provide Champion Code**
    Provide the *one* complete, executable Python code block that does ALL of the following:

    1.  **DEFINE A CUSTOM CLEANING FUNCTION:** You **MUST** define a custom function (e.g., `def custom_clean(df): ...`) that:
        * Takes a DataFrame (`df_in`) as input and returns a *new, cleaned* DataFrame.
        * Creates *new* numeric columns from the "dirty" text columns (e.g., `df_out['engine_clean'] = pd.to_numeric(df_in['engine'].str.split(' ').str[0], errors='coerce')`).
        * It **must** handle missing values created during cleaning (e.g., `.fillna(0)` or `.fillna(df_out['engine_clean'].median())`).
        * It should return the cleaned DataFrame.

    2.  **APPLY THE CLEANING:**
        * `X_train_cleaned = custom_clean(X_train)`
        * `X_test_cleaned = custom_clean(X_test)`

    3.  **BUILD YOUR PIPELINE:**
        * Define your `ColumnTransformer` (preprocessor) to run on the `X_train_cleaned` data. 
        * **CRITICAL:** Your `make_column_selector` should select the *new clean columns* (e.g., 'engine_clean') and *not* the original dirty ones (e.g., 'engine').
        * Your `ColumnTransformer` must handle *both* the *newly cleaned* numeric columns and any *original* numeric/categorical columns.

    4.  **DEFINE YOUR MODEL:**
        * Define an `XGBRegressor` or `RandomForestRegressor`.

    5.  **WRAP YOUR MODEL (if needed):**
        * If the target variable (`y_train`) is skewed, you **MUST** use `TransformedTargetRegressor` to wrap your entire model pipeline.
    
    6.  **FIT AND PREDICT (FINAL STEP):**
        * Fit the final pipeline: `model_pipeline.fit(X_train_cleaned, y_train)`
        * Create the final prediction: `y_pred = model_pipeline.predict(X_test_cleaned)`

    **OTHER CRITICAL RULES:**
    - You are given: `X_train`, `X_test`, `y_train`, `y_test`, `df`.
    - **DO NOT** use `train_test_split`.
    - **DO NOT WRITE ANY IMPORT STATEMENTS.** All necessary libraries are already available to you.
    - Your code's *only* output must be the `y_pred` variable.
    """
    
    try:
        model = genai.GenerativeModel('gemini-pro-latest') 
        response = model.generate_content([system_prompt, summary])
        return response.text
    except Exception as e:
        st.error(f"An error occurred while calling the Gemini API: {e}")
        return None

# --- Main App Function ---
def main_app():
    st.write("Upload a CSV, get EDA, and run a model tournament.")
    uploaded_file = st.file_uploader("Choose a .csv file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded and loaded successfully!")
            st.session_state.df = df

            st.header("Phase 1: Automated EDA")
            tab1, tab2, tab3 = st.tabs(["Data Summary", "Column Details", "Visuals"])
            with tab1:
                st.subheader("DataFrame Summary")
                # Drop non-numeric columns for describe() to avoid errors
                st.dataframe(df.describe(include=np.number))
            with tab2:
                st.subheader("Column Details")
                buffer = io.StringIO()
                df.info(buf=buffer)
                s = buffer.getvalue()
                st.text(s)
            
            with tab3:
                st.subheader("Basic Visualizations")
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                if not numeric_cols:
                    st.write("No numeric columns found to visualize.")
                else:
                    col_to_plot = st.selectbox("Select a numeric column", numeric_cols, key="eda_visuals")
                    
                    if col_to_plot:
                        st.markdown("---")
                        hist_data = df[col_to_plot].value_counts().reset_index()
                        hist_data.columns = [col_to_plot, 'Count'] 
                        sort_desc = st.checkbox("Sort by Count (Descending)", value=True)
                        if sort_desc:
                            hist_data = hist_data.sort_values(by='Count', ascending=False)
                        else:
                            hist_data = hist_data.sort_values(by=col_to_plot, ascending=True)

                        view_mode = st.radio("Select View Mode", ("Bar Chart", "Data Table"), horizontal=True)

                        if view_mode == "Bar Chart":
                            st.bar_chart(hist_data, x=col_to_plot, y='Count')
                        else:
                            st.dataframe(hist_data, use_container_width=True)

            # --- UPDATED TITLE ---
            st.header("Phase 2: AI-Powered Model Tournament")
            
            if st.button("🤖 Run Model Tournament"):
                if 'df' in st.session_state:
                    
                    # 1. Run Baseline Model
                    baseline_results_str, baseline_metrics, data_split = run_baseline_model(st.session_state.df)
                    
                    st.subheader("Model Results")
                    st.markdown("---")
                    st.markdown("### 1. Baseline Model (Ridge Regression)")
                    
                    if baseline_metrics and data_split:
                        st.code(baseline_results_str, language='text')
                        
                        X_train, X_test, y_train, y_test = data_split
                        target_col = baseline_metrics['target']
                        
                        # 2. Get AI Champion Model
                        suggestion_text = get_ai_suggestion(st.session_state.df, baseline_results_str, target_col)
                        
                        st.markdown("### 2. AI Champion Model (with Custom Cleaning)")
                        
                        if suggestion_text:
                            code = extract_python_code(suggestion_text)
                            
                            if code:
                                # 3. Run AI Champion Model
                                y_pred_ai, status_msg = run_model_code(
                                    code, st.session_state.df, 
                                    X_train, X_test, y_train, y_test
                                )
                                
                                if y_pred_ai is not None:
                                    # 4. Calculate and display AI model metrics
                                    r2_ai = r2_score(y_test, y_pred_ai)
                                    rmse_ai = root_mean_squared_error(y_test, y_pred_ai)
                                    
                                    results_str_ai = f"RMSE: ${rmse_ai:,.2f}\nR-squared: {r2_ai:.4f}"
                                    st.code(results_str_ai, language='text')
                                    
                                    code_pattern = r'```python\n(.*?)```'
                                    cleaned_suggestion_text = re.sub(code_pattern, 
                                                                     "\n\n*(Code for this model is in the expander below)*\n", 
                                                                     suggestion_text, 
                                                                     count=1,
                                                                     flags=re.DOTALL)

                                    with st.expander("See AI's full analysis"):
                                        st.markdown(cleaned_suggestion_text)
                                    with st.expander("See AI's Champion Model Code"):
                                        st.code(code, language='python')
                                else:
                                    st.error(status_msg)
                            else:
                                st.error("The AI's response did not contain a runnable code block.")
                    else:
                        st.error(baseline_results_str)
                else:
                    st.error("Please upload a file first.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("Awaiting for a CSV file to be uploaded.")


# --- App Gating Logic ---
st.sidebar.header("Configuration")
st.sidebar.write("To use this app, you need a Gemini API key.")
st.sidebar.markdown("Get one from [Google AI Studio](https://aistudio.google.com/app/apikey).")
api_key = st.sidebar.text_input("Enter your Gemini API key:", type="password", help="Your key is not stored.")

if api_key:
    try:
        genai.configure(api_key=api_key)
        st.sidebar.success("API Key configured successfully!")
        main_app()
    except Exception as e:
        st.sidebar.error("Invalid API Key. Please check your key and try again.")
        st.info("Please enter a valid Gemini API key in the sidebar to use the app.")
else:
    st.info("Please enter your Gemini API key in the sidebar to start the app.")