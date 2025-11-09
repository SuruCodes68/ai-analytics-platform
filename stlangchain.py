import streamlit as st
import pandas as pd
import io
import re
import contextlib
# import google.generativeai as genai  <-- WE NO LONGER NEED THIS
import numpy as np

# --- Imports for our hard-coded baseline model ---
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
# --- FIX: Added OrdinalEncoder and PowerTransformer to the import list ---
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

# --- IMPORTS FOR THE AI'S SANDBOX ---
# We import everything the AI might need, so it doesn't have to.
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.compose import make_column_selector
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# --- NEW: Imports for LangChain Manual Chat Mode ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # <-- This is the fix
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage # <-- Import SystemMessage


# --- Page Configuration ---
st.set_page_config(page_title="My Data Analytics Platform", layout="wide")
# 1. CHANGED EMOJI
st.title("✨ AI-Powered Data Analytics Platform")

# 3. INITIALIZE SESSION STATE FOR USER NOTES
# This will persist across reruns and won't be cleared by the 'Run Tournament' button
if 'user_notes' not in st.session_state:
    st.session_state.user_notes = ""
    
# --- CRITICAL BUG FIX: Initialize other state variables ---
if 'current_file_name' not in st.session_state:
    st.session_state.current_file_name = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_code' not in st.session_state:
    st.session_state.current_code = ""
if 'data_split' not in st.session_state:
    st.session_state.data_split = None
if 'reasoning' not in st.session_state:
    st.session_state.reasoning = ""
if 'dropped' not in st.session_state:
    st.session_state.dropped = ""
if 'target_col' not in st.session_state:
    st.session_state.target_col = ""


# --- Function to Extract Python Code ---
def extract_python_code(markdown_text):
    """
    Finds the first Python code block in a markdown string.
    """
    # Look for the section starting with Champion Code, then find the python block
    code_match = re.search(r"\*\*4\. Champion Code:\*\*(.*?)```python\n(.*?)\n```", markdown_text, re.DOTALL)
    if code_match:
        return code_match.group(2).strip() # Return the code block
    
    # Fallback for old prompt or manual refinement
    match = re.search(r'```python\n(.*?)```', markdown_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

# --- Function to Run the Extracted Code (UPDATED SANDBOX) ---
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
            'OrdinalEncoder': OrdinalEncoder, # --- FIX: Added OrdinalEncoder to the sandbox ---
            'PowerTransformer': PowerTransformer, # --- FIX: Added PowerTransformer to the sandbox ---
            'ColumnTransformer': ColumnTransformer,
            'TransformedTargetRegressor': TransformedTargetRegressor,
            'make_column_selector': make_column_selector, 
            'XGBRegressor': XGBRegressor, 
            'RandomForestRegressor': RandomForestRegressor,
            'LGBMRegressor': LGBMRegressor,
            'CatBoostRegressor': CatBoostRegressor
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

# --- Hard-Coded Baseline Model Function (UPDATED) ---
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
        
        # Handle the target column for the Used Car dataset
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
        
        # --- NEW: Calculate dropped columns ---
        all_features = X.columns.tolist()
        used_features = numeric_features + categorical_features
        dropped_features = [col for col in all_features if col not in used_features]
        # --- END NEW ---

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
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop' # This is where the dropping happens
        )

        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', Ridge(alpha=1.0)) 
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_pipeline.fit(X_train, y_train)

        y_pred = model_pipeline.predict(X_test)
        
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results = { 
            "model": "Baseline (Ridge Regression)", 
            "target": target_col, 
            "rmse": rmse, 
            "r2": r2,
            "dropped_features": dropped_features # Add dropped features to results
        }
        results_str = f"Target: {target_col}\nRMSE: ${rmse:,.2f}\nR-squared: {r2:.4f}"
        
        return results_str, results, (X_train, X_test, y_train, y_test)

    except Exception as e:
        return f"Error running baseline model: {e}", None, None

# --- AI Function (REWRITTEN TO USE LANGCHAIN) ---
def get_ai_suggestion(df, baseline_results_str, target_col, api_key):
    """
    Sends the data summary AND baseline results to Gemini using LangChain.
    Asks for a *better* model that includes CUSTOM DATA CLEANING.
    """
    st.info("AI is analyzing baseline and finding a 'Champion' model...")
    
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    # --- Add a hint about dirty columns for the AI ---
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
    
    # --- UPDATED PROMPT FOR FULL TOURNAMENT + TRANSPARENSY ---
    # MOVED THE system_prompt DEFINITION OUTSIDE of the 'summary' string
    system_prompt = f"""
    You are an expert Data Scientist. Your job is to analyze the following dataset summary and beat my baseline model.

    **My Baseline Model (Ridge Regression) Results:**
    {baseline_results_str} 
    
    **CRITICAL ANALYSIS (IF APPLICABLE):**
    The baseline R-squared might be low. This can happen if the baseline's simple preprocessor cannot handle 'dirty' text-numeric columns (like a column 'engine' having values '1248 CC' or 'mileage' having '23.4 kmpl'). The baseline model treats these as categorical, which can be wrong.
    
    **YOUR DATASET HINTS:**
    - `df.info()`: (see user message)
    - `Potential 'Dirty' Columns`: {dirty_col_hint}

    **Your Task:**
    Analyze the hints above. Then, provide your response in **exactly** four parts:

    **1. Champion Model Choice:**
    (State the champion model you chose from this list: `XGBRegressor`, `RandomForestRegressor`, `LGBMRegressor`, or `CatBoostRegressor`.)

    **2. Reasoning:**
    (Briefly explain your model choice. **Crucially, state if you are performing a custom cleaning step.** - If you see 'dirty' columns in the hint, explain how your cleaning function will fix them.
    - If the `dirty_col_hint` is "No obvious dirty text-numeric columns found," state that no custom cleaning is needed and you will just use a more powerful model with the standard preprocessor.)

    **3. Dropped Columns:**
    (You MUST state which original columns from the `X_train` dataframe were *not* used in your final model and provide a brief justification for each, e.g., "- `Name`: Dropped, because I engineered a `Brand` feature from it." or "- `Unnamed: 0`: Dropped, it appears to be an index.")

    **4. Champion Code:**
    Provide the *one* complete, executable Python code block that does ALL of the following:

    1.  **DEFINE A CUSTOM CLEANING FUNCTION (IF NEEDED):** * **If (and only if)** you identified 'dirty' text columns in your reasoning, you **MUST** define a custom function (e.g., `def custom_clean(df): ...`) that creates new numeric columns from them.
        * **If no cleaning is needed,** DO NOT define a `custom_clean` function.

    2.  **APPLY THE CLEANING (IF NEEDED):**
        * If you defined a function:
            * `X_train_cleaned = custom_clean(X_train)`
            * `X_test_cleaned = custom_clean(X_test)`
        * If you did not define a function:
            * `X_train_cleaned = X_train`
            * `X_test_cleaned = X_test`

    3.  **BUILD YOUR PIPELINE:**
        * Define your `ColumnTransformer` (preprocessor) to run on the `X_train_cleaned` data. 
        * **CRITICAL:** Your preprocessor must correctly handle ALL numeric and categorical columns you intend to use. If you created *new* clean columns (e.g., 'engine_clean'), select those. If you are using the *original* columns (e.g., 'age', 'sex', 'smoker'), select those.

    4.  **DEFINE YOUR MODEL:**
        * Define your chosen champion: `XGBRegressor`, `RandomForestRegressor`, `LGBMRegressor`, or `CatBoostRegressor`.

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
        # --- THIS IS THE FIX ---
        # We now use LangChain for this call, which is compatible with our new libraries
        llm = ChatGoogleGenerativeAI(model="gemini-pro-latest", google_api_key=api_key)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=summary)
        ]
        
        response = llm.invoke(messages)
        return response.content # LangChain responses are in the .content attribute
    except Exception as e:
        st.error(f"An error occurred while calling the Gemini API: {e}")
        return None

# --- Main App Function ---
def main_app(api_key): # <-- Pass in the api_key
    st.write("Upload a CSV (up to 200MB), get EDA, and run a model tournament.")
    uploaded_file = st.file_uploader("Choose a .csv file", type="csv")

    if uploaded_file is not None:
        # --- THIS IS THE FIX: The 'try' block must wrap ALL app logic ---
        try: # <--- THIS IS THE 'try' BLOCK (around line 377)
            
            # --- CRITICAL BUG FIX: Check for new file upload and clear state if it's different ---
            # --- This logic is now INSIDE the 'uploaded_file is not None' block ---
            if st.session_state.current_file_name != uploaded_file.name:
                st.session_state.current_file_name = uploaded_file.name
                # Clear all model and chat related session state
                st.session_state.messages = []
                st.session_state.current_code = ""
                st.session_state.data_split = None
                st.session_state.reasoning = ""
                st.session_state.dropped = ""
                st.session_state.target_col = ""
                st.success(f"New file loaded: {uploaded_file.name}. All model history has been reset.")
                # We keep 'user_notes' as they are personal and not file-specific
            
            # Limit file size (200MB)
            if uploaded_file.size > 200 * 1024 * 1024:
                st.error("File size exceeds 200MB. Please upload a smaller file.")
                return

            df = pd.read_csv(uploaded_file)
            st.session_state.df = df # We still need to load the df on every run

            # --- Only show this on the first load for a new file ---
            if not st.session_state.data_split: 
                st.success("File uploaded and loaded successfully!")

            st.header("Phase 1: Automated EDA")
            tab1, tab2, tab3 = st.tabs(["Data Summary", "Column Details", "Visuals"])
            with tab1:
                st.subheader("DataFrame Summary")
                # Describe all columns, including non-numeric
                st.dataframe(df.describe(include='all').T)
            with tab2:
                st.subheader("Column Details")
                buffer = io.StringIO()
                df.info(buf=buffer)
                s = buffer.getvalue()
                st.text(s)
            
            # --- "SMART" VISUALS TAB ---
            with tab3:
                st.subheader("Column Visualizer")
                
                # Dropdown to select ANY column
                all_columns = df.columns.tolist()
                col_to_plot = st.selectbox("Select a column to visualize", all_columns)

                if col_to_plot:
                    st.markdown("---")
                    col_data = df[col_to_plot]
                    dtype = col_data.dtype
                    n_unique = col_data.nunique()

                    st.markdown(f"**Column:** `{col_to_plot}` | **Type:** `{dtype}` | **Unique Values:** `{n_unique}`")

                    # --- Logic to decide which plot to show ---

                    # 1. If column is NUMERIC
                    if np.issubdtype(dtype, np.number):
                        
                        # A) If it's CONTINUOUS (like 'bmi' or 'price')
                        if n_unique > 50: 
                            st.markdown(f"**Plot Type:** Histogram (for continuous numeric data)")
                            
                            # Create histogram data
                            hist_values, bin_edges = np.histogram(col_data.dropna(), bins=50)
                            
                            # --- NEW: Create clean, rounded bin labels ---
                            # Round the bin edges to 2 decimal places for readability
                            rounded_edges = np.round(bin_edges, 2)
                            
                            # Create string labels for each bin (e.g., "10.00 - 20.50")
                            bin_labels = []
                            for i in range(len(rounded_edges) - 1):
                                start = rounded_edges[i]
                                end = rounded_edges[i+1]
                                bin_labels.append(f"{start} - {end}")
                            
                            hist_df = pd.DataFrame({
                                'Bin Range': bin_labels,
                                'count': hist_values
                            })
                            
                            # Plot using the new clean bin labels
                            st.bar_chart(hist_df, x='Bin Range', y='count')

                        # B) If it's DISCRETE (like 'age' or 'Seats')
                        else:
                            st.markdown(f"**Plot Type:** Bar Chart (for discrete numeric data)")
                            st.bar_chart(col_data.value_counts().sort_index())

                    # 2. If column is CATEGORICAL (text/object)
                    else:
                        
                        # A) If it's LOW CARDINALITY (like 'Fuel_Type' or 'Transmission')
                        if n_unique <= 50:
                            st.markdown(f"**Plot Type:** Bar Chart (for low-count categorical data)")
                            st.bar_chart(col_data.value_counts())
                        
                        # B) If it's HIGH CARDINALITY (like 'Name' or 'engine' in the dirty set)
                        else:
                            st.markdown(f"**Plot Type:** Sample of Unique Values (high-count text data)")
                            st.write(f"Showing the first 50 unique values (out of {n_unique}):")
                            st.dataframe(col_data.unique()[:50], use_container_width=True)

            # --- END OF NEW VISUALS TAB ---

            # --- AI Modeling Section ---
            st.header("Phase 2: AI-Powered Model Tournament")
            
            if st.button("🤖 Run Model Tournament"):
                # --- NEW: Clear old chat history on new run ---
                st.session_state.messages = []
                st.session_state.current_code = ""
                # --- END NEW ---

                if 'df' in st.session_state:
                    
                    # 1. Run Baseline Model
                    baseline_results_str, baseline_metrics, data_split = run_baseline_model(st.session_state.df)
                    
                    st.subheader("Model Results")
                    st.markdown("---")
                    st.markdown("### 1. Baseline Model (Ridge Regression)")
                    
                    if baseline_metrics and data_split:
                        st.code(baseline_results_str, language='text')
                        
                        # --- NEW: Display dropped columns for Baseline ---
                        if baseline_metrics.get('dropped_features'):
                            st.warning(f"**Baseline Model Dropped:** {len(baseline_metrics['dropped_features'])} columns (e.g., {baseline_metrics['dropped_features'][:5]}...). These were not identified as numeric or categorical.")
                        
                        X_train, X_test, y_train, y_test = data_split
                        st.session_state.data_split = data_split # Save data split for chat
                        
                        # 2. Get AI Champion Model
                        suggestion_text = get_ai_suggestion(st.session_state.df, baseline_results_str, baseline_metrics['target'], api_key) # <-- Pass api_key
                        
                        if suggestion_text:
                            code = extract_python_code(suggestion_text)
                            
                            if code:
                                # --- NEW: Save code and metrics for chat mode ---
                                st.session_state.current_code = code
                                st.session_state.target_col = baseline_metrics['target']
                                # --- END NEW ---
                                
                                # --- Parse all 4 parts of the AI's structured response ---
                                model_choice_match = re.search(r"\*\*1\. Champion Model Choice:\*\*(.*?)\n\*\*2\.", suggestion_text, re.DOTALL)
                                reasoning_match = re.search(r"\*\*2\. Reasoning:\*\*(.*?)\n\*\*3\.", suggestion_text, re.DOTALL)
                                dropped_match = re.search(r"\*\*3\. Dropped Columns:\*\*(.*?)\n\*\*4\.", suggestion_text, re.DOTALL)

                                model_choice_text = model_choice_match.group(1).strip() if model_choice_match else "AI Champion"
                                reasoning_text = reasoning_match.group(1).strip() if reasoning_match else "No analysis text found."
                                dropped_text = dropped_match.group(1).strip() if dropped_match else None
                                
                                # --- NEW: Save reasoning for chat mode ---
                                st.session_state.reasoning = reasoning_text
                                st.session_state.dropped = dropped_text
                                # --- END NEW ---

                                st.markdown(f"### 2. AI Champion Model ({model_choice_text})")
                                
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
                                    
                                    # Show the clean reasoning text AND dropped columns note
                                    if dropped_text:
                                        st.info(f"**AI's Note on Dropped Columns:** {dropped_text}")
                                        
                                    with st.expander("See AI's full analysis"):
                                        st.markdown(reasoning_text)
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
            
            # --- NEW: LANGCHAIN CHAT MODE (PHASE 3) ---
            # --- THIS ENTIRE BLOCK IS NOW INDENTED TO BE *INSIDE* THE 'try' BLOCK ---
            if 'current_code' in st.session_state and st.session_state.current_code:
                st.markdown("---")
                st.header("Phase 3: Manual Refinement (Chat Mode)")
                
                # --- CHANGE: REMOVED PERSONAL NOTES SECTION FROM HERE ---
                
                # --- NEW: Made this prompt more generic for any dataset ---
                st.info("The AI Champion model is a great start, but it might have missed some domain-specific context that you, the human expert, know.\n\n"
                        "Use this chat to give the AI instructions to fix its code. **For example, you could ask it to:**\n"
                        "> *'Create a new feature called 'Age_Squared' by squaring the 'Age' column in the 'custom_clean' function.'*\n\n"
                        "Or...\n\n"
                        "> *'The 'Mileage' column has mixed units (kmpl and km/kg). Rewrite the 'custom_clean' function to check the 'Fuel_Type' column and convert all mileage to a single, consistent unit.'*")

                # Initialize chat history
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                # Display chat messages from history
                for message in st.session_state.messages:
                    with st.chat_message(message.type):
                        st.markdown(message.content)

                # Accept user input
                if user_prompt := st.chat_input("Your instructions to the AI..."):
                    # Add user message to chat history
                    st.session_state.messages.append(HumanMessage(content=user_prompt))
                    with st.chat_message("human"):
                        st.markdown(user_prompt)

                    with st.chat_message("ai"):
                        with st.spinner("AI is refining the code..."):
                            
                            # --- This is the LangChain "Refiner" Logic ---
                            
                            # 1. Define the AI model, passing in the user's API key
                            llm = ChatGoogleGenerativeAI(model="gemini-pro-latest", google_api_key=api_key)
                            
                            # 2. Create the "Refiner" prompt template
                            # --- THIS IS THE CORRECTED BLOCK ---
                            refiner_prompt = ChatPromptTemplate.from_messages([
                                ("system", """
                                You are an expert Python data science coder. Your job is to refine the user's code based on their instructions. 
                                You must *only* output the new, complete, runnable Python code block and nothing else.
                                Do not add any explanations, just the code block.
                                The user has already provided all necessary imports. DO NOT add any `import` statements.
                                The code must end by defining the `y_pred` variable.
                                """),
                                MessagesPlaceholder(variable_name="chat_history"),
                                ("human", """
                                Here is the code to fix:
                                ```python
                                {current_code}
                                Here are my new instructions:
                            {user_instruction}
                            
                            Please provide the new, complete code block.
                            """) # <-- **FIXED: Added closing triple-quote**
                        ]) # <-- **FIXED: Added closing parenthesis**
                        # --- END OF CORRECTED BLOCK ---
                        
                        # 3. Create the chain
                        chain = refiner_prompt | llm
                        
                        # 4. Run the chain
                        response = chain.invoke({
                            "chat_history": st.session_state.messages,
                            "current_code": st.session_state.current_code,
                            "user_instruction": user_prompt
                        })
                        
                        ai_response_text = response.content
                        st.session_state.messages.append(AIMessage(content=ai_response_text))
                        
                        # 5. Extract the new code from the AI's response
                        new_code = extract_python_code(ai_response_text)
                        
                        if new_code:
                            st.session_state.current_code = new_code # Update the code for the next loop
                            
                            # 6. Run the new, refined code
                            X_train, X_test, y_train, y_test = st.session_state.data_split
                            y_pred_new, status_msg = run_model_code(
                                new_code, st.session_state.df, 
                                X_train, X_test, y_train, y_test
                            )
                            
                            if y_pred_new is not None:
                                # 7. Show the new, refined results!
                                r2_new = r2_score(y_test, y_pred_new)
                                rmse_new = root_mean_squared_error(y_test, y_pred_new)
                                
                                results_str_new = f"**Refined Model Results:**\nRMSE: ${rmse_new:,.2f}\nR-squared: {r2_new:.4f}"
                                st.success(results_str_new)
                                st.markdown("You can refine this code again, or scroll up and click 'Run Model Tournament' to start over with a new file.")
                                with st.expander("See the new refined code"):
                                    st.code(new_code, language='python')
                            else:
                                st.error(f"The refined code failed to run: {status_msg}")
                        else:
                            st.error("The AI's response did not contain a valid Python code block. Please try rephasing your request.")
                            st.markdown(ai_response_text) # Show the raw response if code extraction failed
    
    # --- THIS IS THE FIX: This 'except' block is now correctly aligned with the 'try' block on line 377 ---
        except Exception as e:
            st.error(f"An error occurred in the main app: {e}")

    else:
        # --- CRITICAL BUG FIX: If no file is uploaded, clear the state ---
        if st.session_state.current_file_name is not None:
            st.session_state.current_file_name = None
            st.session_state.messages = []
            st.session_state.current_code = ""
            st.session_state.data_split = None
            st.session_state.reasoning = ""
            st.session_state.dropped = ""
            st.session_state.target_col = ""
            
        st.info("Awaiting for a CSV file to be uploaded.")
# --- This goes at the VERY END of your .py file ---

# 1. Add a title to the sidebar
st.sidebar.title("Configuration")

# 2. ADDED LINK TO AI STUDIO
st.sidebar.markdown("[Get your Gemini API Key from Google AI Studio](https://aistudio.google.com/app/apikey)")

# 3. Ask for the API key in the sidebar, as you wanted
api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password")

# 4. Check if the key was provided
if api_key:
    # 2. ADDED SUCCESS MESSAGE
    st.sidebar.success("Gemini API Key accepted!")
    
    # --- NEW: MOVED PERSONAL NOTES TO SIDEBAR ---
    with st.sidebar.expander("📝 My Personal Notes"):
        st.text_area(
            "Jot down your thoughts, questions, or code snippets here. Your notes will be saved.", 
            key="user_notes", # This key links the text area to st.session_state.user_notes
            height=150
        )

    # If we have a key, run the main app logic
    main_app(api_key)
else:
    # If not, show a warning and wait for the key
    st.sidebar.warning("Please enter your API key to start.")
    st.info("Awaiting for your Gemini API Key in the sidebar...")