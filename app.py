import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# --- Load Models and Scaler ---
@st.cache_resource
def load_assets():
    try:
        # Check if the models exist yet
        if not os.path.exists('models/scaler.pkl'):
            return None, None
            
        scaler = joblib.load('models/scaler.pkl')
        
        models = {
            'Decision Tree': joblib.load('models/decision_tree.pkl'),
            'Logistic Regression': joblib.load('models/logistic_regression.pkl'),
            'Random Forest': joblib.load('models/random_forest.pkl')
        }
        return scaler, models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

@st.cache_data
def load_historical_data():
    if os.path.exists('simulated_student_data.csv'):
        return pd.read_csv('simulated_student_data.csv')
    return pd.DataFrame()

scaler, models = load_assets()
historical_data = load_historical_data()

# --- Main App UI ---
st.title("🎓 Kitale National Polytechnic")
st.subheader("Student Academic Performance Platform")

if not models:
    st.warning("⚠️ Models not found! Please run the training script (`phase2_3_models.py`) first to generate the model files.")
    st.stop()

# --- Layout: Tabs for different functionalities ---
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Single Prediction Dialog", "📁 Batch Prediction (Class Roster)", "📊 Analytics Dashboard", "🤖 Model Performance"])

# --- Tab 1: Single Prediction ---
with tab1:
    st.markdown("""
    Use this module to predict the outcome for a **single student** by manually entering their continuous assessment records.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Enter Student Data")
        
        department = st.selectbox(
            "Department",
            ['Applied Sciences', 'Business', 'Computing and Informatics', 'Engineering'],
            key='single_dept'
        )
        
        attendance = st.slider("Attendance Percentage (%)", min_value=0, max_value=100, value=75, key='single_att')
        cat_1_score = st.slider("CAT 1 Score (%)", min_value=0, max_value=100, value=65, key='single_cat1')
        cat_2_score = st.slider("CAT 2 Score (%)", min_value=0, max_value=100, value=65, key='single_cat2')
        assignment_score = st.slider("Assignment Score (%)", min_value=0, max_value=100, value=70, key='single_ass')
        
        selected_model_name = st.selectbox(
            "Select Prediction Model",
            list(models.keys()),
            index=2, # Default to Random Forest
            key='single_model'
        )
        
        predict_btn = st.button("Predict Performance", type="primary", key='single_btn')

    with col2:
        if predict_btn:
            input_data = pd.DataFrame({
                'Attendance_Pct': [attendance],
                'Assignment_Score': [assignment_score],
                'CAT_1_Score': [cat_1_score],
                'CAT_2_Score': [cat_2_score],
                'Department_Business': [1 if department == 'Business' else 0],
                'Department_Computing and Informatics': [1 if department == 'Computing and Informatics' else 0],
                'Department_Engineering': [1 if department == 'Engineering' else 0]
            })
            
            try:
                scaled_input = scaler.transform(input_data)
                model = models[selected_model_name]
                prediction = model.predict(scaled_input)[0]
                probability = model.predict_proba(scaled_input)[0]
                
                st.subheader("Prediction Results")
                
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    if prediction == 1:
                        st.success("## 🟢 PASS")
                        st.markdown("**Status:** The student is on track to pass the final exam.")
                    else:
                        st.error("## 🔴 FAIL (At Risk)")
                        st.markdown("**Status:** The student is at high risk of failing.")
                        st.markdown("**Recommendation:** Early intervention and academic support recommended.")
                        
                with res_col2:
                    st.metric(label="Model Confidence (Pass Probability)", value=f"{probability[1]*100:.1f}%")
                    
                if selected_model_name in ['Random Forest', 'Decision Tree']:
                    importances = model.feature_importances_
                    dept_importance = sum(importances[4:7]) # sum of department encodings
                    combined_importances = {
                        'Attendance': importances[0],
                        'Assignments': importances[1],
                        'CAT 1 Score': importances[2],
                        'CAT 2 Score': importances[3],
                        'Department': dept_importance
                    }
                    
                    st.subheader("What influenced this prediction?")
                    st.bar_chart(pd.Series(combined_importances))

            except Exception as e:
                st.error(f"Prediction Error: {e}. Please ensure the input data matches the model's expected format.")

# --- Tab 2: Batch Prediction ---
with tab2:
    st.markdown("""
    Upload an Excel or CSV file containing an entire class roster to instantly identify "At-Risk" students in bulk.
    Your file must contain the following columns: `StudentID`, `Department`, `Attendance_Pct`, `Assignment_Score`, `CAT_1_Score`, `CAT_2_Score`.
    """)
    
    uploaded_file = st.file_uploader("Upload Class Roster", type=['csv', 'xlsx'])
    
    batch_model_name = st.selectbox(
        "Select Prediction Model",
        list(models.keys()),
        index=2, # Default to Random Forest
        key='batch_model'
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_roster = pd.read_csv(uploaded_file)
            else:
                df_roster = pd.read_excel(uploaded_file)
                
            st.write("### Uploaded Preview")
            st.dataframe(df_roster.head())
            
            # Check for required columns
            required_cols = ['Department', 'Attendance_Pct', 'Assignment_Score', 'CAT_1_Score', 'CAT_2_Score']
            if not all(col in df_roster.columns for col in required_cols):
                st.error(f"Missing required columns! Ensure your file has: {', '.join(required_cols)}")
            else:
                if st.button("Run Batch Prediction", type="primary"):
                    with st.spinner('Running predictions for all students...'):
                        # Prepare Data
                        X_batch = pd.DataFrame({
                            'Attendance_Pct': df_roster['Attendance_Pct'],
                            'Assignment_Score': df_roster['Assignment_Score'],
                            'CAT_1_Score': df_roster['CAT_1_Score'],
                            'CAT_2_Score': df_roster['CAT_2_Score'],
                            'Department_Business': (df_roster['Department'] == 'Business').astype(int),
                            'Department_Computing and Informatics': (df_roster['Department'] == 'Computing and Informatics').astype(int),
                            'Department_Engineering': (df_roster['Department'] == 'Engineering').astype(int)
                        })
                        
                        scaled_batch = scaler.transform(X_batch)
                        model = models[batch_model_name]
                        
                        predictions = model.predict(scaled_batch)
                        probabilities = model.predict_proba(scaled_batch)[:, 1]
                        
                        df_results = df_roster.copy()
                        df_results['Prediction'] = ['Pass' if p == 1 else 'Fail' for p in predictions]
                        df_results['Pass_Probability'] = np.round(probabilities * 100, 1)
                        df_results['Risk_Level'] = ['High' if p == 'Fail' else 'Low' for p in df_results['Prediction']]
                        
                        st.success(f"Successfully evaluated {len(df_results)} students!")
                        
                        # Show styled dataframe (highlight at-risk students)
                        def highlight_at_risk(row):
                            if row['Risk_Level'] == 'High':
                                return ['background-color: #ffcccc'] * len(row)
                            return [''] * len(row)
                            
                        st.dataframe(df_results.style.apply(highlight_at_risk, axis=1), use_container_width=True)
                        
                        # Download button
                        csv = df_results.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Evaluated Roster (CSV)",
                            data=csv,
                            file_name='evaluated_class_roster.csv',
                            mime='text/csv',
                        )

        except Exception as e:
            st.error(f"Error processing file: {e}")

# --- Tab 3: Analytics Dashboard ---
with tab3:
    st.markdown("""
    Explore historical trends and aggregated performance data across the institution to identify macro-level issues.
    """)
    
    if historical_data.empty:
        st.warning("Historical data file (`simulated_student_data.csv`) not found.")
    else:
        # Top level metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Students Tracked", f"{len(historical_data)}")
        with col2:
            pass_rate = (historical_data['Status'] == 'Pass').mean() * 100
            st.metric("Overall Institutional Pass Rate", f"{pass_rate:.1f}%")
        with col3:
            avg_attendance = historical_data['Attendance_Pct'].mean()
            st.metric("Average Institution Attendance", f"{avg_attendance:.1f}%")
            
        st.divider()
        
        # Row 1 of Charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Department Pass Rates
            st.subheader("Pass Rates by Department")
            dept_status = historical_data.groupby(['Department', 'Status']).size().unstack(fill_value=0)
            if 'Pass' in dept_status.columns and 'Fail' in dept_status.columns:
                 dept_status['Pass_Rate'] = (dept_status['Pass'] / (dept_status['Pass'] + dept_status['Fail'])) * 100
                 fig1 = px.bar(dept_status.reset_index(), x='Department', y='Pass_Rate', 
                               color='Department', title="Departmental Success Percentage")
                 fig1.update_layout(yaxis_title="Pass Rate (%)", showlegend=False)
                 st.plotly_chart(fig1, use_container_width=True)
            else:
                 st.info("Not enough variance in historical pass/fail data to chart.")
                 
        with chart_col2:
            # Score Distributions
            st.subheader("Performance Distributions")
            # Melt dataframe for violin plot
            melted_scores = historical_data.melt(id_vars=['Department', 'Status'], 
                                                 value_vars=['Attendance_Pct', 'CAT_1_Score', 'CAT_2_Score', 'Assignment_Score'],
                                                 var_name='Metric', value_name='Score')
            fig2 = px.box(melted_scores, x='Metric', y='Score', color='Status', 
                          title="Score Distributions (Pass vs Fail)")
            st.plotly_chart(fig2, use_container_width=True)
            
        # Row 2 of Charts
        st.subheader("Attendance vs. CAT 1 Performance Map")
        fig3 = px.scatter(historical_data, x='Attendance_Pct', y='CAT_1_Score', color='Status',
                          hover_data=['StudentID', 'Department'],
                          title="Correlation between Attendance and CAT 1",
                          color_discrete_map={'Pass': 'green', 'Fail': 'red'})
        st.plotly_chart(fig3, use_container_width=True)
# --- Tab 4: Model Performance ---
with tab4:
    st.markdown("""
    This section provides transparency into the Machine Learning models powering the predictions. 
    It displays the evaluation metrics generated during the training phase.
    """)
    
    col_perf1, col_perf2 = st.columns(2)
    
    with col_perf1:
        st.subheader("Model Accuracy Comparison")
        if os.path.exists('evaluation_plots/accuracy_comparison.png'):
            st.image('evaluation_plots/accuracy_comparison.png', caption="Comparison of overall accuracy across the three algorithms.")
        else:
            st.info("Accuracy Comparison chart not found. Ensure `phase2_3_models.py` was run.")
            
    with col_perf2:
        st.subheader("Confusion Matrices")
        st.markdown("A confusion matrix shows exactly where a model gets things right, and where it gets confused (e.g., predicting a Pass when the student actually Failed).")
        
        matrix_selection = st.selectbox(
            "Select Model Matrix to View",
            ['Logistic Regression', 'Random Forest', 'Decision Tree']
        )
        
        matrix_path = f'evaluation_plots/confusion_matrix_{matrix_selection.replace(" ", "_").lower()}.png'
        if os.path.exists(matrix_path):
            st.image(matrix_path, caption=f"Confusion Matrix for {matrix_selection}")
        else:
            st.info(f"Matrix for {matrix_selection} not found.")
