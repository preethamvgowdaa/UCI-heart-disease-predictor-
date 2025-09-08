import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration !! 
st.set_page_config(
    page_title="Heart Disease Risk Calculator",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .stAlert {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        # Load the Kaggle dataset
        df = pd.read_csv('data/heart.csv')
        
        # Display original shape
        st.sidebar.write(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Preprocess the data
        df = preprocess_data(df)
        
        return df
    except FileNotFoundError:
        st.error("❌ File not found! Please ensure 'heart.csv' is in the 'data' folder.")
        st.info(f"Current working directory: {os.getcwd()}")
        st.info("Looking for file at: data/heart.csv")
        
        # Check if data folder exists
        if os.path.exists('data'):
            st.info("✅ 'data' folder exists")
            files = os.listdir('data')
            if files:
                st.info(f"Files in data folder: {files}")
            else:
                st.warning("⚠️ 'data' folder is empty")
        else:
            st.warning("⚠️ 'data' folder does not exist. Please create a 'data' folder and add heart.csv")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.info("Make sure the CSV file is in the correct format and located at 'data/heart.csv'")
        return None

def preprocess_data(df):
    """Preprocess the Kaggle heart disease dataset"""
    
    # Create a copy
    df_processed = df.copy()
    
    # Drop id and dataset columns if they exist
    columns_to_drop = ['id', 'dataset']
    for col in columns_to_drop:
        if col in df_processed.columns:
            df_processed = df_processed.drop(col, axis=1)
    
    # Rename 'num' to 'target' if it exists
    if 'num' in df_processed.columns:
        df_processed['target'] = (df_processed['num'] > 0).astype(int)
        df_processed = df_processed.drop('num', axis=1)
    
    # Convert sex to binary
    if df_processed['sex'].dtype == 'object':
        df_processed['sex'] = df_processed['sex'].map({'Male': 1, 'Female': 0})
    
    # Convert chest pain type
    if 'cp' in df_processed.columns and df_processed['cp'].dtype == 'object':
        cp_mapping = {
            'typical angina': 0,
            'atypical angina': 1,
            'non-anginal': 2,
            'asymptomatic': 3
        }
        df_processed['cp'] = df_processed['cp'].map(cp_mapping)
    
    # Convert fbs (fasting blood sugar)
    if 'fbs' in df_processed.columns and df_processed['fbs'].dtype == 'object':
        df_processed['fbs'] = df_processed['fbs'].map({True: 1, False: 0, 'TRUE': 1, 'FALSE': 0})
    elif 'fbs' in df_processed.columns and df_processed['fbs'].dtype == 'bool':
        df_processed['fbs'] = df_processed['fbs'].astype(int)
    
    # Convert restecg
    if 'restecg' in df_processed.columns and df_processed['restecg'].dtype == 'object':
        restecg_mapping = {
            'normal': 0,
            'st-t abnormality': 1,
            'lv hypertrophy': 2
        }
        df_processed['restecg'] = df_processed['restecg'].map(restecg_mapping)
    
    # Rename thalch to thalach if needed
    if 'thalch' in df_processed.columns:
        df_processed = df_processed.rename(columns={'thalch': 'thalach'})
    
    # Convert exang
    if 'exang' in df_processed.columns and df_processed['exang'].dtype == 'object':
        df_processed['exang'] = df_processed['exang'].map({True: 1, False: 0, 'TRUE': 1, 'FALSE': 0})
    elif 'exang' in df_processed.columns and df_processed['exang'].dtype == 'bool':
        df_processed['exang'] = df_processed['exang'].astype(int)
    
    # Convert slope
    if 'slope' in df_processed.columns and df_processed['slope'].dtype == 'object':
        slope_mapping = {
            'upsloping': 0,
            'flat': 1,
            'downsloping': 2
        }
        df_processed['slope'] = df_processed['slope'].map(slope_mapping)
    
    # Convert thal
    if 'thal' in df_processed.columns and df_processed['thal'].dtype == 'object':
        thal_mapping = {
            'normal': 0,
            'fixed defect': 1,
            'reversable defect': 2,
            'reversible defect': 2  # Handle typo in dataset
        }
        df_processed['thal'] = df_processed['thal'].map(thal_mapping)
    
    # Handle missing values
    df_processed = df_processed.dropna()
    
    # Ensure all columns are numeric
    for col in df_processed.columns:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Drop rows with any NaN values after conversion
    df_processed = df_processed.dropna()
    
    return df_processed

# Train model function
@st.cache_resource
def train_model(df):
    if df is None or df.empty:
        return None, None, 0, []
    
    # Prepare features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Calculate accuracy
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, accuracy, X.columns

# Feature descriptions
feature_descriptions = {
    'age': 'Age in years',
    'sex': 'Sex (1 = male; 0 = female)',
    'cp': 'Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal, 3: asymptomatic)',
    'trestbps': 'Resting blood pressure (mm Hg)',
    'chol': 'Serum cholesterol (mg/dl)',
    'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
    'restecg': 'Resting ECG results (0: normal, 1: ST-T abnormality, 2: LV hypertrophy)',
    'thalach': 'Maximum heart rate achieved',
    'exang': 'Exercise induced angina (1 = yes; 0 = no)',
    'oldpeak': 'ST depression induced by exercise relative to rest',
    'slope': 'Slope of peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)',
    'ca': 'Number of major vessels colored by fluoroscopy (0-4)',
    'thal': 'Thalassemia (0: normal, 1: fixed defect, 2: reversible defect)'
}

# Main app
def main():
    st.title("❤️ Heart Disease Risk Calculator & Analytics Dashboard")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check the data file.")
        st.stop()
        return
    
    if df.empty:
        st.error("The dataset is empty after preprocessing.")
        st.stop()
        return
    
    # Show data info in sidebar
    st.sidebar.markdown("### Dataset Info")
    st.sidebar.write(f"Total records: {len(df)}")
    st.sidebar.write(f"Features: {len(df.columns)-1}")
    st.sidebar.write(f"Target distribution:")
    st.sidebar.write(df['target'].value_counts())
    
    # Train model
    with st.spinner("Training model..."):
        model, scaler, accuracy, feature_names = train_model(df)
    
    if model is None:
        st.error("Failed to train model.")
        return
    
    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page:",
        ["🏠 Home", "🔮 Risk Calculator", "📊 Data Analysis", "📈 Feature Correlations"]
    )
    
    if page == "🏠 Home":
        st.header("Welcome to Heart Disease Prediction System")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patients", f"{len(df):,}")
        with col2:
            st.metric("Model Accuracy", f"{accuracy:.2%}")
        with col3:
            disease_rate = (df['target'].sum()/len(df))*100
            st.metric("Disease Rate", f"{disease_rate:.1f}%")
        with col4:
            st.metric("Healthy Rate", f"{100-disease_rate:.1f}%")
        
        st.markdown("---")
        
        st.subheader("About This Application")
        st.info("""
        This application uses machine learning to predict heart disease risk based on various medical parameters.
        
        **Features:**
        - 🔮 Individual risk assessment
        - 📊 Comprehensive data visualization
        - 📈 Feature correlation analysis
        - 🎯 High accuracy predictions
        
        **Dataset:** Heart Disease Dataset with over 900 patient records
        **Model:** Random Forest Classifier with optimized hyperparameters
        """)
        
        # Quick Stats
        st.subheader("Quick Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            # Disease distribution pie chart
            disease_counts = df['target'].value_counts()
            fig = px.pie(
                values=disease_counts.values,
                names=['No Disease', 'Has Disease'],
                title="Disease Distribution",
                color_discrete_sequence=['#2ecc71', '#e74c3c']
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Age distribution
            fig = px.histogram(
                df, x='age', color='target',
                title="Age Distribution by Disease Status",
                labels={'target': 'Disease Status', 'age': 'Age (years)'},
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                nbins=20
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender distribution
            gender_disease = df.groupby(['sex', 'target']).size().reset_index(name='count')
            gender_disease['sex'] = gender_disease['sex'].map({0: 'Female', 1: 'Male'})
            gender_disease['target'] = gender_disease['target'].map({0: 'No Disease', 1: 'Has Disease'})
            
            fig = px.bar(
                gender_disease, x='sex', y='count', color='target',
                title="Gender Distribution by Disease Status",
                color_discrete_map={'No Disease': '#2ecc71', 'Has Disease': '#e74c3c'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average metrics by disease status
            avg_metrics = df.groupby('target')[['age', 'trestbps', 'chol', 'thalach']].mean()
            
            st.markdown("### Average Metrics by Disease Status")
            for col in avg_metrics.columns:
                col1_m, col2_m = st.columns(2)
                with col1_m:
                    st.metric(f"Avg {col} (No Disease)", f"{avg_metrics.loc[0, col]:.1f}")
                with col2_m:
                    st.metric(f"Avg {col} (Disease)", f"{avg_metrics.loc[1, col]:.1f}")
    
    elif page == "🔮 Risk Calculator":
        st.header("Individual Heart Disease Risk Assessment")
        st.markdown("Enter patient information to calculate disease risk")
        
        # Create input form
        with st.form("risk_calculator_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Age", min_value=1, max_value=120, value=50, help="Patient's age in years")
                sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
                cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
                                format_func=lambda x: ["Typical Angina", "Atypical Angina", 
                                                      "Non-anginal Pain", "Asymptomatic"][x])
                trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 
                                          min_value=80, max_value=250, value=120)
                chol = st.number_input("Cholesterol (mg/dl)", 
                                      min_value=100, max_value=600, value=200)
            
            with col2:
                fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", 
                                 options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                restecg = st.selectbox("Resting ECG", options=[0, 1, 2],
                                     format_func=lambda x: ["Normal", "ST-T Abnormality", 
                                                           "LV Hypertrophy"][x])
                thalach = st.number_input("Max Heart Rate", min_value=60, max_value=250, value=150)
                exang = st.selectbox("Exercise Induced Angina", 
                                   options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            
            with col3:
                oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
                slope = st.selectbox("Slope of Peak Exercise ST", options=[0, 1, 2],
                                   format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
                ca = st.selectbox("Number of Major Vessels (0-4)", options=[0, 1, 2, 3, 4])
                thal = st.selectbox("Thalassemia", options=[0, 1, 2],
                                  format_func=lambda x: ["Normal", "Fixed Defect", 
                                                        "Reversible Defect"][x])
            
            submitted = st.form_submit_button("Calculate Risk", type="primary", use_container_width=True)
        
        if submitted:
            # Prepare input
            input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
                                   thalach, exang, oldpeak, slope, ca, thal]])
            
            # Ensure column names match
            input_df = pd.DataFrame(input_data, columns=feature_names)
            input_scaled = scaler.transform(input_df)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            st.markdown("---")
            st.subheader("Risk Assessment Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ HIGH RISK of Heart Disease Detected")
                    risk_percentage = probability[1] * 100
                else:
                    st.success("✅ LOW RISK of Heart Disease")
                    risk_percentage = probability[1] * 100
                
                # Risk gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = risk_percentage,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Risk Score (%)"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#e74c3c" if risk_percentage > 50 else "#2ecc71"},
                        'steps': [
                            {'range': [0, 25], 'color': "#d4edda"},
                            {'range': [25, 50], 'color': "#fff3cd"},
                            {'range': [50, 75], 'color': "#f8d7da"},
                            {'range': [75, 100], 'color': "#f5c6cb"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.metric("Risk Probability", f"{risk_percentage:.1f}%")
                st.metric("Confidence Level", f"{max(probability)*100:.1f}%")
                
                # Risk factors summary
                st.markdown("### Key Risk Factors")
                
                risk_factors = []
                if age > 55:
                    risk_factors.append("- Age > 55 years")
                if sex == 1:
                    risk_factors.append("- Male gender")
                if cp == 3:
                    risk_factors.append("- Asymptomatic chest pain")
                if trestbps > 140:
                    risk_factors.append("- High blood pressure")
                if chol > 240:
                    risk_factors.append("- High cholesterol")
                if fbs == 1:
                    risk_factors.append("- High fasting blood sugar")
                if exang == 1:
                    risk_factors.append("- Exercise induced angina")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.write(factor)
                else:
                    st.write("No major risk factors identified")
                
                # Recommendations
                st.markdown("### Recommendations")
                if prediction == 1:
                    st.warning("""
                    **Immediate Actions:**
                    - Schedule consultation with a cardiologist
                    - Get comprehensive cardiac evaluation
                    - Monitor blood pressure daily
                    - Review current medications with doctor
                    
                    **Lifestyle Changes:**
                    - Adopt heart-healthy diet
                    - Regular moderate exercise
                    - Stress management techniques
                    - Quit smoking if applicable
                    """)
                else:
                    st.info("""
                    **Preventive Measures:**
                    - Maintain healthy lifestyle
                    - Regular exercise (150 min/week)
                    - Balanced diet (Mediterranean diet recommended)
                    - Annual health check-ups
                    - Monitor blood pressure and cholesterol
                    - Maintain healthy weight
                    """)
    
    elif page == "📊 Data Analysis":
        st.header("Comprehensive Data Analysis")
        
        # Dataset overview
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Features", len(df.columns)-1)
        with col3:
            st.metric("Disease Cases", f"{df['target'].sum():,}")
        with col4:
            st.metric("Healthy Cases", f"{(len(df) - df['target'].sum()):,}")
        
        # Data preview
        st.subheader("Data Preview")
        
        # Add filtering options
        col1, col2, col3 = st.columns(3)
        with col1:
            show_disease = st.selectbox("Filter by Disease Status", 
                                       ["All", "No Disease", "Has Disease"])
        with col2:
            show_gender = st.selectbox("Filter by Gender", 
                                      ["All", "Male", "Female"])
        with col3:
            num_rows = st.slider("Number of rows to display", 5, 50, 10)
        
        # Apply filters
        filtered_df = df.copy()
        if show_disease != "All":
            filtered_df = filtered_df[filtered_df['target'] == (1 if show_disease == "Has Disease" else 0)]
        if show_gender != "All":
            filtered_df = filtered_df[filtered_df['sex'] == (1 if show_gender == "Male" else 0)]
        
        st.dataframe(filtered_df.head(num_rows))
        
        # Statistical summary
        st.subheader("Statistical Summary")
        st.dataframe(df.describe().round(2))
        
        # Visualizations
        st.subheader("Data Visualizations")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Relationships", "Comparisons", "Feature Analysis"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Age distribution
                fig = px.histogram(df, x='age', nbins=30,
                                 title="Age Distribution",
                                 labels={'age': 'Age (years)', 'count': 'Number of Patients'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Cholesterol distribution
                fig = px.histogram(df, x='chol', nbins=30,
                                 title="Cholesterol Distribution",
                                 labels={'chol': 'Cholesterol (mg/dl)', 'count': 'Number of Patients'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Blood pressure distribution
                fig = px.histogram(df, x='trestbps', nbins=30,
                                 title="Resting Blood Pressure Distribution",
                                 labels={'trestbps': 'Blood Pressure (mm Hg)', 'count': 'Number of Patients'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Max heart rate distribution
                fig = px.histogram(df, x='thalach', nbins=30,
                                 title="Maximum Heart Rate Distribution",
                                 labels={'thalach': 'Max Heart Rate', 'count': 'Number of Patients'})
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Age vs Max Heart Rate
                fig = px.scatter(df, x='age', y='thalach', color='target',
                               title="Age vs Maximum Heart Rate",
                               labels={'age': 'Age', 'thalach': 'Max Heart Rate'},
                               color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Cholesterol vs Blood Pressure
                fig = px.scatter(df, x='chol', y='trestbps', color='target',
                               title="Cholesterol vs Blood Pressure",
                               labels={'chol': 'Cholesterol', 'trestbps': 'Blood Pressure'},
                               color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                # Chest pain type distribution
                cp_counts = df.groupby(['cp', 'target']).size().reset_index(name='count')
                cp_counts['cp'] = cp_counts['cp'].map({0: 'Typical', 1: 'Atypical', 
                                                       2: 'Non-anginal', 3: 'Asymptomatic'})
                cp_counts['target'] = cp_counts['target'].map({0: 'No Disease', 1: 'Has Disease'})
                
                fig = px.bar(cp_counts, x='cp', y='count', color='target',
                           title="Chest Pain Type Distribution",
                           color_discrete_map={'No Disease': '#2ecc71', 'Has Disease': '#e74c3c'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Exercise induced angina
                exang_counts = df.groupby(['exang', 'target']).size().reset_index(name='count')
                exang_counts['exang'] = exang_counts['exang'].map({0: 'No', 1: 'Yes'})
                exang_counts['target'] = exang_counts['target'].map({0: 'No Disease', 1: 'Has Disease'})
                
                fig = px.bar(exang_counts, x='exang', y='count', color='target',
                           title="Exercise Induced Angina",
                           color_discrete_map={'No Disease': '#2ecc71', 'Has Disease': '#e74c3c'})
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            # Feature importance
            st.subheader("Feature Importance for Prediction")
            
            if model is not None:
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=True)
                
                fig = px.bar(feature_importance, x='importance', y='feature',
                           orientation='h', title="Feature Importance Score",
                           labels={'importance': 'Importance', 'feature': 'Feature'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Top features explanation
                st.markdown("### Top 5 Most Important Features")
                top_features = feature_importance.nlargest(5, 'importance')
                for idx, row in top_features.iterrows():
                    st.write(f"**{row['feature']}**: {row['importance']:.3f} - {feature_descriptions.get(row['feature'], 'No description')}")
    
    elif page == "📈 Feature Correlations":
        st.header("Feature Correlation Analysis")
        
        # Correlation matrix
        st.subheader("Correlation Matrix")
        
        # Calculate correlation
        corr_matrix = df.corr()
        
        # Create heatmap
        fig = px.imshow(corr_matrix,
                       labels=dict(x="Features", y="Features", color="Correlation"),
                       x=corr_matrix.columns,
                       y=corr_matrix.columns,
                       color_continuous_scale='RdBu_r',
                       zmin=-1, zmax=1,
                       title="Feature Correlation Heatmap")
        fig.update_layout(height=700, width=900)
        st.plotly_chart(fig, use_container_width=True)
        
        # Target correlations
        st.subheader("Correlations with Heart Disease")
        
        target_corr = df.corr()['target'].sort_values(ascending=False)[1:]
        
        # Color code correlations
        colors = ['#e74c3c' if x > 0 else '#2ecc71' for x in target_corr.values]
        
        fig = px.bar(x=target_corr.values, y=target_corr.index,
                    orientation='h',
                    title="Feature Correlations with Heart Disease",
                    labels={'x': 'Correlation Coefficient', 'y': 'Features'},
                    color=target_corr.values,
                    color_continuous_scale='RdBu_r')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Interactive scatter plots
        st.subheader("Interactive Feature Explorer")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox("Select X-axis", options=list(df.columns[:-1]))
        with col2:
            y_axis = st.selectbox("Select Y-axis", options=list(df.columns[:-1]), index=1)
        with col3:
            color_by = st.selectbox("Color by", options=['target', 'sex', 'cp'])
        
        # Create scatter plot
        if color_by == 'target':
            color_map = {0: '#2ecc71', 1: '#e74c3c'}
        else:
            color_map = None
        
        fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by,
                        title=f"{x_axis} vs {y_axis}",
                        color_discrete_map=color_map,
                        hover_data=df.columns)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation insights
        st.subheader("Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Strongest Positive Correlations with Disease")
            positive_corr = target_corr[target_corr > 0].head(5)
            for feat, corr in positive_corr.items():
                st.write(f"- **{feat}**: {corr:.3f}")
        
        with col2:
            st.markdown("### Strongest Negative Correlations with Disease")
            negative_corr = target_corr[target_corr < 0].head(5)
            for feat, corr in negative_corr.items():
                st.write(f"- **{feat}**: {corr:.3f}")
        
        # Feature descriptions
        st.subheader("Feature Descriptions")
        
        with st.expander("Click to see feature descriptions"):
            for feature, description in feature_descriptions.items():
                if feature in df.columns:
                    st.write(f"**{feature}**: {description}")

if __name__ == "__main__":
    main()