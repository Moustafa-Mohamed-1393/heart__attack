import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb_lib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from xgboost import XGBClassifier, plot_importance
from sklearn.inspection import permutation_importance
from io import StringIO


# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("./Heart_Attack_Risk_Levels_Dataset.csv")
    df['Gender'] = df['Gender'].map({1: 'Male', 0: 'Female'})
    
    # Create age bins
    df['Age Group'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 75, 100], 
                            labels=['<30', '30-45', '45-60', '60-75', '75+'])
    
    # Create blood pressure categories
    df['BP Category'] = pd.cut(df['Systolic blood pressure'], 
                              bins=[0, 90, 120, 140, 160, 180, 200, 220, 300],
                              labels=['Low', 'Normal', 'Elevated', 'Stage 1 Hypertension', 
                                     'Stage 2 Hypertension', 'Hypertensive Crisis', 
                                     'Severe Hypertension', 'Critical'])
    
    # Create blood sugar categories
    df['Blood Sugar Category'] = pd.cut(df['Blood sugar'],
                                      bins=[0, 70, 100, 125, 200, 300, 400, 500, 600],
                                      labels=['Low', 'Normal', 'Prediabetes', 'Diabetes',
                                              'High Diabetes', 'Very High', 'Extreme', 'Dangerous'])
    
    # Create heart rate categories
    df['Heart Rate Category'] = pd.cut(df['Heart rate'],
                                     bins=[0, 40, 60, 100, 140, 200, 300, 500, 1200],
                                     labels=['Very Low', 'Low', 'Normal', 'Elevated',
                                             'High', 'Very High', 'Extreme', 'Dangerous'])
    return df

df = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", 
                        ["Dataset Overview", 
                         "Risk Level Distribution", 
                         "Age Analysis", 
                         "Gender Analysis", 
                         "Biomarkers Analysis", 
                         "Advanced Visualizations", 
                         "Modeling"])



# Main content based on selected page
if page == "Dataset Overview":
    st.title("Heart Attack Risk Dataset Overview")
    
    st.header("Dataset Preview")
    st.dataframe(df.head())
    
    st.header("Dataset Information")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")
    
    st.subheader("Columns Information")
    st.write(df.columns.tolist())
    
    st.subheader("Missing Values")
    st.write(df.isnull().sum())
    
    st.subheader("Descriptive Statistics")
    st.write(df.describe().T)

elif page == "Risk Level Distribution":
    st.title("Risk Level Distribution Analysis")
    
    st.header("Distribution of Risk Levels")
    risk_dist = px.pie(df, names='Risk_Level', title='Distribution of Risk Levels',
                    color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(risk_dist)
    
    st.header("Recommendations by Risk Level")
    rec_dist = px.bar(df, x='Risk_Level', color='Recommendation', 
                     title='Recommendations by Risk Level',
                     color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(rec_dist)
    
    st.header("Test Results by Risk Level")
    result_dist = px.bar(df, x='Risk_Level', color='Result', 
                        title='Test Results by Risk Level',
                        color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(result_dist)

elif page == "Age Analysis":
    st.title("Age-Related Analysis")
    
    st.header("Age Distribution by Risk Level")
    age_dist = px.histogram(df, x='Age', color='Risk_Level', nbins=30,
                           title='Age Distribution by Risk Level',
                           barmode='overlay', opacity=0.7,
                           color_discrete_sequence=px.colors.sequential.RdBu)
    age_dist.update_layout(bargap=0.1)
    st.plotly_chart(age_dist)
    
    st.header("Age Group Distribution")
    age_group_dist = px.histogram(df, x='Age Group', color='Risk_Level',
                                title='Age Group Distribution by Risk Level',
                                barmode='group',
                                color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(age_group_dist)
    
    st.header("Average Age by Risk Level")
    avg_age = df.groupby('Risk_Level')['Age'].mean().reset_index()
    avg_age_fig = px.bar(avg_age, x='Risk_Level', y='Age', 
                        title='Average Age by Risk Level',
                        color='Risk_Level',
                        color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(avg_age_fig)

elif page == "Gender Analysis":
    st.title("Gender-Related Analysis")
    
    st.header("Gender Distribution by Risk Level")
    gender_dist = px.histogram(df, x='Gender', color='Risk_Level',
                             title='Gender Distribution by Risk Level',
                             barmode='group',
                             color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(gender_dist)
    
    st.header("Gender Proportion by Risk Level")
    gender_prop = df.groupby(['Risk_Level', 'Gender']).size().unstack().fillna(0)
    gender_prop = gender_prop.div(gender_prop.sum(axis=1), axis=0)
    gender_prop_fig = px.bar(gender_prop, barmode='group',
                            title='Gender Proportion by Risk Level',
                            color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(gender_prop_fig)
    
    st.header("Gender and Age Interaction")
    gender_age_fig = px.box(df, x='Gender', y='Age', color='Risk_Level',
                           title='Age Distribution by Gender and Risk Level',
                           color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(gender_age_fig)

elif page == "Biomarkers Analysis":
    st.title("Biomarkers Analysis")
    
    st.header("CK-MB vs Troponin by Risk Level")
    biomarkers_fig = px.scatter(df, x='CK-MB', y='Troponin', color='Risk_Level',
                              title='CK-MB vs Troponin by Risk Level',
                              color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(biomarkers_fig)
    
    st.header("Blood Pressure Analysis")
    bp_fig = px.box(df, x='Risk_Level', y='Systolic blood pressure', 
                   title='Systolic Blood Pressure by Risk Level',
                   color='Risk_Level',
                   color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(bp_fig)
    
    st.header("Blood Sugar Analysis")
    bs_fig = px.box(df, x='Risk_Level', y='Blood sugar', 
                   title='Blood Sugar by Risk Level',
                   color='Risk_Level',
                   color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(bs_fig)
    
    st.header("Heart Rate Analysis")
    hr_fig = px.box(df, x='Risk_Level', y='Heart rate', 
                   title='Heart Rate by Risk Level',
                   color='Risk_Level',
                   color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(hr_fig)
elif page == "Advanced Visualizations":
    st.title("Advanced Visualizations")

    st.header("3D Scatter Plot: Age, Blood Pressure, Troponin")

    scatter_3d = px.scatter_3d(df, 
                               x='Age', 
                               y='Systolic blood pressure', 
                               z='Troponin',
                               color='Risk_Level', 
                               size='CK-MB',
                               title='3D View: Age, Blood Pressure, and Troponin by Risk Level',
                               color_discrete_sequence=px.colors.sequential.RdBu,
                               opacity=0.7)
    scatter_3d.update_layout(height=700)
    st.plotly_chart(scatter_3d)

    st.header("Parallel Coordinates Plot: Risk Factors")

    df['Risk_Level_Num'] = df['Risk_Level'].astype('category').cat.codes

    fig = go.Figure(data=go.Parcoords(
        line=dict(color=df['Risk_Level_Num'],
                  colorscale='RdBu',
                  showscale=True,
                  cmin=df['Risk_Level_Num'].min(),
                  cmax=df['Risk_Level_Num'].max()),
        dimensions=[
            dict(label='Age', values=df['Age']),
            dict(label='Systolic BP', values=df['Systolic blood pressure']),
            dict(label='Diastolic BP', values=df['Diastolic blood pressure']),
            dict(label='Blood Sugar', values=df['Blood sugar']),
            dict(label='CK-MB', values=df['CK-MB']),
            dict(label='Troponin', values=df['Troponin'])
        ]
    ))
    fig.update_layout(title='Parallel Coordinates Plot of Risk Factors')
    st.plotly_chart(fig)
    st.header("Age vs Biomarkers (Colored by Opposite Biomarker)")

    from plotly.subplots import make_subplots

    fig_biomarkers = make_subplots(rows=1, cols=2, subplot_titles=('Age vs CK-MB', 'Age vs Troponin'))

    fig_biomarkers.add_trace(
        go.Scatter(
            x=df['Age'],
            y=df['CK-MB'],
            mode='markers',
            marker=dict(
                color=df['Troponin'],
                colorscale='RdBu',
                showscale=True,
                size=8,
                opacity=0.7
            ),
            name='CK-MB'
        ),
        row=1,
        col=1
    )

    fig_biomarkers.add_trace(
        go.Scatter(
            x=df['Age'],
            y=df['Troponin'],
            mode='markers',
            marker=dict(
                color=df['CK-MB'],
                colorscale='RdBu',
                showscale=True,
                size=8,
                opacity=0.7
            ),
            name='Troponin'
        ),
        row=1,
        col=2
    )

    fig_biomarkers.update_layout(
        title_text='Age vs Biomarkers (Color represents the other biomarker)',
        height=500
    )
    st.plotly_chart(fig_biomarkers)
elif page == "Modeling":
    st.title("Heart Attack Risk Prediction Models")
    
    # Data preparation
    df['Risk_Level_Num'] = df['Risk_Level'].astype('category').cat.codes
    y = df['Risk_Level_Num']
    all_features = df.select_dtypes(include=[np.number]).drop(columns=['Risk_Level_Num']).columns.tolist()
    X = df[all_features]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Enhanced model dictionary with metadata
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000),
            "needs_scaling": True,
            "type": "sklearn"
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=42),
            "needs_scaling": False,
            "type": "sklearn"
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "needs_scaling": False,
            "type": "sklearn"
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "needs_scaling": False,
            "type": "sklearn"
        },
        "XGBoost": {
            "model": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
            "needs_scaling": False,
            "type": "sklearn"
        },
        "Dummy Classifier": {
            "model": DummyClassifier(strategy='most_frequent'),
            "needs_scaling": False,
            "type": "sklearn"
        },
        "Neural Network": {
            "model": Sequential([
                Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                Dense(32, activation='relu'),
                Dense(3, activation='softmax')
            ]),
            "needs_scaling": True,
            "type": "keras",
            "compile_params": {
                "optimizer": 'adam',
                "loss": 'categorical_crossentropy',
                "metrics": ['accuracy']
            },
            "fit_params": {
                "epochs": 50,
                "batch_size": 32,
                "validation_split": 0.2,
                "verbose": 0
            }
        }
    }

    # Train all models
    y_train_nn = to_categorical(y_train)
    y_test_nn = to_categorical(y_test)
    
    for name, model_info in models.items():
        if model_info["type"] == "sklearn":
            if model_info["needs_scaling"]:
                model_info["model"].fit(X_train_scaled, y_train)
            else:
                model_info["model"].fit(X_train, y_train)
        elif model_info["type"] == "keras":
            model_info["model"].compile(**model_info["compile_params"])
            history = model_info["model"].fit(
                X_train_scaled, 
                y_train_nn,
                **model_info["fit_params"]
            )
            model_info["history"] = history

    # Create tabs for visualization
    tab_names = ["All Models Comparison"] + list(models.keys())
    tabs = st.tabs(tab_names)

    # Tab 0: All Models Comparison
    with tabs[0]:
        st.header("Model Comparison Overview")
        
        # Accuracy comparison
        results = []
        for name, model_info in models.items():
            if model_info["type"] == "sklearn":
                X_input = X_test_scaled if model_info["needs_scaling"] else X_test
                y_pred = model_info["model"].predict(X_input)
                acc = accuracy_score(y_test, y_pred)
                results.append({'Model': name, 'Accuracy': acc})
            else:  # Neural Network
                nn_loss, nn_acc = model_info["model"].evaluate(X_test_scaled, y_test_nn, verbose=0)
                results.append({'Model': name, 'Accuracy': nn_acc})
        
        results_df = pd.DataFrame(results).sort_values('Accuracy')
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(results_df.style.format({'Accuracy': '{:.2%}'}).highlight_max(subset=['Accuracy'], color='lightgreen'))
        
        with col2:
            fig = plt.figure(figsize=(8, 4))
            sns.barplot(x='Accuracy', y='Model', data=results_df, palette='viridis')
            plt.title('Model Accuracy Comparison')
            plt.xlim(0, 1)
            st.pyplot(fig)
        
        # Feature importance across models
        st.subheader("Feature Importance Across Models")
        importance_dfs = {}
        
        for name, model_info in models.items():
            if name == "Logistic Regression":
                coefs = pd.DataFrame(model_info["model"].coef_[0], index=X.columns, columns=['Coefficient'])
                coefs['abs'] = coefs['Coefficient'].abs()
                importance_dfs[name] = coefs['abs'].rename(name)
            elif name in ["Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost"]:
                if hasattr(model_info["model"], 'feature_importances_'):
                    importances = pd.DataFrame(model_info["model"].feature_importances_, index=X.columns, columns=['importance'])
                    importance_dfs[name] = importances['importance']
        
        # Normalized heatmap
        if importance_dfs:
            combined_importance = pd.concat(importance_dfs, axis=1)
            combined_importance = combined_importance.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
            
            fig2 = plt.figure(figsize=(12, 8))
            sns.heatmap(combined_importance, annot=True, cmap='viridis', fmt='.2f')
            plt.title('Normalized Feature Importance Across Models')
            st.pyplot(fig2)

    # Individual model tabs
    for i, (name, model_info) in enumerate(models.items(), start=1):
        with tabs[i]:
            st.header(f"{name} Visualization")
            
            # Model-specific visualizations
            if name == "Logistic Regression":
                st.subheader("Coefficients")
                coefs = pd.DataFrame(model_info["model"].coef_[0], index=X.columns, columns=['Coefficient'])
                coefs['Absolute Value'] = coefs['Coefficient'].abs()
                st.dataframe(coefs.sort_values('Absolute Value', ascending=False))
                
                fig = plt.figure(figsize=(10, 6))
                sns.barplot(x='Coefficient', y=coefs.index, data=coefs.sort_values('Coefficient', ascending=False))
                plt.title('Logistic Regression Coefficients')
                st.pyplot(fig)
                
            elif name == "Decision Tree":
                st.subheader("Tree Visualization")
                fig = plt.figure(figsize=(20, 10))
                plot_tree(model_info["model"], filled=True, feature_names=X.columns, 
                         class_names=df['Risk_Level'].unique(), max_depth=3)
                st.pyplot(fig)
                
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame(model_info["model"].feature_importances_, 
                                           index=X.columns, columns=['Importance'])
                importance_df = importance_df.sort_values('Importance', ascending=False)
                
                fig2 = plt.figure(figsize=(10, 6))
                sns.barplot(x='Importance', y=importance_df.index, data=importance_df)
                plt.title('Decision Tree Feature Importance')
                st.pyplot(fig2)
                
            elif name in ["Random Forest", "Gradient Boosting"]:
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame(model_info["model"].feature_importances_, 
                                           index=X.columns, columns=['Importance'])
                importance_df = importance_df.sort_values('Importance', ascending=False)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(importance_df.style.format({'Importance': '{:.4f}'}))
                
                with col2:
                    fig = plt.figure(figsize=(10, 6))
                    sns.barplot(x='Importance', y=importance_df.index, data=importance_df)
                    plt.title(f'{name} Feature Importance')
                    st.pyplot(fig)
                
                st.subheader("Permutation Importance")
                result = permutation_importance(model_info["model"], X_test, y_test, 
                                             n_repeats=10, random_state=42)
                perm_importance = pd.DataFrame(result.importances_mean, 
                                             index=X.columns, columns=['Importance'])
                perm_importance = perm_importance.sort_values('Importance', ascending=False)
                
                fig2 = plt.figure(figsize=(10, 6))
                sns.barplot(x='Importance', y=perm_importance.index, data=perm_importance)
                plt.title(f'{name} Permutation Importance')
                st.pyplot(fig2)
                
            elif name == "XGBoost":
                st.subheader("Feature Importance")
                fig = plt.figure(figsize=(10, 6))
                plot_importance(model_info["model"], importance_type='weight', height=0.8, ax=fig.gca())
                plt.title('XGBoost Feature Importance (Weight)')
                st.pyplot(fig)
                
                st.subheader("Gain Importance")
                fig2 = plt.figure(figsize=(10, 6))
                plot_importance(model_info["model"], importance_type='gain', height=0.8, ax=fig2.gca())
                plt.title('XGBoost Feature Importance (Gain)')
                st.pyplot(fig2)
            
            elif name == "Neural Network":
                st.subheader("Training History")
                fig, axs = plt.subplots(1, 2, figsize=(12, 4))
                axs[0].plot(model_info["history"].history['accuracy'], label='Train Accuracy')
                axs[0].plot(model_info["history"].history['val_accuracy'], label='Val Accuracy')
                axs[0].set_title('Accuracy over epochs')
                axs[0].legend()

                axs[1].plot(model_info["history"].history['loss'], label='Train Loss')
                axs[1].plot(model_info["history"].history['val_loss'], label='Val Loss')
                axs[1].set_title('Loss over epochs')
                axs[1].legend()
                st.pyplot(fig)
                
                st.subheader("Performance Metrics")
                nn_loss, nn_acc = model_info["model"].evaluate(X_test_scaled, y_test_nn, verbose=0)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{nn_acc:.2%}")
                    st.metric("Loss", f"{nn_loss:.4f}")
                
                with col2:
                    y_pred = np.argmax(model_info["model"].predict(X_test_scaled), axis=1)
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm = plt.figure(figsize=(6, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                xticklabels=df['Risk_Level'].unique(), 
                                yticklabels=df['Risk_Level'].unique())
                    plt.title('Confusion Matrix')
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    st.pyplot(fig_cm)
            
            # Common visualizations for sklearn models
            if model_info["type"] == "sklearn":
                st.subheader("Performance Metrics")
                X_input = X_test_scaled if model_info["needs_scaling"] else X_test
                y_pred = model_info["model"].predict(X_input)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
                    
                with col2:
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm = plt.figure(figsize=(6, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                xticklabels=df['Risk_Level'].unique(), 
                                yticklabels=df['Risk_Level'].unique())
                    plt.title('Confusion Matrix')
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    st.pyplot(fig_cm)
                
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, 
                                             target_names=df['Risk_Level'].unique(), 
                                             output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

    # ===== Prediction Interface =====
    st.sidebar.header("Model Prediction")
    selected_model = st.sidebar.selectbox(
        "Choose a model:",
        options=list(models.keys())
    )

    st.sidebar.subheader("Input Features")
    input_features = {}
    for feature in X.columns:
        min_val = float(X[feature].min())
        max_val = float(X[feature].max())
        default_val = float(X[feature].median())
        input_features[feature] = st.sidebar.slider(
            f"{feature}",
            min_value=min_val,
            max_value=max_val,
            value=default_val
        )

    if st.sidebar.button("Predict"):
        input_df = pd.DataFrame([input_features])
        model_info = models[selected_model]
        
        if model_info["needs_scaling"]:
            input_scaled = scaler.transform(input_df)
        else:
            input_scaled = input_df
        
        if model_info["type"] == "keras":
            pred_probs = model_info["model"].predict(input_scaled)
            pred = np.argmax(pred_probs, axis=1)
        else:
            pred = model_info["model"].predict(input_scaled)
        
        risk_levels = dict(enumerate(df['Risk_Level'].astype('category').cat.categories))
        predicted_risk = risk_levels[pred[0]]
        
        st.sidebar.success(f"Predicted Risk Level: **{predicted_risk}**")
        st.sidebar.write(f"Using model: **{selected_model}**")
# Add some styling
st.markdown("""
<style>
    .main {
        max-width: 1200px;
        padding: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    h1 {
        color: #dc3545;
    }
    h2 {
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)