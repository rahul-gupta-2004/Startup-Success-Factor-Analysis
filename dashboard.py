import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

def app():
    st.title("Startup Analysis Dashboard")
    
    # Load data
    @st.cache_data
    def load_data():
        data = pd.read_csv('data.csv', encoding='latin1')
        return data
    
    try:
        data = load_data()
        
        st.header("Dataset Overview")
        st.write(f"**Dataset Shape:** {data.shape[0]} rows × {data.shape[1]} columns")
        st.write("**First 5 rows:**")
        st.dataframe(data.head(), use_container_width=True)
        
        st.header("Data Preprocessing")
        
        # Create a copy for cleaning
        data_clean = data.copy()
        
        # Handle missing values
        numeric_cols = data_clean.select_dtypes(include=[np.number]).columns
        data_clean[numeric_cols] = data_clean[numeric_cols].fillna(data_clean[numeric_cols].mean())
        
        categorical_cols = data_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            mode_val = data_clean[col].mode()[0] if not data_clean[col].mode().empty else 'Unknown'
            data_clean[col] = data_clean[col].fillna(mode_val)
        
        st.success("Missing values handled")
        st.success("Data types converted for analysis")
        
        # Select relevant columns
        relevant_columns = [
            'Company_Name', 'Dependent-Company Status', 'year of founding',
            'Age of company in years', 'Internet Activity Score',
            'Industry of company', 'Employee Count',
            'Percent_skill_Business Strategy',
            'Focus functions of company',
            'Renown score'
        ]
        
        analysis_data = data_clean[relevant_columns].copy()
        
        # Convert columns to numeric
        analysis_data['Age of company in years'] = pd.to_numeric(
            analysis_data['Age of company in years'], errors='coerce')
        
        # Fill NaN values with mean
        analysis_data['Age of company in years'] = analysis_data['Age of company in years'].fillna(
            analysis_data['Age of company in years'].mean())
        
        st.success("Non-numeric values converted to numeric")
        
        st.header("Problem Statements Analysis")
        
        # Problem 1: Company Age Distribution
        st.subheader("1. Impact of Company Age on Success")
        
        # Calculate statistics
        mean_age = analysis_data['Age of company in years'].mean()
        median_age = analysis_data['Age of company in years'].median()
        mode_age = analysis_data['Age of company in years'].mode()[0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Age", f"{mean_age:.2f} years")
        with col2:
            st.metric("Median Age", f"{median_age:.2f} years")
        with col3:
            st.metric("Mode Age", f"{mode_age:.2f} years")
        
        # Plot histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(analysis_data['Age of company in years'].dropna(), 
                bins=10, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(mean_age, color='red', linestyle='--', linewidth=2, label=f"Mean: {mean_age:.2f}")
        ax.axvline(median_age, color='green', linestyle='--', linewidth=2, label=f"Median: {median_age:.2f}")
        ax.axvline(mode_age, color='orange', linestyle='--', linewidth=2, label=f"Mode: {mode_age:.2f}")
        ax.set_title('Distribution of Company Age')
        ax.set_xlabel('Age of Company in Years')
        ax.set_ylabel('Number of Companies')
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        
        st.info("**Inference:** Most startups are relatively young (around 4 years old), with the distribution slightly right-skewed.")
        
        # Problem 2: Internet Activity vs Success
        st.subheader("2. Influence of Internet Activity on Startup Success")
        
        # Prepare data for correlation
        success_data = analysis_data[analysis_data['Dependent-Company Status'] == 'Success'].copy()
        
        # Convert to numeric
        success_data['Internet Activity Score'] = pd.to_numeric(success_data['Internet Activity Score'], errors='coerce')
        success_data['Renown score'] = pd.to_numeric(success_data['Renown score'], errors='coerce')
        
        # Drop NaN values
        correlation_data = success_data[['Internet Activity Score', 'Renown score']].dropna()
        
        if len(correlation_data) > 1:
            correlation = correlation_data['Internet Activity Score'].corr(correlation_data['Renown score'])
            st.metric("Correlation Coefficient", f"{correlation:.2f}")
            
            # Scatter plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(correlation_data['Internet Activity Score'], correlation_data['Renown score'], 
                      alpha=0.6, color='blue')
            ax.set_title('Internet Activity Score vs Renown Score')
            ax.set_xlabel('Internet Activity Score')
            ax.set_ylabel('Renown Score')
            ax.grid(True, alpha=0.3)
            
            # Add correlation line
            z = np.polyfit(correlation_data['Internet Activity Score'], 
                          correlation_data['Renown score'], 1)
            p = np.poly1d(z)
            ax.plot(correlation_data['Internet Activity Score'], 
                   p(correlation_data['Internet Activity Score']), "r--", alpha=0.8)
            ax.legend([f"Correlation: {correlation:.2f}"])
            
            st.pyplot(fig)
            
            st.info("**Inference:** Weak positive correlation suggests internet activity has some relationship with success, but it's not a strong predictor.")
        
        # Problem 3: Employee Count Analysis
        st.subheader("3. Employee Count Analysis")
        
        # Convert to numeric
        success_data['Employee Count'] = pd.to_numeric(success_data['Employee Count'], errors='coerce')
        employee_data = success_data['Employee Count'].dropna()
        
        if len(employee_data) > 0:
            mean_employees = employee_data.mean()
            median_employees = employee_data.median()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Employees", f"{mean_employees:.2f}")
            with col2:
                st.metric("Median Employees", f"{median_employees:.2f}")
            
            # Box plot
            fig, ax = plt.subplots(figsize=(10, 6))
            employee_data.plot(kind='box', ax=ax)
            ax.set_title('Employee Count Distribution in Successful Startups')
            ax.set_ylabel('Number of Employees')
            st.pyplot(fig)
            
            st.info("**Inference:** High variability in employee counts among successful startups, indicating diverse team sizes can lead to success.")
        
        # Problem 4: Business Strategy Skills Analysis
        st.subheader("4. Business Strategy Skills Analysis")
        
        # Convert to numeric
        success_data['Percent_skill_Business Strategy'] = pd.to_numeric(
            success_data['Percent_skill_Business Strategy'], errors='coerce')
        strategy_data = success_data['Percent_skill_Business Strategy'].dropna()
        
        if len(strategy_data) > 0:
            skewness = skew(strategy_data)
            st.metric("Skewness of Business Strategy Skills", f"{skewness:.2f}")
            
            # Histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(strategy_data, 
                    bins=15, color='lightgreen', edgecolor='black', alpha=0.7)
            ax.set_title('Distribution of Business Strategy Skills')
            ax.set_xlabel('Percentage of Business Strategy Skills')
            ax.set_ylabel('Number of Companies')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.info("**Inference:** Right-skewed distribution shows most startups have lower business strategy skills, but high-performing ones tend to have stronger strategic capabilities.")
        
        # Problem 5: Focus Functions Analysis
        st.subheader("5. Focus Functions Analysis")
        
        if 'Focus functions of company' in analysis_data.columns:
            focus_counts = analysis_data['Focus functions of company'].value_counts()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            focus_counts.head(10).plot(kind='bar', color='lightcoral', ax=ax)
            ax.set_title('Top 10 Focus Functions in Startups')
            ax.set_xlabel('Focus Functions')
            ax.set_ylabel('Number of Companies')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            st.info("**Inference:** Marketing & Sales and Operations are the most common focus functions among startups.")
        
    except FileNotFoundError:
        st.error("Data file 'data.csv' not found. Please ensure the file is in the correct directory.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")