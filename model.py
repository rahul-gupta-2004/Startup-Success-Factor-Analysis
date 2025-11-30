import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection
from sklearn.metrics import mean_squared_error, r2_score

def app():
    st.title("Linear Regression Analysis")
    
    st.header("What is Linear Regression?")
    st.write("A statistical method that models the relationship between a dependent variable and one or more independent variables using a linear equation.")
    
    # Load and prepare data
    @st.cache_data
    def load_data():
        data = pd.read_csv('data.csv', encoding='latin1')
        return data
    
    try:
        data = load_data()
        
        # Data preprocessing for regression
        regression_data = data[['Internet Activity Score', 'Renown score']].copy()
        
        # Convert to numeric and handle non-numeric values
        regression_data['Internet Activity Score'] = pd.to_numeric(
            regression_data['Internet Activity Score'], errors='coerce')
        regression_data['Renown score'] = pd.to_numeric(
            regression_data['Renown score'], errors='coerce')
        
        # Drop rows with missing values
        regression_data = regression_data.dropna()
        
        st.header("Data Preparation")
        st.write(f"Original data points: {len(data)}")
        st.write(f"After cleaning: {len(regression_data)} valid data points")
        
        if len(regression_data) > 0:
            # Prepare features and target
            X = regression_data[['Internet Activity Score']].values
            y = regression_data['Renown score'].values
            
            # Split data
            X_train, X_test, y_train, y_test = model_selection.train_test_split(
                X, y, test_size=0.2, random_state=42)
            
            # Create and train model
            model = linear_model.LinearRegression()
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Get coefficients
            intercept = model.intercept_
            coefficient = model.coef_[0]
            
            st.header("Regression Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{r2:.4f}")
            with col2:
                st.metric("RMSE", f"{rmse:.4f}")
            with col3:
                st.metric("Slope", f"{coefficient:.4f}")
            
            st.write(f"**Regression Equation:** Renown Score = {coefficient:.4f} × Internet Activity + {intercept:.4f}")
            
            # Plot regression
            st.subheader("Regression Visualization")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Scatter plot with regression line
            ax.scatter(X, y, alpha=0.6, color='blue', label='Actual Data')
            
            # Create line for regression
            x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_line = model.predict(x_line)
            ax.plot(x_line, y_line, color='red', linewidth=2, 
                    label=f'y = {coefficient:.4f}x + {intercept:.4f}')
            
            ax.set_xlabel('Internet Activity Score')
            ax.set_ylabel('Renown Score')
            ax.set_title('Linear Regression: Internet Activity vs Renown Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            st.header("Interpretation")
            
            st.subheader("What the Numbers Mean:")
            st.write(f"""
            - **R² Score ({r2:.4f})**: Very close to zero, meaning only {r2*100:.2f}% of the variation in Renown Score is explained by Internet Activity
            - **Slope ({coefficient:.4f})**: Almost zero, suggesting Internet Activity has very little effect on Renown Score
            - **RMSE ({rmse:.4f})**: Average prediction error is {rmse:.2f} units
            """)
            
            st.subheader("Simple Explanation:")
            st.write("""
            The linear regression shows that **Internet Activity Score alone is not a good predictor** of a startup's Renown Score. 
            
            The nearly flat line and R² close to zero tell us that:
            - High internet activity doesn't guarantee high renown
            - Other factors likely influence success more strongly
            - Startups should focus on multiple success factors, not just online presence
            """)
            
        else:
            st.error("Insufficient data for regression analysis after cleaning.")
            
    except FileNotFoundError:
        st.error("Data file 'data.csv' not found.")
        
        # Demo with sample data
        st.info("Showing demo with sample data:")
        
        # Generate sample data
        np.random.seed(42)
        n_samples = 100
        X_demo = np.random.normal(100, 50, n_samples)
        y_demo = 3.5 + 0.001 * X_demo + np.random.normal(0, 2, n_samples)
        
        # Create and train model
        X_demo = X_demo.reshape(-1, 1)
        model_demo = linear_model.LinearRegression()
        model_demo.fit(X_demo, y_demo)
        
        # Predictions and metrics
        y_pred_demo = model_demo.predict(X_demo)
        r2_demo = r2_score(y_demo, y_pred_demo)
        
        st.write(f"**Demo Regression Equation:** y = {model_demo.coef_[0]:.4f}x + {model_demo.intercept_:.4f}")
        st.write(f"**Demo R²:** {r2_demo:.4f}")
        
        st.warning("This is demo data. For actual analysis, please provide the data.csv file.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")