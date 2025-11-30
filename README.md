# Startup Analysis Dashboard

A Streamlit web application for analysing startup data and performing statistical analysis.

## Project Overview

This project provides an interactive dashboard to analyse startup company data, including:

- Company age distribution

- Internet activity impact on success

- Employee count analysis

- Business strategy skills analysis

- Linear regression modelling

## Features

### Dashboard

- Dataset overview and preprocessing

- 5 key problem statements with visualisations

- Statistical analysis and insights

- Interactive charts and graphs

### Model

- Linear regression analysis

- Real-time predictions

- Model performance metrics

- Interactive input for predictions

## Installation

1\. Clone the repository:

```

git clone https://github.com/yourusername/startup-analysis.git

cd startup-analysis

```

2\. Install required packages:

```

pip install streamlit pandas numpy matplotlib scikit-learn scipy streamlit-option-menu

```

3\. Run the application:

```

streamlit run main.py

```

## Project Structure

```

startup-analysis/

├── main.py              # Main application entry point

├── dashboard.py         # Dashboard tab implementation

├── model.py            # Linear regression model tab

├── data.csv            # Dataset file

└── README.md           # Project documentation

```

## Usage

1\. After running `streamlit run main.py`, open your browser to the local URL shown in the terminal (usually http://localhost:8501)

2\. Navigate between the Dashboard and Model tabs using the sidebar

3\. On the Dashboard tab, explore the 5 problem statements with visualisations

4\. On the Model tab, use the input field to predict Renown Scores based on Internet Activity

## Data Analysis

The dashboard analyses:

- Company Age: Distribution and statistics of startup ages

- Internet Activity: Correlation with success metrics

- Employee Count: Patterns in successful startups

- Business Strategy: Skills distribution analysis

- Linear Regression: Predict renown scores from internet activity

## Technologies Used

- Streamlit - Web application framework

- Pandas - Data manipulation and analysis

- NumPy - Numerical computing

- Matplotlib - Data visualisation

- Scikit-learn - Machine learning models

- SciPy - Statistical analysis

## Deployment

To deploy on Streamlit Cloud:

1\. Push your code to GitHub

2\. Go to https://share.streamlit.io/

3\. Connect your repository

4\. Deploy!
