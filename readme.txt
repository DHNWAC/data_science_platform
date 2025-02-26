# Data Science Platform MVP

An end-to-end data science platform for exploratory data analysis (EDA), data visualization, churn prediction, and sales forecasting.

## Features

- **Data Upload**: Upload CSV or Excel files for analysis
- **Exploratory Data Analysis (EDA)**: Get insights about your data with summary statistics, missing values, and correlations
- **Data Visualization**: Create interactive charts with Plotly
- **Churn Prediction**: Build and evaluate machine learning models to predict customer churn
- **Sales Forecasting**: Create time series forecasts for future sales

## Technology Stack

- **Backend**: Flask
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5

## Project Structure

```
data_science_platform/
├── app.py              # Main Flask application
├── requirements.txt    # Dependencies
├── static/             # Static files (CSS, JS)
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
├── templates/          # HTML templates
│   ├── index.html
│   ├── upload.html
│   ├── eda.html
│   ├── visualize.html
│   ├── churn_prediction.html
│   └── sales_forecast.html
├── models/             # Trained ML models
│   ├── churn_model.pkl
│   └── sales_model.pkl
└── utils/              # Utility functions
    ├── __init__.py
    ├── data_processor.py
    ├── eda.py
    ├── visualizer.py
    └── model_trainer.py
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd data_science_platform
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and go to:
```
http://127.0.0.1:5000/
```

## Usage Guide

### 1. Upload Data
- Go to the Upload page
- Select a CSV or Excel file from your computer
- Click "Upload & Analyze"

### 2. Exploratory Data Analysis
- View summary statistics, missing values, and correlations
- Understand the structure and quality of your data

### 3. Data Visualization
- Select chart types: bar, line, scatter, histogram, etc.
- Choose columns for X and Y axes
- Generate interactive visualizations

### 4. Churn Prediction
- Select target column (churn indicator)
- Choose features for prediction
- Select model type (Random Forest or XGBoost)
- Train the model and evaluate performance
- Make predictions for new customers

### 5. Sales Forecasting
- Select time column and target column (sales)
- Choose additional features if needed
- Set forecast periods
- Train the model and generate forecasts
- View and download forecast results

## Model Information

### Churn Prediction
The platform uses classification models to predict customer churn:
- **Random Forest**: Good for handling non-linear relationships and feature interactions
- **XGBoost**: High performance gradient boosting for structured data

### Sales Forecasting
The platform uses regression models with time-series features:
- Creates lagged features and time-based attributes
- Supports XGBoost and Random Forest for forecasting
- Provides metrics like RMSE, MAE, and R²

## Future Enhancements

- User authentication and multi-user support
- Data preprocessing tools (handling missing values, encoding categorical features)
- More advanced visualization options
- Additional machine learning models
- Model deployment and API integration
- Scheduled retraining and monitoring
- PDF report generation

## License

This project is licensed under the MIT License - see the LICENSE file for details.
