import os
import pandas as pd
import numpy as np
import pickle
import json
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from werkzeug.utils import secure_filename

# Import utility modules
from utils.data_processor import DataProcessor
from utils.eda import EDAGenerator
from utils.visualizer import Visualizer
from utils.model_trainer import ModelTrainer

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # For session management
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize utility classes
data_processor = DataProcessor()
eda_generator = EDAGenerator()
visualizer = Visualizer()
model_trainer = ModelTrainer()

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the uploaded file
            df = data_processor.load_data(filepath)
            
            # Store data info in session
            session['filepath'] = filepath
            session['columns'] = df.columns.tolist()
            session['data_shape'] = df.shape
            
            return redirect(url_for('eda'))
    
    return render_template('upload.html')

@app.route('/eda')
def eda():
    if 'filepath' not in session:
        return redirect(url_for('upload'))
    
    filepath = session.get('filepath')
    df = data_processor.load_data(filepath)
    
    # Generate EDA information
    summary_stats = eda_generator.get_summary_stats(df)
    missing_values = eda_generator.get_missing_values(df)
    data_types = eda_generator.get_data_types(df)
    correlations = eda_generator.get_correlations(df)
    
    return render_template('eda.html', 
                           summary_stats=summary_stats,
                           missing_values=missing_values,
                           data_types=data_types,
                           correlations=correlations,
                           columns=session.get('columns'),
                           data_shape=session.get('data_shape'))

@app.route('/visualize')
def visualize():
    if 'filepath' not in session:
        return redirect(url_for('upload'))
    
    filepath = session.get('filepath')
    columns = session.get('columns')
    
    return render_template('visualize.html', columns=columns)

@app.route('/api/visualize', methods=['POST'])
def api_visualize():
    if 'filepath' not in session:
        return jsonify({'error': 'No data loaded'}), 400
    
    filepath = session.get('filepath')
    df = data_processor.load_data(filepath)
    
    chart_type = request.json.get('chart_type')
    x_column = request.json.get('x_column')
    y_column = request.json.get('y_column')
    
    # Generate visualization
    chart_data = visualizer.generate_chart(df, chart_type, x_column, y_column)
    
    return jsonify(chart_data)

@app.route('/churn_prediction')
def churn_prediction():
    if 'filepath' not in session:
        return redirect(url_for('upload'))
    
    filepath = session.get('filepath')
    columns = session.get('columns')
    
    return render_template('churn_prediction.html', columns=columns)

@app.route('/api/train_churn_model', methods=['POST'])
def api_train_churn_model():
    if 'filepath' not in session:
        return jsonify({'error': 'No data loaded'}), 400
    
    filepath = session.get('filepath')
    df = data_processor.load_data(filepath)
    
    target_column = request.json.get('target_column')
    feature_columns = request.json.get('feature_columns', [])
    model_type = request.json.get('model_type', 'random_forest')  # Default to random forest
    
    # Train model
    model, metrics = model_trainer.train_model(
        df, feature_columns, target_column, model_type=model_type
    )
    
    # Save model
    model_path = os.path.join('models', 'churn_model.pkl')
    os.makedirs('models', exist_ok=True)
    # Save model and model_trainer for class label mapping
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'trainer': model_trainer}, f)
    
    return jsonify({'metrics': metrics})

@app.route('/api/predict_churn', methods=['POST'])
def api_predict_churn():
    # Load the model
    model_path = os.path.join('models', 'churn_model.pkl')
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model not trained yet'}), 400
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        # Restore model_trainer with class mapping information
        model_trainer.__dict__.update(model_data['trainer'].__dict__)
    
    # Get data for prediction
    data = request.json.get('data', {})
    
    # Make prediction
    prediction = model_trainer.predict(model, data)
    
    return jsonify({'prediction': prediction})

@app.route('/sales_forecast')
def sales_forecast():
    if 'filepath' not in session:
        return redirect(url_for('upload'))
    
    filepath = session.get('filepath')
    columns = session.get('columns')
    
    return render_template('sales_forecast.html', columns=columns)

@app.route('/api/train_sales_model', methods=['POST'])
def api_train_sales_model():
    if 'filepath' not in session:
        return jsonify({'error': 'No data loaded'}), 400
    
    filepath = session.get('filepath')
    df = data_processor.load_data(filepath)
    
    target_column = request.json.get('target_column')
    feature_columns = request.json.get('feature_columns', [])
    time_column = request.json.get('time_column')
    forecast_periods = request.json.get('forecast_periods', 12)  # Default to 12 periods
    model_type = request.json.get('model_type', 'xgboost')  # Default to XGBoost
    
    # Train model
    model, metrics = model_trainer.train_forecast_model(
        df, feature_columns, target_column, time_column, 
        forecast_periods=forecast_periods, model_type=model_type
    )
    
    # Save model
    model_path = os.path.join('models', 'sales_model.pkl')
    os.makedirs('models', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return jsonify({'metrics': metrics})

@app.route('/api/predict_sales', methods=['POST'])
def api_predict_sales():
    # Load the model
    model_path = os.path.join('models', 'sales_model.pkl')
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model not trained yet'}), 400
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Get parameters for forecast
    forecast_periods = request.json.get('forecast_periods', 12)
    
    # Get data for prediction context if needed
    data = request.json.get('data', {})
    
    # Make prediction
    forecast = model_trainer.forecast(model, forecast_periods, data)
    
    return jsonify({'forecast': forecast})

if __name__ == '__main__':
    app.run(debug=True)
