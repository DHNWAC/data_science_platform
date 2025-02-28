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
    color_by = request.json.get('color_by')
    
    # Generate visualization
    chart_data = visualizer.generate_chart(df, chart_type, x_column, y_column, color=color_by)
    
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
    model_type = request.json.get('model_type', 'xgboost')  # Default to XGBoost
    
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
    try:
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
        
        # Return the prediction
        return jsonify({'prediction': prediction})
    
    except Exception as e:
        # Log the error for debugging
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': f"An error occurred during prediction: {str(e)}"}), 500

# Load sample data if user doesn't have their own
@app.route('/api/load_sample_data', methods=['POST'])
def api_load_sample_data():
    # Copy the sample churn dataset to the uploads folder
    try:
        sample_path = os.path.join('sample_data', 'customer_churn_datasettestingmaster.csv')
        target_path = os.path.join(app.config['UPLOAD_FOLDER'], 'customer_churn_sample.csv')
        
        # Ensure sample_data directory exists
        if not os.path.exists(sample_path):
            return jsonify({'error': 'Sample data not found'}), 404
        
        # Copy the file
        import shutil
        shutil.copy(sample_path, target_path)
        
        # Process the sample file
        df = data_processor.load_data(target_path)
        
        # Store data info in session
        session['filepath'] = target_path
        session['columns'] = df.columns.tolist()
        session['data_shape'] = df.shape
        
        return jsonify({'success': True, 'message': 'Sample data loaded successfully'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# New routes for demo request and pricing pages
@app.route('/demo', methods=['GET', 'POST'])
def demo():
    if request.method == 'POST':
        # In a real application, you would process the form submission here
        # For example, save the demo request to a database and send an email notification
        
        # For this MVP, we'll just return the same template 
        # (the success message is handled client-side with JavaScript)
        return render_template('demo.html')
    
    return render_template('demo.html')

@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

if __name__ == '__main__':
    app.run(debug=True)