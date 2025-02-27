import os
import pandas as pd
import numpy as np
import pickle
import json
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from werkzeug.utils import secure_filename
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

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

# Initialize LLM for auto visualization
# Note: In production, you would want to initialize this on-demand or use a service
def initialize_llm():
    try:
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")  # Smaller model for demonstration
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        return tokenizer, model
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return None, None

tokenizer, model = initialize_llm()

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
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return jsonify({'metrics': metrics})

@app.route('/api/predict_churn', methods=['POST'])
def api_predict_churn():
    # Load the model
    model_path = os.path.join('models', 'churn_model.pkl')
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model not trained yet'}), 400
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
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

# NEW ROUTES BELOW

# New route for pricing page
@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

# New route for general classification page (replacing churn_prediction)
@app.route('/classification')
def classification():
    if 'filepath' not in session:
        return redirect(url_for('upload'))
    
    filepath = session.get('filepath')
    columns = session.get('columns')
    
    return render_template('classification.html', columns=columns)

# New route for GLM analysis
@app.route('/glm')
def glm_analysis():
    if 'filepath' not in session:
        return redirect(url_for('upload'))
    
    filepath = session.get('filepath')
    columns = session.get('columns')
    
    return render_template('glm.html', columns=columns)

# New route for auto visualization
@app.route('/auto_visualize')
def auto_visualize():
    if 'filepath' not in session:
        return redirect(url_for('upload'))
    
    filepath = session.get('filepath')
    columns = session.get('columns')
    
    return render_template('auto_visualize.html', columns=columns)

# API endpoint for general classification
@app.route('/api/train_classification_model', methods=['POST'])
def api_train_classification_model():
    if 'filepath' not in session:
        return jsonify({'error': 'No data loaded'}), 400
    
    filepath = session.get('filepath')
    df = data_processor.load_data(filepath)
    
    # Parse request data
    target_column = request.json.get('target_column')
    problem_type = request.json.get('problem_type', 'binary')  # 'binary' or 'multiclass'
    feature_columns = request.json.get('feature_columns', [])
    model_type = request.json.get('model_type', 'random_forest')
    hyperparameters = request.json.get('hyperparameters', {})
    
    # Train model with the appropriate parameters
    if model_type == 'random_forest':
        n_estimators = hyperparameters.get('n_estimators', 100)
        max_depth = hyperparameters.get('max_depth', 10)
        model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'random_state': 42
        }
    elif model_type == 'xgboost':
        n_estimators = hyperparameters.get('n_estimators', 100)
        learning_rate = hyperparameters.get('learning_rate', 0.1)
        max_depth = hyperparameters.get('max_depth', 6)
        model_params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'random_state': 42
        }
    elif model_type == 'logistic':
        C = hyperparameters.get('C', 1.0)
        penalty = hyperparameters.get('penalty', 'l2')
        model_params = {
            'C': C,
            'penalty': penalty,
            'random_state': 42
        }
    
    try:
        model, metrics = model_trainer.train_classification_model(
            df, feature_columns, target_column, 
            problem_type=problem_type,
            model_type=model_type,
            model_params=model_params
        )
        
        # Save model metadata
        model_metadata = {
            'target_column': target_column,
            'feature_columns': feature_columns,
            'problem_type': problem_type,
            'model_type': model_type
        }
        
        with open(os.path.join('models', 'classification_metadata.json'), 'w') as f:
            json.dump(model_metadata, f)
        
        return jsonify({'metrics': metrics})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API endpoint for prediction with classification model
@app.route('/api/predict_classification', methods=['POST'])
def api_predict_classification():
    try:
        # Load model metadata
        with open(os.path.join('models', 'classification_metadata.json'), 'r') as f:
            model_metadata = json.load(f)
            
        # Load the model
        model_path = os.path.join('models', 'classification_model.pkl')
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model not trained yet'}), 400
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Get data for prediction
        data = request.json.get('data', {})
        problem_type = request.json.get('problem_type', model_metadata.get('problem_type', 'binary'))
        
        # Make prediction
        prediction = model_trainer.predict_classification(
            model_data, 
            data, 
            problem_type=problem_type
        )
        
        return jsonify({'prediction': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API endpoint for GLM model building
@app.route('/api/build_glm_model', methods=['POST'])
def api_build_glm_model():
    if 'filepath' not in session:
        return jsonify({'error': 'No data loaded'}), 400
    
    filepath = session.get('filepath')
    df = data_processor.load_data(filepath)
    
    # Parse request data
    target_column = request.json.get('target_column')
    feature_columns = request.json.get('feature_columns', [])
    distribution = request.json.get('distribution', 'gaussian')
    link_function = request.json.get('link_function', 'identity')
    alpha = request.json.get('alpha', 0.001)
    penalty = request.json.get('penalty', 'l2')
    l1_ratio = request.json.get('l1_ratio', 0.5)
    include_interaction = request.json.get('include_interaction', False)
    
    try:
        # Prepare formula for statsmodels
        formula = f"{target_column} ~ "
        
        # Add main effects
        main_effects = " + ".join(feature_columns)
        formula += main_effects
        
        # Add interaction terms if requested
        if include_interaction and len(feature_columns) > 1:
            interactions = []
            for i in range(len(feature_columns)):
                for j in range(i+1, len(feature_columns)):
                    interactions.append(f"{feature_columns[i]}:{feature_columns[j]}")
            
            if interactions:
                formula += " + " + " + ".join(interactions)
        
        # Prepare data
        model_df = df[feature_columns + [target_column]].copy()
        
        # Handle missing values
        model_df = model_df.dropna()
        
        # Fit the GLM model
        if distribution == 'gaussian':
            family = sm.families.Gaussian(link=getattr(sm.families.links, link_function.capitalize()))
        elif distribution == 'binomial':
            family = sm.families.Binomial(link=getattr(sm.families.links, link_function.capitalize()))
        elif distribution == 'poisson':
            family = sm.families.Poisson(link=getattr(sm.families.links, link_function.capitalize()))
        elif distribution == 'gamma':
            family = sm.families.Gamma(link=getattr(sm.families.links, link_function.capitalize()))
        
        # Fit the model
        glm_model = smf.glm(formula=formula, data=model_df, family=family)
        glm_result = glm_model.fit_regularized(alpha=alpha, L1_wt=l1_ratio if penalty == 'elasticnet' else (1.0 if penalty == 'l1' else 0.0))
        
        # Save the model
        model_path = os.path.join('models', 'glm_model.pkl')
        os.makedirs('models', exist_ok=True)
        glm_result.save(model_path)
        
        # Save metadata
        glm_metadata = {
            'target_column': target_column,
            'feature_columns': feature_columns,
            'distribution': distribution,
            'link_function': link_function,
            'formula': formula,
            'include_interaction': include_interaction
        }
        
        with open(os.path.join('models', 'glm_metadata.json'), 'w') as f:
            json.dump(glm_metadata, f)
        
        # Extract coefficients
        coefficients = {}
        for term in glm_result.params.index:
            coefficients[term] = {
                'value': float(glm_result.params[term]),
                'std_err': float(glm_result.bse[term]) if term in glm_result.bse else None,
                'p_value': float(glm_result.pvalues[term]) if term in glm_result.pvalues else None
            }
        
        # Prepare diagnostic information
        diagnostics = {
            'null_deviance': float(glm_result.null_deviance),
            'residual_deviance': float(glm_result.deviance),
            'degrees_of_freedom': int(glm_result.df_resid),
            'dispersion': float(glm_result.scale),
            'residuals': glm_result.resid_pearson.tolist(),
            'fitted': glm_result.fittedvalues.tolist()
        }
        
        # Prepare QQ plot data
        residuals = glm_result.resid_pearson
        qq_x, qq_y = stats.probplot(residuals, dist='norm', fit=False)
        diagnostics['qq_x'] = qq_x[0].tolist()
        diagnostics['qq_y'] = qq_y.tolist()
        
        # Prepare metrics
        metrics = {
            'aic': float(glm_result.aic),
            'bic': float(glm_result.bic),
            'log_likelihood': float(glm_result.llf),
            'deviance': float(glm_result.deviance),
            'r_squared': 1.0 - (glm_result.deviance / glm_result.null_deviance)
        }
        
        return jsonify({
            'coefficients': coefficients,
            'formula': formula,
            'diagnostics': diagnostics,
            'metrics': metrics
        })
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# API endpoint for GLM prediction
@app.route('/api/predict_glm', methods=['POST'])
def api_predict_glm():
    try:
        # Load model metadata
        with open(os.path.join('models', 'glm_metadata.json'), 'r') as f:
            glm_metadata = json.load(f)
        
        # Load model
        model_path = os.path.join('models', 'glm_model.pkl')
        if not os.path.exists(model_path):
            return jsonify({'error': 'No GLM model has been trained yet'}), 400
        
        glm_result = sm.load(model_path)
        
        # Get data for prediction
        data = request.json.get('data', {})
        
        # Convert to dataframe with single row
        pred_df = pd.DataFrame([data])
        
        # Make prediction
        prediction = glm_result.predict(pred_df)
        
        # Calculate confidence interval
        alpha = 0.05  # 95% confidence interval
        se = glm_result.get_prediction(pred_df).se_mean[0]
        t_value = stats.t.ppf(1 - alpha/2, glm_result.df_resid)
        lower_bound = prediction[0] - t_value * se
        upper_bound = prediction[0] + t_value * se
        
        return jsonify({
            'prediction': {
                'value': float(prediction[0]),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API endpoint for auto visualization
@app.route('/api/auto_visualize', methods=['POST'])
def api_auto_visualize():
    if 'filepath' not in session:
        return jsonify({'error': 'No data loaded'}), 400
    
    filepath = session.get('filepath')
    df = data_processor.load_data(filepath)
    
    # Parse request data
    analysis_goal = request.json.get('analysis_goal', '')
    selected_columns = request.json.get('selected_columns', [])
    visualization_count = request.json.get('visualization_count', 3)
    include_advanced_charts = request.json.get('include_advanced_charts', True)
    
    try:
        # Filter data to selected columns
        if selected_columns:
            filtered_df = df[selected_columns].copy()
        else:
            filtered_df = df.copy()
        
        # Generate visualizations based on the analysis goal
        # In a real implementation, this would use the LLM to suggest visualizations
        visualizations = generate_visualizations(
            filtered_df, 
            analysis_goal, 
            visualization_count, 
            include_advanced_charts
        )
        
        return jsonify({'visualizations': visualizations})
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Function to generate visualizations using LLM guidance
def generate_visualizations(df, analysis_goal, count=3, include_advanced=True):
    try:
        # Use LLM to suggest visualizations
        suggestions = generate_visualization_suggestions(df, analysis_goal, count, include_advanced)
        
        visualizations = []
        for i, suggestion in enumerate(suggestions):
            # Generate the visualization based on the suggestion
            chart_type = suggestion['chart_type']
            x_column = suggestion['x_column']
            y_column = suggestion.get('y_column')
            
            chart_data = {}
            
            # Create basic chart types
            if chart_type == 'bar':
                fig = px.bar(df, x=x_column, y=y_column, title=suggestion['title'])
                chart_data = fig.to_json()
            elif chart_type == 'line':
                fig = px.line(df, x=x_column, y=y_column, title=suggestion['title'])
                chart_data = fig.to_json()
            elif chart_type == 'scatter':
                fig = px.scatter(df, x=x_column, y=y_column, title=suggestion['title'])
                chart_data = fig.to_json()
            elif chart_type == 'histogram':
                fig = px.histogram(df, x=x_column, title=suggestion['title'])
                chart_data = fig.to_json()
            elif chart_type == 'box':
                fig = px.box(df, x=x_column, y=y_column, title=suggestion['title'])
                chart_data = fig.to_json()
            elif chart_type == 'heatmap' and include_advanced:
                # Create a correlation matrix for numerical columns
                corr_df = df.select_dtypes(include=['number']).corr()
                fig = px.imshow(corr_df, text_auto=True, title=suggestion['title'])
                chart_data = fig.to_json()
            elif chart_type == 'pie':
                # Get counts for categorical column
                counts = df[x_column].value_counts()
                fig = px.pie(names=counts.index, values=counts.values, title=suggestion['title'])
                chart_data = fig.to_json()
            
            visualizations.append({
                'title': suggestion['title'],
                'explanation': suggestion['explanation'],
                'chart_data': chart_data
            })
        
        return visualizations
    
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        # Fallback to simple visualizations if LLM fails
        return generate_fallback_visualizations(df, count)

# Function to use LLM to suggest visualizations
def generate_visualization_suggestions(df, analysis_goal, count=3, include_advanced=True):
    # Check if LLM is initialized
    if not (tokenizer and model):
        # Fallback to rule-based suggestions
        return generate_rule_based_suggestions(df, analysis_goal, count, include_advanced)
    
    try:
        # In a real implementation, this would prompt the LLM with:
        # 1. Analysis goal
        # 2. Column information (names, types, sample data)
        # 3. Ask for visualization recommendations
        
        # For this example, we'll use a rule-based approach instead
        return generate_rule_based_suggestions(df, analysis_goal, count, include_advanced)
        
    except Exception as e:
        print(f"Error using LLM for visualization suggestions: {e}")
        return generate_rule_based_suggestions(df, analysis_goal, count, include_advanced)

# Rule-based visualization suggestions as a fallback
def generate_rule_based_suggestions(df, analysis_goal, count=3, include_advanced=True):
    # Get column types
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]' or
                (df[col].dtype == 'object' and pd.to_datetime(df[col], errors='coerce').notna().all())]
    
    suggestions = []
    
    # Analyze the goal text for keywords
    goal_lower = analysis_goal.lower()
    
    # Time Series Analysis
    if any(kw in goal_lower for kw in ['time', 'trend', 'over time', 'temporal', 'date', 'month', 'year']):
        if date_cols and numeric_cols:
            suggestions.append({
                'chart_type': 'line',
                'x_column': date_cols[0],
                'y_column': numeric_cols[0],
                'title': f'Trend of {numeric_cols[0]} Over Time',
                'explanation': f'This line chart shows how {numeric_cols[0]} changes over time. Line charts are ideal for visualizing trends and patterns in time series data.'
            })
    
    # Correlations
    if any(kw in goal_lower for kw in ['correlate', 'correlation', 'relationship', 'between', 'against', 'versus']):
        if len(numeric_cols) >= 2:
            suggestions.append({
                'chart_type': 'scatter',
                'x_column': numeric_cols[0],
                'y_column': numeric_cols[1],
                'title': f'Relationship Between {numeric_cols[0]} and {numeric_cols[1]}',
                'explanation': f'This scatter plot shows the relationship between {numeric_cols[0]} and {numeric_cols[1]}. Scatter plots help identify correlations and patterns between two numerical variables.'
            })
        
        if include_advanced and len(numeric_cols) > 2:
            suggestions.append({
                'chart_type': 'heatmap',
                'x_column': None,  # Not used directly in correlation heatmaps
                'y_column': None,
                'title': 'Correlation Matrix of Numerical Variables',
                'explanation': 'This heatmap shows the correlation coefficients between all numerical variables. Darker blue indicates stronger positive correlation, while darker red indicates stronger negative correlation.'
            })
    
    # Distributions
    if any(kw in goal_lower for kw in ['distribution', 'spread', 'range', 'histogram', 'frequency']):
        if numeric_cols:
            suggestions.append({
                'chart_type': 'histogram',
                'x_column': numeric_cols[0],
                'title': f'Distribution of {numeric_cols[0]}',
                'explanation': f'This histogram shows the distribution of {numeric_cols[0]}. Histograms display the frequency distribution of a numerical variable, helping identify the central tendency, spread, and shape of the data.'
            })
        
        if numeric_cols and categorical_cols:
            suggestions.append({
                'chart_type': 'box',
                'x_column': categorical_cols[0],
                'y_column': numeric_cols[0],
                'title': f'Distribution of {numeric_cols[0]} by {categorical_cols[0]}',
                'explanation': f'This box plot shows the distribution of {numeric_cols[0]} across different categories of {categorical_cols[0]}. Box plots display the median, quartiles, and potential outliers, making it easy to compare distributions across groups.'
            })
    
    # Comparisons
    if any(kw in goal_lower for kw in ['compare', 'comparison', 'difference', 'versus', 'vs', 'against']):
        if categorical_cols and numeric_cols:
            suggestions.append({
                'chart_type': 'bar',
                'x_column': categorical_cols[0],
                'y_column': numeric_cols[0],
                'title': f'{numeric_cols[0]} by {categorical_cols[0]}',
                'explanation': f'This bar chart compares {numeric_cols[0]} across different categories of {categorical_cols[0]}. Bar charts are effective for comparing values across different categories.'
            })
        
        if categorical_cols:
            suggestions.append({
                'chart_type': 'pie',
                'x_column': categorical_cols[0],
                'title': f'Composition of {categorical_cols[0]}',
                'explanation': f'This pie chart shows the proportion of each category in {categorical_cols[0]}. Pie charts are useful for showing how a whole is divided into parts.'
            })
    
    # Ensure we have at least the requested number of suggestions
    while len(suggestions) < count and (numeric_cols or categorical_cols):
        # Add more generic visualizations
        if len(suggestions) < count and numeric_cols:
            if len(suggestions) % 3 == 0:
                suggestions.append({
                    'chart_type': 'histogram',
                    'x_column': numeric_cols[min(len(suggestions) % len(numeric_cols), len(numeric_cols)-1)],
                    'title': f'Distribution of {numeric_cols[min(len(suggestions) % len(numeric_cols), len(numeric_cols)-1)]}',
                    'explanation': f'This histogram shows the frequency distribution of {numeric_cols[min(len(suggestions) % len(numeric_cols), len(numeric_cols)-1)]}.'
                })
            elif len(suggestions) % 3 == 1 and len(numeric_cols) >= 2:
                idx1 = min(len(suggestions) % len(numeric_cols), len(numeric_cols)-1)
                idx2 = min((len(suggestions) + 1) % len(numeric_cols), len(numeric_cols)-1)
                if idx1 != idx2:
                    suggestions.append({
                        'chart_type': 'scatter',
                        'x_column': numeric_cols[idx1],
                        'y_column': numeric_cols[idx2],
                        'title': f'{numeric_cols[idx1]} vs {numeric_cols[idx2]}',
                        'explanation': f'This scatter plot shows the relationship between {numeric_cols[idx1]} and {numeric_cols[idx2]}.'
                    })
            elif categorical_cols:
                cat_idx = min(len(suggestions) % len(categorical_cols), len(categorical_cols)-1)
                num_idx = min(len(suggestions) % len(numeric_cols), len(numeric_cols)-1)
                suggestions.append({
                    'chart_type': 'bar',
                    'x_column': categorical_cols[cat_idx],
                    'y_column': numeric_cols[num_idx],
                    'title': f'{numeric_cols[num_idx]} by {categorical_cols[cat_idx]}',
                    'explanation': f'This bar chart compares {numeric_cols[num_idx]} across different categories of {categorical_cols[cat_idx]}.'
                })
        
        # Break if we can't add more suggestions
        if len(suggestions) == 0 or len(suggestions) >= count:
            break
    
    # Limit to the requested count
    return suggestions[:count]

# Generate fallback visualizations if everything else fails
def generate_fallback_visualizations(df, count=3):
    visualizations = []
    
    # Basic summary visualizations
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Add histogram for first numeric column
    if numeric_cols and len(visualizations) < count:
        fig = px.histogram(df, x=numeric_cols[0], title=f'Distribution of {numeric_cols[0]}')
        visualizations.append({
            'title': f'Distribution of {numeric_cols[0]}',
            'explanation': 'Automatically generated histogram showing the distribution of values.',
            'chart_data': fig.to_json()
        })
    
    # Add bar chart for first categorical column
    if categorical_cols and numeric_cols and len(visualizations) < count:
        fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0], title=f'{numeric_cols[0]} by {categorical_cols[0]}')
        visualizations.append({
            'title': f'{numeric_cols[0]} by {categorical_cols[0]}',
            'explanation': 'Automatically generated bar chart showing comparison across categories.',
            'chart_data': fig.to_json()
        })
    
    # Add correlation heatmap if multiple numeric columns
    if len(numeric_cols) > 1 and len(visualizations) < count:
        corr_df = df[numeric_cols].corr()
        fig = px.imshow(corr_df, text_auto=True, title='Correlation Matrix')
        visualizations.append({
            'title': 'Correlation Matrix',
            'explanation': 'Automatically generated heatmap showing correlations between numerical variables.',
            'chart_data': fig.to_json()
        })
    
    # If we still need more visualizations, add pie chart for categorical
    if categorical_cols and len(visualizations) < count:
        counts = df[categorical_cols[0]].value_counts()
        fig = px.pie(names=counts.index, values=counts.values, title=f'Composition of {categorical_cols[0]}')
        visualizations.append({
            'title': f'Composition of {categorical_cols[0]}',
            'explanation': 'Automatically generated pie chart showing the distribution of categories.',
            'chart_data': fig.to_json()
        })
    
    return visualizations

if __name__ == '__main__':
    app.run(debug=True)
