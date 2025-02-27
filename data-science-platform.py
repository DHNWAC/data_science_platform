# Project Structure
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
