import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import json
import plotly.express as px
import plotly.graph_objects as go

class Visualizer:
    def __init__(self):
        self.chart_types = {
            'bar': self.create_bar_chart,
            'line': self.create_line_chart,
            'scatter': self.create_scatter_plot,
            'histogram': self.create_histogram,
            'box': self.create_box_plot,
            'pie': self.create_pie_chart,
            'heatmap': self.create_heatmap,
            'correlation': self.create_correlation_matrix
        }
    
    def generate_chart(self, df, chart_type, x_column=None, y_column=None, **kwargs):
        """Generate a chart based on the specified type and columns"""
        if chart_type not in self.chart_types:
            return {'error': f'Chart type {chart_type} not supported'}
        
        if chart_type == 'correlation':
            return self.chart_types[chart_type](df)
        
        if x_column is None:
            return {'error': 'X column must be specified'}
        
        # For charts that need both x and y columns
        if chart_type in ['scatter', 'line'] and y_column is None:
            return {'error': f'Y column must be specified for {chart_type} chart'}
        
        return self.chart_types[chart_type](df, x_column, y_column, **kwargs)
    
    def create_bar_chart(self, df, x_column, y_column=None, **kwargs):
        """Create a bar chart"""
        if y_column:
            # If y_column is provided, create a grouped bar chart
            fig = px.bar(df, x=x_column, y=y_column, **kwargs)
        else:
            # If y_column is not provided, create a count plot
            value_counts = df[x_column].value_counts().reset_index()
            value_counts.columns = ['value', 'count']
            fig = px.bar(value_counts, x='value', y='count', **kwargs)
            fig.update_layout(xaxis_title=x_column, yaxis_title='Count')
        
        return {'chart': fig.to_json()}
    
    def create_line_chart(self, df, x_column, y_column, **kwargs):
        """Create a line chart"""
        fig = px.line(df, x=x_column, y=y_column, **kwargs)
        return {'chart': fig.to_json()}
    
    def create_scatter_plot(self, df, x_column, y_column, color=None, **kwargs):
        """Create a scatter plot"""
        fig = px.scatter(df, x=x_column, y=y_column, color=color, **kwargs)
        return {'chart': fig.to_json()}
    
    def create_histogram(self, df, x_column, y_column=None, **kwargs):
        """Create a histogram"""
        fig = px.histogram(df, x=x_column, y=y_column, **kwargs)
        return {'chart': fig.to_json()}
    
    def create_box_plot(self, df, x_column, y_column=None, **kwargs):
        """Create a box plot"""
        if y_column:
            fig = px.box(df, x=x_column, y=y_column, **kwargs)
        else:
            fig = px.box(df, y=x_column, **kwargs)
        
        return {'chart': fig.to_json()}
    
    def create_pie_chart(self, df, x_column, y_column=None, **kwargs):
        """Create a pie chart"""
        if y_column:
            # Group by x_column and calculate sum of y_column
            grouped_df = df.groupby(x_column)[y_column].sum().reset_index()
            fig = px.pie(grouped_df, names=x_column, values=y_column, **kwargs)
        else:
            # Count occurrences of each category
            value_counts = df[x_column].value_counts().reset_index()
            value_counts.columns = ['value', 'count']
            fig = px.pie(value_counts, names='value', values='count', **kwargs)
        
        return {'chart': fig.to_json()}
    
    def create_heatmap(self, df, x_column, y_column, **kwargs):
        """Create a heatmap"""
        # Pivot the data to create a matrix suitable for a heatmap
        pivot_table = df.pivot_table(index=y_column, columns=x_column, aggfunc='size', fill_value=0)
        
        # Create a heatmap using Plotly
        fig = px.imshow(pivot_table, **kwargs)
        fig.update_layout(xaxis_title=x_column, yaxis_title=y_column)
        
        return {'chart': fig.to_json()}
    
    def create_correlation_matrix(self, df, **kwargs):
        """Create a correlation matrix heatmap"""
        # Get only numerical columns
        numerical_df = df.select_dtypes(include=['int64', 'float64'])
        
        if len(numerical_df.columns) > 1:
            # Calculate correlation matrix
            corr_matrix = numerical_df.corr()
            
            # Create a heatmap
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", **kwargs)
            fig.update_layout(title="Correlation Matrix")
            
            return {'chart': fig.to_json()}
        else:
            return {'error': 'Not enough numerical columns for correlation matrix'}
    
    def create_time_series_plot(self, df, time_column, value_column, **kwargs):
        """Create a time series plot"""
        # Ensure time_column is in datetime format
        if df[time_column].dtype != 'datetime64[ns]':
            df[time_column] = pd.to_datetime(df[time_column])
        
        # Sort by time
        df = df.sort_values(by=time_column)
        
        # Create the time series plot
        fig = px.line(df, x=time_column, y=value_column, **kwargs)
        fig.update_layout(xaxis_title=time_column, yaxis_title=value_column)
        
        return {'chart': fig.to_json()}
    
    def create_pairplot(self, df, columns=None, hue=None, **kwargs):
        """Create a pair plot with Plotly"""
        if columns is None:
            # Get only numerical columns
            columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if len(columns) < 2:
            return {'error': 'Need at least 2 columns for a pair plot'}
        
        # Create a scatter matrix
        fig = px.scatter_matrix(df, dimensions=columns, color=hue, **kwargs)
        
        return {'chart': fig.to_json()}
