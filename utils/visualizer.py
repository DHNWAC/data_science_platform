import pandas as pd
import numpy as np
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
            'correlation': self.create_correlation_matrix
        }
        # Removed 'heatmap'
    
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
        """Create a bar chart with improved aesthetics"""
        # Make sure we're not passing duplicate color parameters
        user_color = kwargs.pop('color', None)
        
        if y_column:
            # If both columns are numeric, use averages
            if pd.api.types.is_numeric_dtype(df[x_column]) and pd.api.types.is_numeric_dtype(df[y_column]):
                # Bin the x values for better visualization if there are many unique values
                if df[x_column].nunique() > 20:
                    df['x_binned'] = pd.cut(df[x_column], bins=10)
                    grouped = df.groupby('x_binned')[y_column].agg(['mean', 'count']).reset_index()
                    grouped['x_binned'] = grouped['x_binned'].astype(str)
                    fig = px.bar(
                        grouped, 
                        x='x_binned', 
                        y='mean',
                        # Use a more vibrant color scale
                        color_continuous_scale='Viridis',
                        labels={'mean': f'Average {y_column}', 'x_binned': x_column, 'count': 'Count'},
                        template="plotly_white",
                        **kwargs
                    )
                    fig.update_layout(coloraxis_colorbar=dict(title="Count"))
                else:
                    # Group by x_column and calculate mean of y_column
                    grouped = df.groupby(x_column)[y_column].agg(['mean', 'count']).reset_index()
                    fig = px.bar(
                        grouped, 
                        x=x_column, 
                        y='mean',
                        # Use a more vibrant color scale
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        labels={'mean': f'Average {y_column}', 'count': 'Count'},
                        template="plotly_white",
                        **kwargs
                    )
            # If x is categorical and y is numeric, show means per category
            elif pd.api.types.is_numeric_dtype(df[y_column]):
                # Group by x_column and calculate statistics of y_column
                grouped = df.groupby(x_column)[y_column].agg(['mean', 'count']).reset_index()
                fig = px.bar(
                    grouped, 
                    x=x_column, 
                    y='mean',
                    # Use a more vibrant color scale
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    labels={'mean': f'Average {y_column}', 'count': 'Count'},
                    template="plotly_white",
                    **kwargs
                )
            else:
                # For two categorical variables, create a count-based bar chart
                counts = df.groupby([x_column, y_column]).size().reset_index(name='count')
                fig = px.bar(
                    counts, 
                    x=x_column, 
                    y='count',
                    color=y_column,
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    barmode='group',
                    template="plotly_white",
                    **kwargs
                )
        else:
            # If y_column is not provided, create a count plot
            value_counts = df[x_column].value_counts().reset_index()
            value_counts.columns = ['value', 'count']
            
            # Sort by count for better visualization
            value_counts = value_counts.sort_values('count', ascending=False)
            
            # If too many categories, limit to top N
            if len(value_counts) > 15:
                top_n = value_counts.head(15)
                other_count = value_counts[15:]['count'].sum()
                if other_count > 0:
                    other_row = pd.DataFrame({'value': ['Other'], 'count': [other_count]})
                    value_counts = pd.concat([top_n, other_row])
                else:
                    value_counts = top_n
            
            # Use a specific color for single-variable bar charts
            fig = px.bar(
                value_counts, 
                x='value', 
                y='count',
                color_discrete_sequence=['#3366CC', '#DC3912', '#FF9900', '#109618', '#990099'],
                labels={'value': x_column, 'count': 'Count'},
                template="plotly_white",
                **kwargs
            )
        
        # Enhance overall appearance
        fig.update_layout(
            title=f"{y_column + ' by ' if y_column else 'Distribution of '}{x_column}",
            xaxis_title=x_column,
            yaxis_title=y_column if y_column else 'Count',
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='white',
            font=dict(size=12),
            margin=dict(l=40, r=40, t=50, b=40),
        )
        
        # Make the bars more vibrant
        fig.update_traces(marker_line_width=1, marker_line_color="#333333", opacity=0.8)
        
        return {'chart': fig.to_json()}
    
    def create_line_chart(self, df, x_column, y_column, **kwargs):
        """Create a line chart with improved aesthetics"""
        # Make sure we're not passing duplicate color parameters
        user_color = kwargs.pop('color', None)
        
        # Check if x_column is datetime
        if pd.api.types.is_datetime64_any_dtype(df[x_column]) or 'date' in x_column.lower() or 'time' in x_column.lower():
            # Try to convert to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df[x_column]):
                try:
                    df = df.copy()
                    df[x_column] = pd.to_datetime(df[x_column])
                except:
                    pass  # If conversion fails, proceed with original column
            
            # Sort by date/time for proper line chart
            df = df.sort_values(by=x_column)
            
        # If we have categorical X and numeric Y, we might want to group and aggregate
        if pd.api.types.is_categorical_dtype(df[x_column]) or df[x_column].dtype == 'object':
            if pd.api.types.is_numeric_dtype(df[y_column]):
                # Group by x_column and calculate mean of y_column
                grouped = df.groupby(x_column)[y_column].mean().reset_index()
                fig = px.line(
                    grouped, 
                    x=x_column, 
                    y=y_column,
                    markers=True,
                    line_shape='linear',
                    color_discrete_sequence=['#2C3E50', '#E74C3C', '#3498DB'],
                    template="plotly_white",
                    **kwargs
                )
            else:
                # For two categorical variables, create a count-based line chart
                counts = df.groupby(x_column)[y_column].count().reset_index(name='count')
                fig = px.line(
                    counts, 
                    x=x_column, 
                    y='count',
                    markers=True,
                    line_shape='linear',
                    color_discrete_sequence=['#2C3E50', '#E74C3C', '#3498DB'],
                    template="plotly_white",
                    **kwargs
                )
                fig.update_layout(yaxis_title='Count')
        else:
            # For numeric x and y, create a standard line chart
            fig = px.line(
                df, 
                x=x_column, 
                y=y_column,
                markers=True,
                line_shape='linear',
                color_discrete_sequence=['#2C3E50', '#E74C3C', '#3498DB'],
                template="plotly_white",
                **kwargs
            )
        
        # Add trendline if appropriate
        if pd.api.types.is_numeric_dtype(df[x_column]) and pd.api.types.is_numeric_dtype(df[y_column]):
            # Add a trendline
            fig.add_trace(
                px.scatter(
                    df, 
                    x=x_column, 
                    y=y_column, 
                    trendline="ols"
                ).data[1]
            )
        
        # Enhance overall appearance
        fig.update_layout(
            title=f"{y_column} by {x_column}",
            xaxis_title=x_column,
            yaxis_title=y_column,
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='white',
            font=dict(size=12),
            margin=dict(l=40, r=40, t=50, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add hover template
        fig.update_traces(
            hovertemplate=f"{x_column}: %{{x}}<br>{y_column}: %{{y}}<extra></extra>"
        )
        
        return {'chart': fig.to_json()}
    
    def create_scatter_plot(self, df, x_column, y_column, **kwargs):
        """Create a scatter plot with improved aesthetics"""
        # Handle color parameter - first check if 'color' is in kwargs
        color_col = kwargs.pop('color', None)
        
        if pd.api.types.is_numeric_dtype(df[x_column]) and pd.api.types.is_numeric_dtype(df[y_column]):
            # For numerical x and y
            if color_col:
                # If color is categorical
                if not pd.api.types.is_numeric_dtype(df[color_col]):
                    fig = px.scatter(
                        df, 
                        x=x_column, 
                        y=y_column, 
                        color=color_col,
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        opacity=0.8,
                        template="plotly_white",
                        **kwargs
                    )
                else:
                    # If color is numerical
                    fig = px.scatter(
                        df, 
                        x=x_column, 
                        y=y_column, 
                        color=color_col,
                        color_continuous_scale='Viridis',
                        opacity=0.8,
                        template="plotly_white",
                        **kwargs
                    )
            else:
                # No color variable
                fig = px.scatter(
                    df, 
                    x=x_column, 
                    y=y_column,
                    color_discrete_sequence=['#3366CC'],
                    opacity=0.8,
                    template="plotly_white",
                    **kwargs
                )
                
            # Add a trendline
            fig.add_trace(
                px.scatter(
                    df, 
                    x=x_column, 
                    y=y_column, 
                    trendline="ols"
                ).data[1]
            )
            
            # Add marginal distributions
            marginal_fig = px.scatter(
                df, 
                x=x_column, 
                y=y_column,
                color=color_col if color_col else None,
                marginal_x="histogram", 
                marginal_y="histogram",
                opacity=0.8,
                template="plotly_white",
                **kwargs
            )
            
            # Use the updated figure with marginals
            fig = marginal_fig
        else:
            # For non-numerical values, create a jittered scatter plot
            fig = px.strip(
                df, 
                x=x_column, 
                y=y_column,
                color=color_col if color_col else None,
                opacity=0.8,
                template="plotly_white",
                **kwargs
            )
        
        # Enhance overall appearance
        fig.update_layout(
            title=f"Scatter Plot: {y_column} vs {x_column}",
            xaxis_title=x_column,
            yaxis_title=y_column,
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='white',
            font=dict(size=12),
            margin=dict(l=40, r=40, t=50, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return {'chart': fig.to_json()}
    
    def create_histogram(self, df, x_column, y_column=None, **kwargs):
        """Create a histogram with improved aesthetics"""
        # Handle color parameter - first check if 'color' is in kwargs
        color_col = kwargs.pop('color', None)
        
        # For numerical columns
        if pd.api.types.is_numeric_dtype(df[x_column]):
            if y_column and pd.api.types.is_numeric_dtype(df[y_column]):
                # Colored histogram by a second variable if provided
                fig = px.histogram(
                    df, 
                    x=x_column, 
                    color=y_column,  # Use y_column for coloring
                    marginal="box",
                    opacity=0.8,
                    barmode="overlay",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    template="plotly_white",
                    **kwargs
                )
            else:
                # Basic histogram with KDE curve
                fig = px.histogram(
                    df, 
                    x=x_column,
                    histnorm='probability density',
                    marginal="box",
                    opacity=0.8,
                    color_discrete_sequence=['#3366CC'],
                    template="plotly_white",
                    **kwargs
                )
                
                # Add KDE curve
                try:
                    from scipy.stats import gaussian_kde
                    data = df[x_column].dropna()
                    kde = gaussian_kde(data)
                    x_range = np.linspace(data.min(), data.max(), 1000)
                    y_kde = kde(x_range)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=y_kde,
                            mode='lines',
                            name='KDE',
                            line=dict(color='red', width=2)
                        )
                    )
                except:
                    # If KDE fails, just show histogram
                    pass
        else:
            # For categorical data, create a count-based histogram
            value_counts = df[x_column].value_counts().reset_index()
            value_counts.columns = ['value', 'count']
            
            # Sort by count for better visualization
            value_counts = value_counts.sort_values('count', ascending=False)
            
            fig = px.bar(
                value_counts, 
                x='value', 
                y='count',
                color_discrete_sequence=['#3366CC', '#DC3912', '#FF9900', '#109618'],
                template="plotly_white",
                labels={"value": x_column, "count": "Count"},
                **kwargs
            )
            fig.update_layout(
                xaxis_title=x_column,
                yaxis_title='Count'
            )
        
        # Enhance overall appearance
        fig.update_layout(
            title=f"Distribution of {x_column}",
            xaxis_title=x_column,
            yaxis_title="Density" if pd.api.types.is_numeric_dtype(df[x_column]) else "Count",
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='white',
            font=dict(size=12),
            margin=dict(l=40, r=40, t=50, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Make the bars more vibrant
        if pd.api.types.is_numeric_dtype(df[x_column]):
            fig.update_traces(marker_line_width=1, marker_line_color="#333333", selector=dict(type="histogram"))
        
        return {'chart': fig.to_json()}
    
    def create_box_plot(self, df, x_column, y_column=None, **kwargs):
        """Create a box plot with improved aesthetics"""
        # Handle color parameter - first check if 'color' is in kwargs
        color_col = kwargs.pop('color', None)
        
        if y_column:
            # Box plot with categories on x-axis
            if pd.api.types.is_numeric_dtype(df[y_column]):
                fig = px.box(
                    df, 
                    x=x_column, 
                    y=y_column,
                    notched=True,
                    points="outliers",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    template="plotly_white",
                    **kwargs
                )
            else:
                # Both columns are categorical, create a count-based box plot
                counts = df.groupby([x_column, y_column]).size().reset_index(name='count')
                fig = px.box(
                    counts, 
                    x=x_column, 
                    y='count',
                    color=y_column,
                    notched=True,
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    template="plotly_white",
                    **kwargs
                )
                fig.update_layout(yaxis_title='Count')
        else:
            # Single column box plot
            if pd.api.types.is_numeric_dtype(df[x_column]):
                fig = px.box(
                    df, 
                    y=x_column,
                    notched=True,
                    points="outliers",
                    color_discrete_sequence=['#3366CC'],
                    template="plotly_white",
                    **kwargs
                )
            else:
                # For categorical column, create a count box plot
                return {'error': 'Box plot requires at least one numerical column'}
        
        # Enhance overall appearance
        if y_column:
            title = f"Box Plot: {y_column} by {x_column}"
        else:
            title = f"Box Plot: {x_column}"
            
        fig.update_layout(
            title=title,
            xaxis_title=x_column if y_column else "",
            yaxis_title=y_column if y_column else x_column,
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='white',
            font=dict(size=12),
            margin=dict(l=40, r=40, t=50, b=40)
        )
        
        return {'chart': fig.to_json()}
    
    def create_pie_chart(self, df, x_column, y_column=None, **kwargs):
        """Create a pie chart with improved aesthetics"""
        # Handle color parameter
        color_col = kwargs.pop('color', None)
        
        if y_column:
            # Group by x_column and calculate sum of y_column
            if pd.api.types.is_numeric_dtype(df[y_column]):
                grouped = df.groupby(x_column)[y_column].sum().reset_index()
                
                # If too many categories, limit to top N and group others
                if len(grouped) > 10:
                    grouped = grouped.sort_values(y_column, ascending=False)
                    top_n = grouped.head(9)
                    other_sum = grouped.iloc[9:][y_column].sum()
                    other_row = pd.DataFrame({x_column: ['Other'], y_column: [other_sum]})
                    grouped = pd.concat([top_n, other_row])
                
                fig = px.pie(
                    grouped, 
                    names=x_column, 
                    values=y_column,
                    hole=0.4,
                    template="plotly_white",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    **kwargs
                )
                
                # Add total in the center
                total = grouped[y_column].sum()
                fig.add_annotation(
                    text=f"Total<br>{total:,.0f}",
                    x=0.5, y=0.5,
                    font_size=14,
                    showarrow=False
                )
            else:
                # For non-numeric y_column, use counts
                counts = df.groupby([x_column, y_column]).size().reset_index(name='count')
                
                # Create sunburst chart instead for hierarchical data
                fig = px.sunburst(
                    counts, 
                    path=[x_column, y_column], 
                    values='count',
                    template="plotly_white",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    **kwargs
                )
        else:
            # Count occurrences of each category
            value_counts = df[x_column].value_counts().reset_index()
            value_counts.columns = ['value', 'count']
            
            # If too many categories, limit to top N and group others
            if len(value_counts) > 10:
                top_n = value_counts.head(9)
                other_count = value_counts.iloc[9:]['count'].sum()
                other_row = pd.DataFrame({'value': ['Other'], 'count': [other_count]})
                value_counts = pd.concat([top_n, other_row])
            
            fig = px.pie(
                value_counts, 
                names='value', 
                values='count',
                hole=0.4,
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Bold,
                **kwargs
            )
            
            # Add total in the center
            total = value_counts['count'].sum()
            fig.add_annotation(
                text=f"Total<br>{total:,.0f}",
                x=0.5, y=0.5,
                font_size=14,
                showarrow=False
            )
        
        # Enhance overall appearance
        if y_column:
            title = f"Pie Chart: {y_column} by {x_column}"
        else:
            title = f"Distribution of {x_column}"
            
        fig.update_layout(
            title=title,
            font=dict(size=12),
            margin=dict(l=40, r=40, t=50, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        
        return {'chart': fig.to_json()}
    
    def create_correlation_matrix(self, df, **kwargs):
        """Create a correlation matrix heatmap with improved aesthetics"""
        # Get only numerical columns
        numerical_df = df.select_dtypes(include=['int64', 'float64'])
        
        if len(numerical_df.columns) > 1:
            # Calculate correlation matrix
            corr_matrix = numerical_df.corr()
            
            # Create a heatmap with improved aesthetics
            fig = px.imshow(
                corr_matrix, 
                text_auto=True, 
                aspect="auto",
                color_continuous_scale='RdBu_r',  # Blue-Red diverging colorscale
                zmin=-1, zmax=1,  # Set fixed scale for correlation
                template="plotly_white",
                **kwargs
            )
            
            # Improve layout - ensure axis labels are visible
            fig.update_layout(
                title="Correlation Matrix",
                width=800,
                height=700,
                xaxis=dict(
                    title="",
                    tickangle=-45,
                    tickfont=dict(size=11),
                    tickmode='array',
                    tickvals=list(range(len(corr_matrix.columns))),
                    ticktext=corr_matrix.columns,
                    side='bottom',
                    showgrid=True
                ),
                yaxis=dict(
                    title="",
                    tickfont=dict(size=11),
                    tickmode='array',
                    tickvals=list(range(len(corr_matrix.index))),
                    ticktext=corr_matrix.index,
                    showgrid=True
                ),
                coloraxis_colorbar=dict(
                    title="Correlation",
                    thicknessmode="pixels", thickness=20,
                    lenmode="pixels", len=500,
                    ticks="outside"
                ),
                plot_bgcolor='#f8f9fa',
                paper_bgcolor='white',
                font=dict(size=12),
                margin=dict(l=60, r=40, t=50, b=100)
            )
            
            # Improve hover details and text formatting
            fig.update_traces(
                text=corr_matrix.round(2),
                texttemplate="%{text:.2f}",
                hovertemplate="Row: %{y}<br>Column: %{x}<br>Correlation: %{z:.3f}<extra></extra>"
            )
            
            return {'chart': fig.to_json()}
        else:
            return {'error': 'Not enough numerical columns for correlation matrix. Need at least 2 numerical columns.'}
    
    def create_time_series_plot(self, df, time_column, value_column, **kwargs):
        """Create a time series plot"""
        # Make sure we're not passing duplicate color parameters
        user_color = kwargs.pop('color', None)
        
        # Ensure time_column is in datetime format
        if df[time_column].dtype != 'datetime64[ns]':
            try:
                df = df.copy()
                df[time_column] = pd.to_datetime(df[time_column])
            except:
                return {'error': 'Unable to convert time column to datetime format'}
        
        # Sort by time
        df = df.sort_values(by=time_column)
        
        # Create the time series plot
        fig = px.line(
            df, 
            x=time_column, 
            y=value_column, 
            markers=True,
            color_discrete_sequence=['#3366CC', '#DC3912', '#FF9900'],
            template="plotly_white",
            **kwargs
        )
        
        # Enhance overall appearance
        fig.update_layout(
            title=f"Time Series: {value_column} over Time",
            xaxis_title=time_column,
            yaxis_title=value_column,
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='white',
            font=dict(size=12),
            margin=dict(l=40, r=40, t=50, b=40)
        )
        
        return {'chart': fig.to_json()}