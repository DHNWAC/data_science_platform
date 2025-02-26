import pandas as pd
import numpy as np
import json

class EDAGenerator:
    def get_summary_stats(self, df):
        """Generate summary statistics for all columns"""
        # Basic stats for numerical columns
        numerical_stats = df.describe().transpose().reset_index()
        numerical_stats.columns = ['column', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        numerical_stats = numerical_stats.to_dict('records')
        
        return numerical_stats
    
    def get_missing_values(self, df):
        """Calculate missing values for each column"""
        missing = df.isnull().sum().reset_index()
        missing.columns = ['column', 'missing_count']
        missing['missing_percentage'] = (missing['missing_count'] / len(df) * 100).round(2)
        
        return missing.to_dict('records')
    
    def get_data_types(self, df):
        """Get data types for each column"""
        data_types = df.dtypes.reset_index()
        data_types.columns = ['column', 'data_type']
        data_types['data_type'] = data_types['data_type'].astype(str)
        
        return data_types.to_dict('records')
    
    def get_correlations(self, df):
        """Calculate correlations between numerical columns"""
        # Get only numerical columns
        numerical_df = df.select_dtypes(include=['int64', 'float64'])
        
        if len(numerical_df.columns) > 1:
            corr_matrix = numerical_df.corr().reset_index()
            corr_matrix.columns = ['column'] + list(corr_matrix.columns[1:])
            
            # Transform correlation matrix into a list of correlations
            correlations = []
            for i, row in corr_matrix.iterrows():
                col_name = row['column']
                for col in corr_matrix.columns[1:]:
                    if col_name != col:
                        correlations.append({
                            'column1': col_name,
                            'column2': col,
                            'correlation': row[col]
                        })
            
            # Sort by absolute correlation in descending order
            correlations = sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)
            
            return correlations
        else:
            return []
    
    def get_unique_values(self, df, column):
        """Get unique values and their counts for a column"""
        value_counts = df[column].value_counts().reset_index()
        value_counts.columns = ['value', 'count']
        value_counts['percentage'] = (value_counts['count'] / len(df) * 100).round(2)
        
        return value_counts.to_dict('records')
    
    def get_outliers(self, df, column):
        """Detect outliers using IQR method"""
        if df[column].dtype in ['int64', 'float64']:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
            
            return {
                'column': column,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(df) * 100).round(2)
            }
        else:
            return {
                'column': column,
                'error': 'Column is not numerical'
            }
    
    def generate_full_eda(self, df):
        """Generate a complete EDA report"""
        eda_report = {
            'summary_stats': self.get_summary_stats(df),
            'missing_values': self.get_missing_values(df),
            'data_types': self.get_data_types(df),
            'correlations': self.get_correlations(df),
            'column_details': []
        }
        
        # Generate details for each column
        for column in df.columns:
            column_info = {
                'column': column,
                'data_type': str(df[column].dtype),
                'missing_count': df[column].isnull().sum(),
                'missing_percentage': (df[column].isnull().sum() / len(df) * 100).round(2)
            }
            
            # Add unique values for categorical columns
            if df[column].dtype == 'object' or df[column].nunique() < 10:
                column_info['unique_values'] = self.get_unique_values(df, column)
            
            # Add outlier info for numerical columns
            if df[column].dtype in ['int64', 'float64']:
                column_info['outliers'] = self.get_outliers(df, column)
            
            eda_report['column_details'].append(column_info)
        
        return eda_report
