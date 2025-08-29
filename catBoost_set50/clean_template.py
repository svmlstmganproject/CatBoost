# Fix encoding issues for Windows systems
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error, r2_score, precision_score, recall_score, f1_score
import catboost as cb
from catboost import CatBoostRegressor, CatBoostClassifier
import os

# Set scientific plotting style - black and white theme
plt.style.use('default')  # Reset to default style
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.edgecolor'] = 'white'

# Set color scheme to black, white, and grays
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['black', '#404040', '#808080', '#A0A0A0'])
plt.rcParams['lines.color'] = 'black'
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.titlecolor'] = 'black'
plt.rcParams['legend.edgecolor'] = 'black'
plt.rcParams['legend.facecolor'] = 'white'
plt.rcParams['legend.framealpha'] = 0.8

# Set line styles and markers for scientific appearance
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 6
plt.rcParams['grid.color'] = '#CCCCCC'
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.linewidth'] = 0.5

# Additional scientific plotting settings
plt.rcParams['font.family'] = 'DejaVu Sans'  # Clean, readable font
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

# Set scientific notation for large numbers
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['axes.formatter.limits'] = (-3, 3)

def run_analysis(window_size, stock_name):
    """Run the CatBoost analysis with specified window size"""
    print(f"Starting analysis for {stock_name} with window size: {window_size}")
    
    for kk in range(2):
        # Set random seed for reproducibility
        np.random.seed(42)

        # 1. Load the Excel file
        try:
            # Determine the correct Excel file name based on stock
            excel_file = f"{stock_name.lower()}_final2.xlsx"
            df = pd.read_excel(excel_file)
            print(f"Successfully loaded data: {len(df)} rows")
        except Exception as e:
            print(f"Error loading file: {e}")
            raise

        # Check if 'closed' column exists
        if 'closed' not in df.columns:
            print("Column 'closed' not found. Available columns:", df.columns.tolist())
            raise ValueError("Could not find 'closed' column in the dataset.")
        else:
            print(f"Found 'closed' column with {df['closed'].nunique()} unique values")

        # Check data type and handle categorical data if needed
        print(f"Data type of 'closed' column: {df['closed'].dtype}")
        print(f"Sample data: {df['closed'].head()}")

        # Handle missing values
        missing_count = df['closed'].isna().sum()
        if missing_count > 0:
            print(f"Removing {missing_count} rows with missing closed values")
            df = df.dropna(subset=['closed'])

        # If closed is categorical, encode it
        encoder = None
        if df['closed'].dtype == 'object':
            print("Encoding categorical closed values")
            encoder = LabelEncoder()
            df['closed_encoded'] = encoder.fit_transform(df['closed'])
            print(f"Encoded {len(encoder.classes_)} unique closed values")
            target_col = 'closed_encoded'
            # Display mapping for reference
            mapping = {label: idx for idx, label in enumerate(encoder.classes_)}
            print("Encoding mapping:", mapping)
            # Get number of classes for classification
            num_classes = len(encoder.classes_)
            is_classification = True
        else:
            target_col = 'closed'
            # If numerical, determine range for normalization
            num_classes = 1  # Treating as continuous value
            is_classification = False

        # 2. Prepare data with sliding window
        print(f"Using window size: {window_size}")
        X_windows, y_windows = [], []
        indices_map = []  # To keep track of original indices

        for i in range(window_size, len(df)):
            X_windows.append(df[target_col].iloc[i-window_size:i].values)
            y_windows.append(df[target_col].iloc[i])
            indices_map.append(i)  # Store the original index

        X_windows = np.array(X_windows)
        y_windows = np.array(y_windows)
        indices_map = np.array(indices_map)

        print(f"Created {len(X_windows)} samples with window size {window_size}")

        # Continue with the rest of your existing analysis code...
        # (This is where you would paste the rest of your existing analysis)
        
        print(f"Iteration {kk} completed with window size {window_size}")

    print(f"🎉 All iterations completed successfully for {stock_name} with window size {window_size}!")

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CatBoost analysis for SET50 stock')
    parser.add_argument('--window_size', type=int, default=30, 
                       help='Window size for sliding window (default: 30)')
    parser.add_argument('--stock_name', type=str, default='ADVANC',
                       help='Stock name for analysis (default: ADVANC)')
    args = parser.parse_args()
    
    # Run the analysis with specified window size
    run_analysis(args.window_size, args.stock_name)


