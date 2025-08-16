name="dax"
for kk in range(100):
        
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
    import xgboost as xgb
    import os

    # Set random seed for reproducibility
    np.random.seed(42)

    # 1. Load the Excel file
    try:
        df = pd.read_excel("input_dax_1day.xlsx")
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
    window_size = 100
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

    # 3. Normalize data
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_windows_flat = X_windows.reshape(X_windows.shape[0], -1)
    X_windows_scaled = scaler_X.fit_transform(X_windows_flat)  # Keep as 2D for XGBoost

    # Store original indices for later visualization
    all_indices = np.arange(len(X_windows_scaled))

    # 4. Split data into training and testing sets SEQUENTIALLY
    # Using first 50 samples for training, next 10 for testing
    train_size = 1500+kk  # First 50 samples
    test_size = 100   # Next 10 samples

    # Make sure we have enough data
    total_needed = train_size + test_size
    if len(X_windows_scaled) < total_needed:
        print(f"Warning: Not enough data. Need {total_needed} samples but only have {len(X_windows_scaled)}.")
        train_size = int(len(X_windows_scaled) * 0.8)  # Fallback to 80% train
        test_size = len(X_windows_scaled) - train_size

    # Sequential split
    train_indices = all_indices[:train_size]
    test_indices = all_indices[train_size:train_size+test_size]

    # Extract train and test data
    X_train = X_windows_scaled[train_indices]
    X_test = X_windows_scaled[test_indices]
    y_train = y_windows[train_indices]  # Using original values directly with XGBoost
    y_test = y_windows[test_indices]

    # Get original dataframe indices for test data
    original_test_indices = indices_map[test_indices]

    print(f"Sequential Training set: {len(X_train)} samples (indices 0-{train_size-1})")
    print(f"Sequential Testing set: {len(X_test)} samples (indices {train_size}-{train_size+test_size-1})")

    # 5. Create and configure XGBoost model
    # Configure parameters based on task type
    if is_classification:
        # For classification tasks:
        print(f"Training XGBoost for classification with {num_classes} classes")
        params = {
            'objective': 'multi:softprob' if num_classes > 2 else 'binary:logistic',
            'num_class': num_classes if num_classes > 2 else None,
            'max_depth': 6,
            'eta': 0.01,  # Learning rate
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'eval_metric': 'mlogloss' if num_classes > 2 else 'logloss',
            'seed': 42
        }
        # Clean up params if binary classification
        if num_classes <= 2:
            params.pop('num_class')
    else:
        # For regression tasks:
        print("Training XGBoost for regression")
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'eta': 0.01,  # Learning rate
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'eval_metric': 'rmse',
            'seed': 42
        }

    # 6. Prepare DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Create a watchlist to monitor training
    watchlist = [(dtrain, 'train'), (dtest, 'test')]

    # 7. Train XGBoost model
    num_rounds = 10000  # Can be adjusted
    early_stopping_rounds = 50

    print("Training XGBoost model...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=watchlist,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=100  # Print evaluation every 100 rounds
    )

    print(f"XGBoost training completed with best iteration: {model.best_iteration}")

    # 8. Generate predictions
    train_preds = model.predict(dtrain)
    test_preds = model.predict(dtest)

    # Process predictions based on task type
    if is_classification:
        if num_classes > 2:
            # For multi-class, each prediction is a probability distribution across classes
            # Convert to class index
            train_preds_class = np.argmax(train_preds.reshape(len(train_preds), num_classes), axis=1)
            test_preds_class = np.argmax(test_preds.reshape(len(test_preds), num_classes), axis=1)
        else:
            # For binary classification
            train_preds_class = np.round(train_preds).astype(int)
            test_preds_class = np.round(test_preds).astype(int)
        
        # Store the class predictions
        train_preds_final = train_preds_class
        test_preds_final = test_preds_class
    else:
        # For regression, predictions are already the final values
        train_preds_final = train_preds
        test_preds_final = test_preds

    # 9. Evaluate model performance
    print("\nXGBoost Performance Metrics Summary:")

    if not is_classification:
        # For regression task
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds_final))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds_final))
        
        print(f"In-Sample RMSE: {train_rmse:.4f}")
        print(f"Out-of-Sample RMSE: {test_rmse:.4f}")
        
        # Calculate errors
        test_errors = y_test - test_preds_final
        
        # Create DataFrame for Excel export
        export_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': test_preds_final,
            'Error': test_errors
        })
        
        # Add RMSE as a summary row (will appear at the bottom of the Excel file)
        summary_df = pd.DataFrame({
            'Actual': ['RMSE'],
            'Predicted': [test_rmse],
            'Error': ['']
        })
        
    else:
        # For classification task
        train_accuracy = np.mean(train_preds_final == y_train)
        test_accuracy = np.mean(test_preds_final == y_test)
        
        print(f"In-Sample Accuracy: {train_accuracy:.4f}")
        print(f"Out-of-Sample Accuracy: {test_accuracy:.4f}")
        
        # Calculate errors (for classification, this is just whether prediction was correct)
        test_errors = y_test != test_preds_final
        
        # Convert predictions to original closed labels if applicable
        if encoder is not None:
            test_actual_labels = [encoder.classes_[i] for i in y_test]
            test_pred_labels = [encoder.classes_[i] for i in test_preds_final]
            
            # Create DataFrame for Excel export
            export_df = pd.DataFrame({
                'Actual_Encoded': y_test,
                'Predicted_Encoded': test_preds_final,
                'Actual_Original': test_actual_labels,
                'Predicted_Original': test_pred_labels,
                'Error': test_errors.astype(int)  # 0 for correct, 1 for incorrect
            })
            
            # Add accuracy as a summary row
            summary_df = pd.DataFrame({
                'Actual_Encoded': ['Accuracy'],
                'Predicted_Encoded': [test_accuracy],
                'Actual_Original': [''],
                'Predicted_Original': [''],
                'Error': ['']
            })
        else:
            # Create DataFrame for Excel export
            export_df = pd.DataFrame({
                'Actual': y_test,
                'Predicted': test_preds_final,
                'Error': test_errors.astype(int)  # 0 for correct, 1 for incorrect
            })
            
            # Add accuracy as a summary row
            summary_df = pd.DataFrame({
                'Actual': ['Accuracy'],
                'Predicted': [test_accuracy],
                'Error': ['']
            })
    
    # Create a filename with the iteration number
    excel_filename = f"predicted_xgboost_iter{kk}.xlsx"
    
    # Combine data and summary dataframes
    final_df = pd.concat([export_df, summary_df], ignore_index=True)
    
    # Export to Excel
    final_df.to_excel(excel_filename, index=True, sheet_name='Predictions')
    print(f"Saved prediction results to {excel_filename}")
    
    # Also create a combined file for all iterations
    # Create or append to a master results file
    if kk == 0:
        # Create a new file for the first iteration
        with pd.ExcelWriter("predicted_xgboost.xlsx") as writer:
            export_df.to_excel(writer, sheet_name=f'Iter{kk}_Data', index=True)
            if not is_classification:
                pd.DataFrame({'RMSE': [test_rmse]}).to_excel(writer, sheet_name=f'Iter{kk}_Summary', index=False)
            else:
                pd.DataFrame({'Accuracy': [test_accuracy]}).to_excel(writer, sheet_name=f'Iter{kk}_Summary', index=False)
    else:
        # Read existing file and append new sheet
        with pd.ExcelWriter("predicted_xgboost.xlsx", mode='a', if_sheet_exists='replace') as writer:
            export_df.to_excel(writer, sheet_name=f'Iter{kk}_Data', index=True)
            if not is_classification:
                pd.DataFrame({'RMSE': [test_rmse]}).to_excel(writer, sheet_name=f'Iter{kk}_Summary', index=False)
            else:
                pd.DataFrame({'Accuracy': [test_accuracy]}).to_excel(writer, sheet_name=f'Iter{kk}_Summary', index=False)
    
    print(f"Updated combined results in predicted_xgboost.xlsx")

    # 10. BACKTESTING IMPLEMENTATION
    print("\n=== Starting Backtest Strategy ===")
    
    # Create a dataframe for backtesting
    if not is_classification:
        # For regression task - we'll use the original data for the test period
        backtest_df = pd.DataFrame({
            'timestamp': [i for i in range(len(test_indices))],  # Placeholder timestamps
            'index': original_test_indices,
            'actual': y_test,
            'predicted': test_preds_final,
            'next_actual': np.concatenate([y_test[1:], [np.nan]])  # Next period's actual price
        })
        
        # Calculate price movements (both actual and predicted)
        backtest_df['actual_change'] = backtest_df['next_actual'] - backtest_df['actual']
        backtest_df['predicted_change'] = backtest_df['predicted'] - backtest_df['actual']
        
        # Trading strategy: 
        # Long when prediction > actual (we expect price to rise)
        # Short when prediction < actual (we expect price to fall)
        # No position when prediction == actual
        backtest_df['position'] = np.sign(backtest_df['predicted_change'])
        
        # Calculate returns (excluding the last row which has NaN for next_actual)
        backtest_df['market_return'] = backtest_df['actual_change'] / backtest_df['actual']
        backtest_df['strategy_return'] = backtest_df['position'] * backtest_df['market_return']
        
        # Calculate cumulative returns
        backtest_df['cum_market_return'] = (1 + backtest_df['market_return'].fillna(0)).cumprod() - 1
        backtest_df['cum_strategy_return'] = (1 + backtest_df['strategy_return'].fillna(0)).cumprod() - 1
        
        # Calculate trading stats
        total_trades = (backtest_df['position'].diff() != 0).sum()
        winning_trades = ((backtest_df['strategy_return'] > 0) & (backtest_df['position'] != 0)).sum()
        losing_trades = ((backtest_df['strategy_return'] < 0) & (backtest_df['position'] != 0)).sum()
        win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0
        
        # Calculate Sharpe ratio (assuming daily data - adjust if different frequency)
        strategy_return_mean = backtest_df['strategy_return'].mean()
        strategy_return_std = backtest_df['strategy_return'].std()
        sharpe_ratio = strategy_return_mean / strategy_return_std * np.sqrt(252) if strategy_return_std > 0 else 0
        
        print(f"Backtest Results:")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Strategy Return: {backtest_df['cum_strategy_return'].iloc[-2]:.2%}")
        print(f"Market Return: {backtest_df['cum_market_return'].iloc[-2]:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        
        # Create a backtest results sheet for Excel
        backtest_summary = pd.DataFrame({
            'Metric': ['Total Trades', 'Win Rate', 'Strategy Return', 'Market Return', 'Sharpe Ratio'],
            'Value': [total_trades, f"{win_rate:.2%}", f"{backtest_df['cum_strategy_return'].iloc[-2]:.2%}", 
                     f"{backtest_df['cum_market_return'].iloc[-2]:.2%}", f"{sharpe_ratio:.2f}"]
        })
        
        # Export backtest data
        with pd.ExcelWriter(f"backtest_results_iter{kk}.xlsx") as writer:
            backtest_df.to_excel(writer, sheet_name='BacktestData', index=False)
            backtest_summary.to_excel(writer, sheet_name='Summary', index=False)
            
        # Update combined results
        try:
            with pd.ExcelWriter("predicted_xgboost.xlsx", mode='a', if_sheet_exists='replace') as writer:
                backtest_df.to_excel(writer, sheet_name=f'Iter{kk}_Backtest', index=False)
                backtest_summary.to_excel(writer, sheet_name=f'Iter{kk}_BacktestSummary', index=False)
        except Exception as e:
            print(f"Error updating combined file with backtest results: {e}")
        
        # Visualize backtest results
        plt.figure(figsize=(14, 8))
        
        # Plot 1: Price and Predictions
        plt.subplot(2, 1, 1)
        plt.title(f'Backtest Results - Price vs Predictions (Iteration {kk})', fontsize=14)
        plt.plot(backtest_df['timestamp'], backtest_df['actual'], 'b-', label='Actual Price', alpha=0.7)
        plt.plot(backtest_df['timestamp'], backtest_df['predicted'], 'r--', label='Predicted Price', alpha=0.7)
        
        # Mark long positions with green up arrows
        long_indices = backtest_df[backtest_df['position'] > 0]['timestamp']
        long_prices = backtest_df[backtest_df['position'] > 0]['actual']
        plt.scatter(long_indices, long_prices, marker='^', color='green', s=100, label='Long', alpha=0.7)
        
        # Mark short positions with red down arrows
        short_indices = backtest_df[backtest_df['position'] < 0]['timestamp']
        short_prices = backtest_df[backtest_df['position'] < 0]['actual']
        plt.scatter(short_indices, short_prices, marker='v', color='red', s=100, label='Short', alpha=0.7)
        
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative Returns
        plt.subplot(2, 1, 2)
        plt.title('Cumulative Returns', fontsize=14)
        plt.plot(backtest_df['timestamp'], backtest_df['cum_market_return'], 'b-', label='Market Return', alpha=0.7)
        plt.plot(backtest_df['timestamp'], backtest_df['cum_strategy_return'], 'g-', label='Strategy Return', alpha=0.7)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Return (%)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add strategy stats as text
        stats_text = f"Win Rate: {win_rate:.2%}\n"
        stats_text += f"Strategy Return: {backtest_df['cum_strategy_return'].iloc[-2]:.2%}\n"
        stats_text += f"Market Return: {backtest_df['cum_market_return'].iloc[-2]:.2%}\n"
        stats_text += f"Sharpe Ratio: {sharpe_ratio:.2f}"
        
        #plt.figtext(0.02, 0.02, stats_text, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"backtest_results_iter{kk}.png", dpi=300)
    
    else:
        # For classification tasks, approach depends on number of classes
        # Here's a simplified version that assumes binary classification (up/down)
        print("Classification-based backtesting not implemented in this version.")
        # For classification backtesting, you would map class predictions to trading signals
        # e.g., Class 1 = Long, Class 0 = Short, etc.

    # 11. Feature importance visualization
    plt.figure(figsize=(12, 6))
    xgb.plot_importance(model, max_num_features=15, height=0.8)
    plt.title('XGBoost Feature Importances', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"xgboost_feature_importance{kk}.png", dpi=300)

    # 12. Create visualizations with in-sample and out-of-sample data
    # Create a combined visualization
    plt.figure(figsize=(15, 10))

    # Plot training and testing data
    plt.subplot(2, 1, 1)
    plt.title('DAX closed price prediction with XGBoost: In-Sample vs Out-of-Sample Comparison', fontsize=16)

    # Plot in-sample data (first 50 samples)
    plt.plot(train_indices, y_train, 'bo', label='In-Sample Actual', alpha=0.1)
    plt.plot(train_indices, train_preds_final, 'ro', label='In-Sample Predicted', alpha=0.1)

    # Plot out-of-sample data (next 10 samples)
    plt.plot(test_indices, y_test, 'go', label='Out-of-Sample Actual', markersize=1)
    plt.plot(test_indices, test_preds_final, 'mo', label='Out-of-Sample Predicted', markersize=1)

    plt.xlabel('Sample Index', fontsize=14)
    plt.ylabel('closed Value', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add mapping legend if categorical
    if encoder is not None and len(encoder.classes_) <= 10:
        legend_text = "\n".join([f"{i}: {val}" for i, val in enumerate(encoder.classes_)])
        plt.figtext(0.02, 0.02, f"closed Encoding:\n{legend_text}", 
                    bbox=dict(facecolor='white', alpha=0.8))

    # Plot the time series view
    plt.subplot(2, 1, 2)
    plt.title('DAX closed price prediction with XGBoost: ', fontsize=16)

    # Continuous plot of all available data to show sequence
    x_train = np.arange(len(y_train))
    x_test = np.arange(len(y_train), len(y_train) + len(y_test))

    # Plot actual values
    plt.plot(x_train, y_train, 'b-', label='In-Sample Actual', alpha=0.5)
    plt.plot(x_test, y_test, 'g-', label='Out-of-Sample Actual', alpha=0.5)

    # Plot predicted values
    plt.plot(x_train, train_preds_final, 'r-', label='In-Sample Predicted', alpha=1)
    plt.plot(x_test, test_preds_final, 'm-', label='Out-of-Sample Predicted', alpha=1)

    # Highlight the boundary between training and testing
    plt.axvline(x=len(y_train), color='black', linestyle='--', 
                label='Train/Test Boundary')

    plt.xlabel('Sequential Index', fontsize=14)
    plt.ylabel('closed Value', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"dax_closed_price_xgboost_comparison{kk}.png", dpi=300)

    # 13. Create a detailed comparison of in-sample and out-of-sample predictions
    plt.figure(figsize=(15, 7))

    # Plot in-sample actual vs predicted
    plt.subplot(1, 2, 1)
    plt.title('In-Sample: Actual vs Predicted (XGBoost)', fontsize=14)
    plt.scatter(y_train, train_preds_final, alpha=0.6, color='blue', edgecolor='black')
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 
            'r--', label='Perfect Prediction')
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    if not is_classification:
        # Add RMSE to the plot
        plt.annotate(f'RMSE: {train_rmse:.4f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    else:
        # Add Accuracy to the plot
        plt.annotate(f'Accuracy: {train_accuracy:.4f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

    # Plot out-of-sample actual vs predicted
    plt.subplot(1, 2, 2)
    plt.title('Out-of-Sample: Actual vs Predicted (XGBoost)', fontsize=14)
    plt.scatter(y_test, test_preds_final, alpha=0.6, color='green', edgecolor='black')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 
            'r--', label='Perfect Prediction')
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    if not is_classification:
        # Add RMSE to the plot
        plt.annotate(f'RMSE: {test_rmse:.4f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    else:
        # Add Accuracy to the plot
        plt.annotate(f'Accuracy: {test_accuracy:.4f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"dax_closed_price_xgboost_prediction_comparison{kk}.png", dpi=300)
    #plt.show()
    
    # Print individual predictions
    for i in range(len(test_preds_final)):
        print(f" out-of-sample no.{i}  real value {y_test[i]}  predict value: , {test_preds_final[i]}, error: {y_test[i]-test_preds_final[i]}")
    # Enhanced backtest visualization
# Insert this code in your backtest section (after calculating returns)
   
    # 1. Create a more detailed cumulative return plot
    def generate_enhanced_backtest_plots(backtest_df, kk):
        """
        Generate enhanced backtest visualization plots
        
        Parameters:
        backtest_df (pandas.DataFrame): DataFrame with backtest results
        kk (int): Iteration number
        """
        # Plot settings
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Generate multiple visualizations
        # 1. Combined plot with price and cumulative returns
        fig, axes = plt.subplots(3, 1, figsize=(16, 18))
        
        # Plot 1: Price and Predictions with trading signals
        ax1 = axes[0]
        ax1.set_title(f'Backtest Results - Price vs Predictions (Iteration {kk})', fontsize=16)
        ax1.plot(backtest_df['timestamp'], backtest_df['actual'], 'b-', label='Actual Price', linewidth=2)
        ax1.plot(backtest_df['timestamp'], backtest_df['predicted'], 'r--', label='Predicted Price', linewidth=2)
        
        # Mark long positions with green up arrows
        long_indices = backtest_df[backtest_df['position'] > 0]['timestamp']
        long_prices = backtest_df[backtest_df['position'] > 0]['actual']
        ax1.scatter(long_indices, long_prices, marker='^', color='green', s=150, label='Long', alpha=0.8)
        
        # Mark short positions with red down arrows
        short_indices = backtest_df[backtest_df['position'] < 0]['timestamp']
        short_prices = backtest_df[backtest_df['position'] < 0]['actual']
        ax1.scatter(short_indices, short_prices, marker='v', color='red', s=150, label='Short', alpha=0.8)
        
        # Mark entry/exit points where position changes
        position_changes = backtest_df[backtest_df['position'].diff() != 0]
        ax1.scatter(position_changes['timestamp'], position_changes['actual'], 
                marker='o', color='purple', s=100, label='Position Change', alpha=0.7)
        
        ax1.set_xlabel('Time', fontsize=14)
        ax1.set_ylabel('Price', fontsize=14)
        ax1.legend(loc='upper left', fontsize=12)
        ax1.grid(True, alpha=0.4)
        
        # Plot 2: Cumulative Returns - Enhanced with underwater plot
        ax2 = axes[1]
        ax2.set_title('Cumulative Returns Comparison', fontsize=16)
        ax2.plot(backtest_df['timestamp'], backtest_df['cum_market_return']*100, 'b-', 
                label='Market Return (%)', linewidth=2.5, alpha=0.8)
        ax2.plot(backtest_df['timestamp'], backtest_df['cum_strategy_return']*100, 'g-', 
                label='Strategy Return (%)', linewidth=2.5, alpha=0.8)
        
        # Add horizontal line at 0%
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Add vertical lines at significant trade points
        for idx in position_changes.index:
            if idx > 0:  # Skip the first entry which might be a position change from NaN
                ax2.axvline(x=position_changes.loc[idx, 'timestamp'], color='gray', 
                        linestyle='--', alpha=0.4)
        
        # Add text annotations for final returns
        final_market = backtest_df['cum_market_return'].iloc[-2]*100 if not backtest_df.empty and len(backtest_df) > 1 else 0
        final_strategy = backtest_df['cum_strategy_return'].iloc[-2]*100 if not backtest_df.empty and len(backtest_df) > 1 else 0
        
        ax2.text(0.02, 0.95, f'Final Market Return: {final_market:.2f}%', 
                transform=ax2.transAxes, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
        ax2.text(0.02, 0.87, f'Final Strategy Return: {final_strategy:.2f}%', 
                transform=ax2.transAxes, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
        
        ax2.set_xlabel('Time', fontsize=14)
        ax2.set_ylabel('Cumulative Return (%)', fontsize=14)
        ax2.legend(loc='lower left', fontsize=12)
        ax2.grid(True, alpha=0.4)
        
        # Plot 3: Drawdown analysis (underwater plot)
        ax3 = axes[2]
        ax3.set_title('Strategy Drawdown Analysis', fontsize=16)
        
        # Calculate rolling maximum for market and strategy
        rolling_max_market = backtest_df['cum_market_return'].cummax()
        rolling_max_strategy = backtest_df['cum_strategy_return'].cummax()
        
        # Calculate drawdowns
        drawdown_market = (backtest_df['cum_market_return'] - rolling_max_market) * 100  # Convert to percentage
        drawdown_strategy = (backtest_df['cum_strategy_return'] - rolling_max_strategy) * 100  # Convert to percentage
        
        # Plot drawdowns
        ax3.fill_between(backtest_df['timestamp'], 0, drawdown_market, color='blue', alpha=0.3, label='Market Drawdown')
        ax3.fill_between(backtest_df['timestamp'], 0, drawdown_strategy, color='green', alpha=0.3, label='Strategy Drawdown')
        
        # Calculate and display max drawdowns
        max_dd_market = drawdown_market.min()
        max_dd_strategy = drawdown_strategy.min()
        
        ax3.text(0.02, 0.15, f'Max Market Drawdown: {max_dd_market:.2f}%', 
                transform=ax3.transAxes, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
        ax3.text(0.02, 0.07, f'Max Strategy Drawdown: {max_dd_strategy:.2f}%', 
                transform=ax3.transAxes, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
        
        ax3.set_xlabel('Time', fontsize=14)
        ax3.set_ylabel('Drawdown (%)', fontsize=14)
        ax3.legend(loc='upper left', fontsize=12)
        ax3.grid(True, alpha=0.4)
        
        # Add backtest statistics
        total_trades = (backtest_df['position'].diff() != 0).sum()
        winning_trades = ((backtest_df['strategy_return'] > 0) & (backtest_df['position'] != 0)).sum()
        losing_trades = ((backtest_df['strategy_return'] < 0) & (backtest_df['position'] != 0)).sum()
        win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0
        
        # Calculate Sharpe ratio (assuming daily data - adjust if different frequency)
        strategy_return_mean = backtest_df['strategy_return'].mean()
        strategy_return_std = backtest_df['strategy_return'].std()
        sharpe_ratio = strategy_return_mean / strategy_return_std * np.sqrt(252) if strategy_return_std > 0 else 0
        
        # Create a text box with statistics
        stats_text = (
            f"Backtest Statistics:\n"
            f"Total Trades: {total_trades}\n"
            f"Win Rate: {win_rate:.2%}\n"
            f"Winning Trades: {winning_trades}\n"
            f"Losing Trades: {losing_trades}\n"
            f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
            f"Final Return: {final_strategy:.2f}%\n"
            f"Market Return: {final_market:.2f}%\n"
            f"Max Drawdown: {max_dd_strategy:.2f}%"
        )
        
        plt.figtext(0.85, 0.15, stats_text, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.8'))
        
        plt.tight_layout()
        plt.savefig(f"enhanced_backtest_results_iter{kk}.jpg", dpi=300, bbox_inches='tight')
        
        # 2. Create additional specialized plots
        # Monthly returns plot
        if len(backtest_df) > 20:  # Only create if we have enough data
            plt.figure(figsize=(14, 8))
            
            # Group returns by month (this is a simplification - adjust based on your actual timestamp data)
            # Assuming the timestamp is just an integer index, we'll create synthetic months
            backtest_df['month'] = (backtest_df['timestamp'] // 20) + 1  # Every 20 periods is a new "month"
            
            # Calculate monthly returns
            monthly_market = backtest_df.groupby('month')['market_return'].sum() * 100
            monthly_strategy = backtest_df.groupby('month')['strategy_return'].sum() * 100
            
            # Plot monthly returns
            months = monthly_market.index
            width = 0.35
            
            plt.bar(months - width/2, monthly_market, width, label='Market', alpha=0.7, color='blue')
            plt.bar(months + width/2, monthly_strategy, width, label='Strategy', alpha=0.7, color='green')
            
            plt.title('Monthly Returns Comparison', fontsize=16)
            plt.xlabel('Month', fontsize=14)
            plt.ylabel('Return (%)', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.savefig(f"monthly_returns_iter{kk}.jpg", dpi=300, bbox_inches='tight')
        
        # 3. Return distribution analysis
        plt.figure(figsize=(14, 8))
        
        plt.subplot(1, 2, 1)
        plt.hist(backtest_df['market_return']*100, bins=20, alpha=0.7, color='blue', label='Market')
        plt.hist(backtest_df['strategy_return']*100, bins=20, alpha=0.7, color='green', label='Strategy')
        plt.title('Return Distribution', fontsize=16)
        plt.xlabel('Daily Return (%)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        # Remove NaN values before plotting
        clean_market = backtest_df['market_return'].dropna() * 100
        clean_strategy = backtest_df['strategy_return'].dropna() * 100
        
        if len(clean_market) > 1 and len(clean_strategy) > 1:  # Only plot if we have enough data
            plt.scatter(clean_market, clean_strategy, alpha=0.7, edgecolor='black')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            
            # Add regression line
            if len(clean_market) > 2:  # Need at least 3 points for regression
                z = np.polyfit(clean_market, clean_strategy, 1)
                p = np.poly1d(z)
                plt.plot(clean_market, p(clean_market), "r--", alpha=0.8)
                plt.text(0.05, 0.95, f"y = {z[0]:.2f}x + {z[1]:.2f}", transform=plt.gca().transAxes,
                        fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        
        plt.title('Strategy vs Market Returns', fontsize=16)
        plt.xlabel('Market Return (%)', fontsize=14)
        plt.ylabel('Strategy Return (%)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"return_analysis_iter{kk}.jpg", dpi=300, bbox_inches='tight')
        
        return backtest_df
            # Create strategy-only visualization function
    def plot_strategy_only_returns(backtest_df, kk):
        """
        Generate visualizations showing only strategy returns without market comparison
        
        Parameters:
        backtest_df (pandas.DataFrame): DataFrame with backtest results
        kk (int): Iteration number
        """
        # Plot settings
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create new figure for strategy returns
        fig, axes = plt.subplots(2, 1, figsize=(16, 14))
        
        # Plot 1: Price and Predictions with trading signals - Focus on Strategy
        ax1 = axes[0]
        ax1.set_title(f'Strategy Prediction & Trading Signals (Iteration {kk})', fontsize=16)
        ax1.plot(backtest_df['timestamp'], backtest_df['actual'], 'b-', label='Actual Price', linewidth=2)
        ax1.plot(backtest_df['timestamp'], backtest_df['predicted'], 'r--', label='Predicted Price', linewidth=2)
        
        # Mark long positions with green up arrows
        long_indices = backtest_df[backtest_df['position'] > 0]['timestamp']
        long_prices = backtest_df[backtest_df['position'] > 0]['actual']
        ax1.scatter(long_indices, long_prices, marker='^', color='green', s=150, label='Long', alpha=0.8)
        
        # Mark short positions with red down arrows
        short_indices = backtest_df[backtest_df['position'] < 0]['timestamp']
        short_prices = backtest_df[backtest_df['position'] < 0]['actual']
        ax1.scatter(short_indices, short_prices, marker='v', color='red', s=150, label='Short', alpha=0.8)
        
        # Mark entry/exit points where position changes
        position_changes = backtest_df[backtest_df['position'].diff() != 0]
        ax1.scatter(position_changes['timestamp'], position_changes['actual'], 
                marker='o', color='purple', s=100, label='Position Change', alpha=0.7)
        
        ax1.set_xlabel('Time', fontsize=14)
        ax1.set_ylabel('Price', fontsize=14)
        ax1.legend(loc='upper left', fontsize=12)
        ax1.grid(True, alpha=0.4)
        
        # Plot 2: Strategy Cumulative Returns
        ax2 = axes[1]
        ax2.set_title('Strategy Cumulative Return', fontsize=16)
        ax2.plot(backtest_df['timestamp'], backtest_df['cum_strategy_return']*100, 'g-', 
                label='Strategy Return (%)', linewidth=3, alpha=0.8)
        
        # Add horizontal line at 0%
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Add vertical lines at position changes
        for idx in position_changes.index:
            if idx > 0:  # Skip the first entry which might be a position change from NaN
                ax2.axvline(x=position_changes.loc[idx, 'timestamp'], color='gray', 
                        linestyle='--', alpha=0.4)
        
        # Add text annotations for final returns
        final_strategy = backtest_df['cum_strategy_return'].iloc[-2]*100 if not backtest_df.empty and len(backtest_df) > 1 else 0
        
        ax2.text(0.02, 0.9, f'Final Strategy Return: {final_strategy:.2f}%', 
                transform=ax2.transAxes, fontsize=14, 
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
        
        ax2.set_xlabel('Time', fontsize=14)
        ax2.set_ylabel('Cumulative Return (%)', fontsize=14)
        ax2.legend(loc='lower left', fontsize=12)
        ax2.grid(True, alpha=0.4)
        
        # Calculate strategy statistics
        total_trades = (backtest_df['position'].diff() != 0).sum()
        winning_trades = ((backtest_df['strategy_return'] > 0) & (backtest_df['position'] != 0)).sum()
        losing_trades = ((backtest_df['strategy_return'] < 0) & (backtest_df['position'] != 0)).sum()
        win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0
        
        # Calculate advanced metrics
        strategy_return_mean = backtest_df['strategy_return'].mean()
        strategy_return_std = backtest_df['strategy_return'].std()
        sharpe_ratio = strategy_return_mean / strategy_return_std * np.sqrt(252) if strategy_return_std > 0 else 0
        
        # Calculate drawdown
        rolling_max_strategy = backtest_df['cum_strategy_return'].cummax()
        drawdown_strategy = (backtest_df['cum_strategy_return'] - rolling_max_strategy) * 100
        max_dd_strategy = drawdown_strategy.min()
        
        # Create a text box with statistics
        stats_text = (
            f"Strategy Statistics:\n"
            f"Total Trades: {total_trades}\n"
            f"Win Rate: {win_rate:.2%}\n"
            f"Winning Trades: {winning_trades}\n"
            f"Losing Trades: {losing_trades}\n"
            f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
            f"Final Return: {final_strategy:.2f}%\n"
            f"Max Drawdown: {max_dd_strategy:.2f}%"
        )
        
        plt.figtext(0.85, 0.2, stats_text, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.8'))
        
        plt.tight_layout()
        plt.savefig(f"strategy_only_returns_iter{kk}.jpg", dpi=300, bbox_inches='tight')
        
        # Create additional strategy-specific visualizations
        
        # 1. Strategy Return Distribution
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(backtest_df['strategy_return']*100, bins=20, alpha=0.7, color='green')
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        plt.title('Strategy Return Distribution', fontsize=16)
        plt.xlabel('Daily Return (%)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 2. Strategy Drawdown
        plt.subplot(1, 2, 2)
        plt.fill_between(backtest_df['timestamp'], 0, drawdown_strategy, color='red', alpha=0.3)
        plt.title('Strategy Drawdown', fontsize=16)
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Drawdown (%)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.text(0.05, 0.05, f'Max Drawdown: {max_dd_strategy:.2f}%', 
                transform=plt.gca().transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f"strategy_return_analysis_iter{kk}.jpg", dpi=300, bbox_inches='tight')
        
        # 3. Create strategy-only position visualizations
        plt.figure(figsize=(14, 7))
        plt.title('Strategy Positions and Returns', fontsize=16)
        
        # Create a colormap for returns - red for negative, green for positive
        colors = np.where(backtest_df['strategy_return'] > 0, 'green', 'red')
        
        # Plot strategy returns as bars
        plt.bar(backtest_df['timestamp'], backtest_df['strategy_return']*100, 
            color=colors, alpha=0.6, label='Daily Returns (%)')
        
        # Plot position line (secondary y-axis)
        ax_pos = plt.gca().twinx()
        ax_pos.plot(backtest_df['timestamp'], backtest_df['position'], 
                'b-', linewidth=2, alpha=0.7, label='Position')
        ax_pos.set_ylabel('Position (-1=Short, 0=Neutral, 1=Long)', fontsize=14)
        ax_pos.set_ylim(-1.5, 1.5)
        
        # Create combined legend
        lines1, labels1 = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax_pos.get_legend_handles_labels()
        ax_pos.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)
        
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Return (%)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"strategy_positions_returns_iter{kk}.jpg", dpi=300, bbox_inches='tight')
        
        return backtest_df

# To use this in your code, replace the line that calls generate_enhanced_backtest_plots
# with this line instead:
    backtest_df = plot_strategy_only_returns(backtest_df, kk)
# Insert this function call after creating your backtest_df
# and before your existing plots (around line 383 in your code)
# Right before the 'Visualization backtest results' section:

# Add this line to your existing code at the appropriate position:
    backtest_df = generate_enhanced_backtest_plots(backtest_df, kk)