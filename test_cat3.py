name="dax"
for kk in range(20):
        
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
    from sklearn.metrics import mean_absolute_error, r2_score, precision_score, recall_score, f1_score
    import catboost as cb
    from catboost import CatBoostRegressor, CatBoostClassifier
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
    window_size = 30
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
    X_windows_scaled = scaler_X.fit_transform(X_windows_flat)  # Keep as 2D for CatBoost

    # Store original indices for later visualization
    all_indices = np.arange(len(X_windows_scaled))

    # 4. Split data into training and testing sets SEQUENTIALLY
    # Using first 50 samples for training, next 10 for testing
    train_size = 905+kk  # First 50 samples
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
    y_train = y_windows[train_indices]  # Using original values directly with CatBoost
    y_test = y_windows[test_indices]

    # Get original dataframe indices for test data
    original_test_indices = indices_map[test_indices]

    print(f"Sequential Training set: {len(X_train)} samples (indices 0-{train_size-1})")
    print(f"Sequential Testing set: {len(X_test)} samples (indices {train_size}-{train_size+test_size-1})")

    # 5. Create and configure CatBoost model
    # Configure parameters based on task type
    if is_classification:
        # For classification tasks:
        print(f"Training CatBoost for classification with {num_classes} classes")
        model = CatBoostClassifier(
            iterations=10000,
            learning_rate=0.01,
            depth=6,
            subsample=0.8,
            colsample_bylevel=0.8,
            min_data_in_leaf=1,
            loss_function='MultiClass' if num_classes > 2 else 'Logloss',
            eval_metric='MultiClass' if num_classes > 2 else 'Logloss',
            random_seed=42,
            verbose=100,
            early_stopping_rounds=50,
            use_best_model=True,
            border_count=128,
            l2_leaf_reg=3,
            bagging_temperature=1
        )
    else:
        # For regression tasks:
        print("Training CatBoost for regression")
        model = CatBoostRegressor(
            iterations=10000,
            learning_rate=0.01,
            depth=6,
            subsample=0.8,
            colsample_bylevel=0.8,
            min_data_in_leaf=1,
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=42,
            verbose=100,
            early_stopping_rounds=50,
            use_best_model=True,
            border_count=128,
            l2_leaf_reg=3,
            bagging_temperature=1
        )

    # 6. Train CatBoost model
    print("Training CatBoost model...")
    
    # Fit the model with validation set
    model.fit(
        X_train, 
        y_train,
        eval_set=(X_test, y_test),
        plot=False,  # Set to True if you want to see training plots
        verbose=False
    )

    print(f"CatBoost training completed with best iteration: {model.best_iteration_}")

    # 7. Generate predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # Process predictions based on task type
    if is_classification:
        # For classification, CatBoost returns class predictions directly
        train_preds_final = train_preds.astype(int)
        test_preds_final = test_preds.astype(int)
    else:
        # For regression, predictions are already the final values
        train_preds_final = train_preds
        test_preds_final = test_preds

    # 8. Evaluate model performance
    print("\nCatBoost Performance Metrics Summary:")

    if not is_classification:
        # For regression task
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds_final))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds_final))
        
        # Calculate additional metrics
        # Mean Squared Error
        train_mse = mean_squared_error(y_train, train_preds_final)
        test_mse = mean_squared_error(y_test, test_preds_final)
        
        # Mean Absolute Error
        train_mae = mean_absolute_error(y_train, train_preds_final)
        test_mae = mean_absolute_error(y_test, test_preds_final)
        
        # R^2 Score (Coefficient of Determination)
        train_r2 = r2_score(y_train, train_preds_final)
        test_r2 = r2_score(y_test, test_preds_final)
        
        # Mean Absolute Percentage Error
        def mean_absolute_percentage_error(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-7, None))) * 100
        
        train_mape = mean_absolute_percentage_error(y_train, train_preds_final)
        test_mape = mean_absolute_percentage_error(y_test, test_preds_final)
        
        # Mean Directional Accuracy (MDA)
        def mean_directional_accuracy(y_true, y_pred):
            if len(y_true) < 2:
                return 0
            actual_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            return np.mean(actual_direction == pred_direction) * 100
        
        train_mda = mean_directional_accuracy(y_train, train_preds_final)
        test_mda = mean_directional_accuracy(y_test, test_preds_final)
        
        print(f"In-Sample RMSE: {train_rmse:.4f}")
        print(f"Out-of-Sample RMSE: {test_rmse:.4f}")
        print(f"In-Sample MSE: {train_mse:.4f}")
        print(f"Out-of-Sample MSE: {test_mse:.4f}")
        print(f"In-Sample MAE: {train_mae:.4f}")
        print(f"Out-of-Sample MAE: {test_mae:.4f}")
        print(f"In-Sample R²: {train_r2:.4f}")
        print(f"Out-of-Sample R²: {test_r2:.4f}")
        print(f"In-Sample MAPE: {train_mape:.4f}%")
        print(f"Out-of-Sample MAPE: {test_mape:.4f}%")
        print(f"In-Sample MDA: {train_mda:.4f}%")
        print(f"Out-of-Sample MDA: {test_mda:.4f}%")
        
        # Calculate errors
        test_errors = y_test - test_preds_final
        
        # Create DataFrame for Excel export with all metrics
        export_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': test_preds_final,
            'Error': test_errors,
            'Absolute_Error': np.abs(test_errors),
            'Percentage_Error': (test_errors / y_test) * 100,
            'Squared_Error': test_errors ** 2
        })
        
        # Create a detailed metrics summary dataframe
        metrics_df = pd.DataFrame({
            'Metric': ['RMSE', 'MSE', 'MAE', 'R²', 'MAPE (%)', 'MDA (%)'],
            'In-Sample': [train_rmse, train_mse, train_mae, train_r2, train_mape, train_mda],
            'Out-of-Sample': [test_rmse, test_mse, test_mae, test_r2, test_mape, test_mda]
        })
        
    else:
        # For classification task
        train_accuracy = np.mean(train_preds_final == y_train)
        test_accuracy = np.mean(test_preds_final == y_test)
        
        # Additional classification metrics
        # For multi-class problems, use macro averaging
        if num_classes > 2:
            train_precision = precision_score(y_train, train_preds_final, average='macro', zero_division=0)
            test_precision = precision_score(y_test, test_preds_final, average='macro', zero_division=0)
            train_recall = recall_score(y_train, train_preds_final, average='macro', zero_division=0)
            test_recall = recall_score(y_test, test_preds_final, average='macro', zero_division=0)
            train_f1 = f1_score(y_train, train_preds_final, average='macro', zero_division=0)
            test_f1 = f1_score(y_test, test_preds_final, average='macro', zero_division=0)
        else:
            # For binary classification
            train_precision = precision_score(y_train, train_preds_final, zero_division=0)
            test_precision = precision_score(y_test, test_preds_final, zero_division=0)
            train_recall = recall_score(y_train, train_preds_final, zero_division=0)
            test_recall = recall_score(y_test, test_preds_final, zero_division=0)
            train_f1 = f1_score(y_train, train_preds_final, zero_division=0)
            test_f1 = f1_score(y_test, test_preds_final, zero_division=0)
        
        # Get confusion matrix as string
        test_cm = confusion_matrix(y_test, test_preds_final)
        cm_string = str(test_cm).replace('\n', '; ')
        
        print(f"In-Sample Accuracy: {train_accuracy:.4f}")
        print(f"Out-of-Sample Accuracy: {test_accuracy:.4f}")
        print(f"In-Sample Precision: {train_precision:.4f}")
        print(f"Out-of-Sample Precision: {test_precision:.4f}")
        print(f"In-Sample Recall: {train_recall:.4f}")
        print(f"Out-of-Sample Recall: {test_recall:.4f}")
        print(f"In-Sample F1 Score: {train_f1:.4f}")
        print(f"Out-of-Sample F1 Score: {test_f1:.4f}")
        
        # Calculate errors (for classification, this is just whether prediction was correct)
        test_errors = y_test != test_preds_final
        
        # Create metrics summary dataframe for classification
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Confusion Matrix'],
            'In-Sample': [train_accuracy, train_precision, train_recall, train_f1, ''],
            'Out-of-Sample': [test_accuracy, test_precision, test_recall, test_f1, cm_string]
        })
        
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
                'Error': test_errors.astype(int),  # 0 for correct, 1 for incorrect
                'Is_Correct': (~test_errors).astype(int)
            })
        else:
            # Create DataFrame for Excel export
            export_df = pd.DataFrame({
                'Actual': y_test,
                'Predicted': test_preds_final,
                'Error': test_errors.astype(int),  # 0 for correct, 1 for incorrect
                'Is_Correct': (~test_errors).astype(int)
            })
    
    # Create a filename with the iteration number
    excel_filename = f"predicted_catboost_iter{kk}.xlsx"
    
    # Export to Excel with multiple sheets
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        export_df.to_excel(writer, sheet_name='Predictions', index=True)
        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
    
    print(f"Saved prediction results and metrics to {excel_filename}")
    
    # Also create a combined file for all iterations
    # Create or append to a master results file
    if kk == 0:
        # Create a new file for the first iteration
        with pd.ExcelWriter("predicted_catboost.xlsx", engine='openpyxl') as writer:
            export_df.to_excel(writer, sheet_name=f'Iter{kk}_Data', index=True)
            metrics_df.to_excel(writer, sheet_name=f'Iter{kk}_Metrics', index=False)
    else:
        # Read existing file and append new sheet
        try:
            with pd.ExcelWriter("predicted_catboost.xlsx", mode='a', if_sheet_exists='replace', engine='openpyxl') as writer:
                export_df.to_excel(writer, sheet_name=f'Iter{kk}_Data', index=True)
                metrics_df.to_excel(writer, sheet_name=f'Iter{kk}_Metrics', index=False)
        except Exception as e:
            print(f"Warning: Could not append to combined file: {e}")
    
    print(f"Updated combined results in predicted_catboost.xlsx")

    # 9. BACKTESTING IMPLEMENTATION
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
        
        # Calculate additional trading metrics
        strategy_return_mean = backtest_df['strategy_return'].mean()
        strategy_return_std = backtest_df['strategy_return'].std()
        sharpe_ratio = strategy_return_mean / strategy_return_std * np.sqrt(252) if strategy_return_std > 0 else 0
        
        # Maximum drawdown calculation
        cum_returns = backtest_df['cum_strategy_return']
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / (1 + running_max)
        max_drawdown = drawdown.min()
        
        # Calmar ratio (Annual return / Max drawdown)
        annual_return = backtest_df['cum_strategy_return'].iloc[-2] if not pd.isna(backtest_df['cum_strategy_return'].iloc[-2]) else 0
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        print(f"Backtest Results:")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {losing_trades}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Strategy Return: {backtest_df['cum_strategy_return'].iloc[-2]:.2%}")
        print(f"Market Return: {backtest_df['cum_market_return'].iloc[-2]:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
        print(f"Calmar Ratio: {calmar_ratio:.2f}")
        
        # Create a backtest results sheet for Excel
        backtest_summary = pd.DataFrame({
            'Metric': ['Total Trades', 'Winning Trades', 'Losing Trades', 'Win Rate', 'Strategy Return', 
                      'Market Return', 'Sharpe Ratio', 'Max Drawdown', 'Calmar Ratio'],
            'Value': [total_trades, winning_trades, losing_trades, f"{win_rate:.2%}", 
                     f"{backtest_df['cum_strategy_return'].iloc[-2]:.2%}", 
                     f"{backtest_df['cum_market_return'].iloc[-2]:.2%}", 
                     f"{sharpe_ratio:.2f}", f"{max_drawdown:.2%}", f"{calmar_ratio:.2f}"]
        })
        
        # Export backtest data
        with pd.ExcelWriter(f"backtest_results_iter{kk}.xlsx", engine='openpyxl') as writer:
            backtest_df.to_excel(writer, sheet_name='BacktestData', index=False)
            backtest_summary.to_excel(writer, sheet_name='Summary', index=False)
            # Add model metrics to backtest results file
            metrics_df.to_excel(writer, sheet_name='ModelMetrics', index=False)
            
        # Update combined results
        try:
            with pd.ExcelWriter("predicted_catboost.xlsx", mode='a', if_sheet_exists='replace', engine='openpyxl') as writer:
                backtest_df.to_excel(writer, sheet_name=f'Iter{kk}_Backtest', index=False)
                backtest_summary.to_excel(writer, sheet_name=f'Iter{kk}_BacktestSummary', index=False)
        except Exception as e:
            print(f"Error updating combined file with backtest results: {e}")
        
        # Visualize backtest results
        plt.figure(figsize=(16, 10))
        
        # Plot 1: Price and Predictions
        plt.subplot(3, 1, 1)
        plt.title(f'Backtest Results - Price vs Predictions (Iteration {kk})', fontsize=14)
        plt.plot(backtest_df['timestamp'], backtest_df['actual'], 'b-', label='Actual Price', alpha=0.7, linewidth=2)
        plt.plot(backtest_df['timestamp'], backtest_df['predicted'], 'r--', label='Predicted Price', alpha=0.7, linewidth=2)
        
        # Mark long positions with green up arrows
        long_indices = backtest_df[backtest_df['position'] > 0]['timestamp']
        long_prices = backtest_df[backtest_df['position'] > 0]['actual']
        plt.scatter(long_indices, long_prices, marker='^', color='green', s=80, label='Long', alpha=0.8)
        
        # Mark short positions with red down arrows
        short_indices = backtest_df[backtest_df['position'] < 0]['timestamp']
        short_prices = backtest_df[backtest_df['position'] < 0]['actual']
        plt.scatter(short_indices, short_prices, marker='v', color='red', s=80, label='Short', alpha=0.8)
        
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative Returns
        plt.subplot(3, 1, 2)
        plt.title('Cumulative Returns Comparison', fontsize=14)
        plt.plot(backtest_df['timestamp'], backtest_df['cum_market_return'] * 100, 'b-', 
                label='Market Return', alpha=0.7, linewidth=2)
        plt.plot(backtest_df['timestamp'], backtest_df['cum_strategy_return'] * 100, 'g-', 
                label='Strategy Return', alpha=0.7, linewidth=2)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Return (%)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Drawdown
        plt.subplot(3, 1, 3)
        plt.title('Strategy Drawdown', fontsize=14)
        plt.fill_between(backtest_df['timestamp'], drawdown * 100, 0, color='red', alpha=0.3)
        plt.plot(backtest_df['timestamp'], drawdown * 100, 'r-', linewidth=1)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"backtest_results_iter{kk}.png", dpi=300, bbox_inches='tight')
        
        # Save as JPG as well
        plt.savefig(f"backtest_results_iter{kk}.jpg", dpi=300, bbox_inches='tight')
        plt.close()
    
    else:
        # For classification tasks, create a simple accuracy-based backtest
        print("Classification-based backtesting - simplified version")
        
        # Create a simple strategy based on prediction confidence
        backtest_df = pd.DataFrame({
            'timestamp': [i for i in range(len(test_indices))],
            'index': original_test_indices,
            'actual': y_test,
            'predicted': test_preds_final,
            'correct': (y_test == test_preds_final).astype(int)
        })
        
        # Calculate running accuracy
        backtest_df['running_accuracy'] = backtest_df['correct'].expanding().mean()
        
        # Export classification backtest
        with pd.ExcelWriter(f"backtest_results_iter{kk}.xlsx", engine='openpyxl') as writer:
            backtest_df.to_excel(writer, sheet_name='ClassificationBacktest', index=False)
            metrics_df.to_excel(writer, sheet_name='ModelMetrics', index=False)

    # 10. Feature importance visualization
    try:
        # Get feature importances from CatBoost
        feature_importances = model.get_feature_importance()
        feature_names = [f'Feature_{i}' for i in range(len(feature_importances))]
        
        plt.figure(figsize=(12, 8))
        
        # Sort features by importance
        indices = np.argsort(feature_importances)[::-1][:20]  # Top 20 features
        
        plt.barh(range(len(indices)), feature_importances[indices], color='skyblue', alpha=0.8)
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title('CatBoost Feature Importances (Top 20)', fontsize=14)
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for i, v in enumerate(feature_importances[indices]):
            plt.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"catboost_feature_importance{kk}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"catboost_feature_importance{kk}.jpg", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error creating feature importance plot: {e}")

    # 11. Create visualizations with in-sample and out-of-sample data
    plt.figure(figsize=(16, 12))

    # Plot training and testing data
    plt.subplot(2, 1, 1)
    plt.title('DAX Closed Price Prediction with CatBoost: In-Sample vs Out-of-Sample Comparison', fontsize=16, pad=20)

    # Plot in-sample data
    plt.plot(train_indices, y_train, 'bo', label='In-Sample Actual', alpha=0.6, markersize=4)
    plt.plot(train_indices, train_preds_final, 'ro', label='In-Sample Predicted', alpha=0.6, markersize=4)

    # Plot out-of-sample data
    plt.plot(test_indices, y_test, 'go', label='Out-of-Sample Actual', markersize=6, alpha=0.8)
    plt.plot(test_indices, test_preds_final, 'mo', label='Out-of-Sample Predicted', markersize=6, alpha=0.8)

    plt.xlabel('Sample Index', fontsize=14)
    plt.ylabel('Closed Value', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add mapping legend if categorical
    if encoder is not None and len(encoder.classes_) <= 10:
        legend_text = "\n".join([f"{i}: {val}" for i, val in enumerate(encoder.classes_)])
        plt.figtext(0.02, 0.02, f"Closed Encoding:\n{legend_text}", 
                    bbox=dict(facecolor='white', alpha=0.8))

    # Plot the time series view
    plt.subplot(2, 1, 2)
    plt.title('DAX Closed Price Prediction with CatBoost: Time Series View', fontsize=16, pad=20)

    # Continuous plot of all available data to show sequence
    x_train = np.arange(len(y_train))
    x_test = np.arange(len(y_train), len(y_train) + len(y_test))

    # Plot actual values
    plt.plot(x_train, y_train, 'b-', label='In-Sample Actual', alpha=0.6, linewidth=1)
    plt.plot(x_test, y_test, 'g-', label='Out-of-Sample Actual', alpha=0.8, linewidth=2)

    # Plot predicted values
    plt.plot(x_train, train_preds_final, 'r-', label='In-Sample Predicted', alpha=0.6, linewidth=1)
    plt.plot(x_test, test_preds_final, 'm-', label='Out-of-Sample Predicted', alpha=0.8, linewidth=2)

    # Highlight the boundary between training and testing
    plt.axvline(x=len(y_train), color='black', linestyle='--', 
                label='Train/Test Boundary', linewidth=2)

    plt.xlabel('Sequential Index', fontsize=14)
    plt.ylabel('Closed Value', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"dax_closed_price_catboost_comparison{kk}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"dax_closed_price_catboost_comparison{kk}.jpg", dpi=300, bbox_inches='tight')
    plt.close()

    # 12. Create a detailed comparison of in-sample and out-of-sample predictions
    plt.figure(figsize=(16, 8))

    # Plot in-sample actual vs predicted
    plt.subplot(1, 2, 1)
    plt.title('In-Sample: Actual vs Predicted (CatBoost)', fontsize=14)
    plt.scatter(y_train, train_preds_final, alpha=0.6, color='blue', edgecolor='navy', s=30)
    
    # Perfect prediction line
    min_val = min(min(y_train), min(train_preds_final))
    max_val = max(max(y_train), max(train_preds_final))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction', linewidth=2)
    
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    if not is_classification:
        # Add RMSE to the plot
        plt.annotate(f'RMSE: {train_rmse:.4f}\nR²: {train_r2:.4f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="navy", alpha=0.8),
                    fontsize=10, verticalalignment='top')
    else:
        # Add Accuracy to the plot
        plt.annotate(f'Accuracy: {train_accuracy:.4f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="navy", alpha=0.8),
                    fontsize=10, verticalalignment='top')

    # Plot out-of-sample actual vs predicted
    plt.subplot(1, 2, 2)
    plt.title('Out-of-Sample: Actual vs Predicted (CatBoost)', fontsize=14)
    plt.scatter(y_test, test_preds_final, alpha=0.6, color='green', edgecolor='darkgreen', s=50)
    
    # Perfect prediction line
    min_val = min(min(y_test), min(test_preds_final))
    max_val = max(max(y_test), max(test_preds_final))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction', linewidth=2)
    
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    if not is_classification:
        # Add RMSE to the plot
        plt.annotate(f'RMSE: {test_rmse:.4f}\nR²: {test_r2:.4f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="darkgreen", alpha=0.8),
                    fontsize=10, verticalalignment='top')
    else:
        # Add Accuracy to the plot
        plt.annotate(f'Accuracy: {test_accuracy:.4f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="darkgreen", alpha=0.8),
                    fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.savefig(f"catboost_actual_vs_predicted{kk}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"catboost_actual_vs_predicted{kk}.jpg", dpi=300, bbox_inches='tight')
    plt.close()

    # 13. Create residuals analysis plot (for regression only)
    if not is_classification:
        plt.figure(figsize=(16, 10))
        
        # Residuals vs Fitted
        plt.subplot(2, 2, 1)
        residuals = y_test - test_preds_final
        plt.scatter(test_preds_final, residuals, alpha=0.6, color='blue', s=30)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Fitted Values')
        plt.grid(True, alpha=0.3)
        
        # Histogram of residuals
        plt.subplot(2, 2, 2)
        plt.hist(residuals, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Distribution of Residuals')
        plt.grid(True, alpha=0.3)
        
        # Q-Q Plot
        from scipy import stats
        plt.subplot(2, 2, 3)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot of Residuals')
        plt.grid(True, alpha=0.3)
        
        # Time series of residuals
        plt.subplot(2, 2, 4)
        plt.plot(range(len(residuals)), residuals, 'bo-', alpha=0.6, markersize=4)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('Residuals')
        plt.title('Residuals Over Time')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"catboost_residuals_analysis{kk}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"catboost_residuals_analysis{kk}.jpg", dpi=300, bbox_inches='tight')
        plt.close()

    # 14. Create prediction error analysis
    plt.figure(figsize=(14, 8))
    
    if not is_classification:
        # Error metrics over time
        plt.subplot(2, 1, 1)
        absolute_errors = np.abs(y_test - test_preds_final)
        plt.plot(range(len(absolute_errors)), absolute_errors, 'ro-', alpha=0.7, markersize=4)
        plt.xlabel('Test Sample Index')
        plt.ylabel('Absolute Error')
        plt.title('Absolute Prediction Errors Over Time')
        plt.grid(True, alpha=0.3)
        
        # Cumulative error
        plt.subplot(2, 1, 2)
        cumulative_error = np.cumsum(absolute_errors)
        plt.plot(range(len(cumulative_error)), cumulative_error, 'go-', alpha=0.7, markersize=4)
        plt.xlabel('Test Sample Index')
        plt.ylabel('Cumulative Absolute Error')
        plt.title('Cumulative Absolute Error')
        plt.grid(True, alpha=0.3)
    else:
        # Accuracy over time for classification
        plt.subplot(2, 1, 1)
        correct_predictions = (y_test == test_preds_final).astype(int)
        plt.plot(range(len(correct_predictions)), correct_predictions, 'bo-', alpha=0.7, markersize=4)
        plt.xlabel('Test Sample Index')
        plt.ylabel('Correct (1) / Incorrect (0)')
        plt.title('Prediction Accuracy Over Time')
        plt.grid(True, alpha=0.3)
        
        # Running accuracy
        plt.subplot(2, 1, 2)
        running_accuracy = np.cumsum(correct_predictions) / (np.arange(len(correct_predictions)) + 1)
        plt.plot(range(len(running_accuracy)), running_accuracy, 'go-', alpha=0.7, markersize=4)
        plt.xlabel('Test Sample Index')
        plt.ylabel('Running Accuracy')
        plt.title('Running Accuracy')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"catboost_error_analysis{kk}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"catboost_error_analysis{kk}.jpg", dpi=300, bbox_inches='tight')
    plt.close()

    # 15. Create a comprehensive summary plot
    fig = plt.figure(figsize=(20, 12))
    
    # Main prediction plot
    plt.subplot(2, 3, (1, 2))
    plt.plot(test_indices, y_test, 'go-', label='Actual', markersize=6, linewidth=2, alpha=0.8)
    plt.plot(test_indices, test_preds_final, 'ro-', label='Predicted', markersize=6, linewidth=2, alpha=0.8)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(f'CatBoost Predictions vs Actual (Iteration {kk})', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Performance metrics text
    plt.subplot(2, 3, 3)
    plt.axis('off')
    if not is_classification:
        metrics_text = f"""Performance Metrics:
        
Out-of-Sample RMSE: {test_rmse:.4f}
Out-of-Sample MAE: {test_mae:.4f}
Out-of-Sample R²: {test_r2:.4f}
Out-of-Sample MAPE: {test_mape:.2f}%
Out-of-Sample MDA: {test_mda:.2f}%

Training Samples: {len(y_train)}
Test Samples: {len(y_test)}
Window Size: {window_size}
Best Iteration: {model.best_iteration_}"""
    else:
        metrics_text = f"""Performance Metrics:
        
Out-of-Sample Accuracy: {test_accuracy:.4f}
Out-of-Sample Precision: {test_precision:.4f}
Out-of-Sample Recall: {test_recall:.4f}
Out-of-Sample F1: {test_f1:.4f}

Training Samples: {len(y_train)}
Test Samples: {len(y_test)}
Window Size: {window_size}
Best Iteration: {model.best_iteration_}"""
    
    plt.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Feature importance (top 10)
    plt.subplot(2, 3, 4)
    try:
        feature_importances = model.get_feature_importance()
        top_10_indices = np.argsort(feature_importances)[-10:]
        top_10_importances = feature_importances[top_10_indices]
        feature_names_top10 = [f'F{i}' for i in top_10_indices]
        
        plt.barh(range(len(top_10_importances)), top_10_importances, color='lightcoral', alpha=0.8)
        plt.yticks(range(len(top_10_importances)), feature_names_top10)
        plt.xlabel('Importance')
        plt.title('Top 10 Feature Importances')
        plt.grid(True, alpha=0.3)
    except:
        plt.text(0.5, 0.5, 'Feature importance\nnot available', ha='center', va='center')
        plt.title('Feature Importances')
    
    # Actual vs Predicted scatter
    plt.subplot(2, 3, 5)
    plt.scatter(y_test, test_preds_final, alpha=0.6, color='purple', s=50)
    min_val = min(min(y_test), min(test_preds_final))
    max_val = max(max(y_test), max(test_preds_final))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.grid(True, alpha=0.3)
    
    # Error distribution
    plt.subplot(2, 3, 6)
    if not is_classification:
        errors = y_test - test_preds_final
        plt.hist(errors, bins=15, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.axvline(x=0, color='red', linestyle='--')
    else:
        correct = (y_test == test_preds_final).astype(int)
        plt.bar(['Incorrect', 'Correct'], [np.sum(1-correct), np.sum(correct)], 
                color=['red', 'green'], alpha=0.7)
        plt.ylabel('Count')
        plt.title('Prediction Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'CatBoost Model Summary - Iteration {kk}', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(f"catboost_summary{kk}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"catboost_summary{kk}.jpg", dpi=300, bbox_inches='tight')
    plt.close()

    # Clean up memory
    plt.close('all')

    print(f"\nIteration {kk} completed successfully!")
    print("="*60)
    print(f"Files generated for iteration {kk}:")
    print(f"  - predicted_catboost_iter{kk}.xlsx")
    print(f"  - backtest_results_iter{kk}.xlsx")
    print(f"  - catboost_feature_importance{kk}.png/jpg")
    print(f"  - dax_closed_price_catboost_comparison{kk}.png/jpg")
    print(f"  - catboost_actual_vs_predicted{kk}.png/jpg")
    if not is_classification:
        print(f"  - catboost_residuals_analysis{kk}.png/jpg")
        print(f"  - backtest_results_iter{kk}.png/jpg")
    print(f"  - catboost_error_analysis{kk}.png/jpg")
    print(f"  - catboost_summary{kk}.png/jpg")
    print("="*60)

print(f"\n🎉 All {20} iterations completed successfully!")
print(f"📊 Master file created: predicted_catboost.xlsx")
print(f"📈 Total graphs generated: {20 * 7} images (PNG and JPG formats)")
print("✅ CatBoost analysis complete!")