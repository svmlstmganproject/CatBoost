import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def extract_set50_data():
    """Extract RMSE and Sharpe ratio data from all SET50 Excel files"""
    
    base_path = r"D:\CatBoost\SET50"
    stock_symbols = ['AOT', 'ADVANC', 'BANPU', 'BBL', 'BDMS', 'BEM', 'BH', 'BTS', 'CBG', 'CENTEL',
                     'COM_7', 'CPALL', 'CPF', 'CPN', 'DELTA', 'EA', 'EGCO', 'GLOBAL', 'GPSC', 'HMPRO',
                     'INTUCH', 'IVL', 'KBANK', 'KTB', 'KTC', 'LH', 'MINT', 'MTC', 'PTT', 'PTTEP',
                     'PTTGC', 'RATCH', 'SAWAD', 'SCC', 'TISCO', 'TOP', 'TTB', 'TU', 'WHA']
    
    print("=== EXTRACTING SET50 DATA ===\n")
    
    # Initialize arrays
    rmse_values = []
    sharpe_values = []
    successful_stocks = []
    failed_stocks = []
    
    for stock in stock_symbols:
        print(f"Processing {stock}...")
        
        # Construct file path
        file_path = os.path.join(base_path, stock, 'backtest_results_iter0.xlsx')
        
        if not os.path.exists(file_path):
            print(f"  ❌ File not found")
            failed_stocks.append(stock)
            rmse_values.append(np.nan)
            sharpe_values.append(np.nan)
            continue
        
        try:
            # Extract RMSE from ModelMetrics sheet
            df_model = pd.read_excel(file_path, sheet_name='ModelMetrics')
            # RMSE is in Row 1, Column 2 (0-indexed) - Out-of-Sample RMSE
            rmse_value = df_model.iloc[0, 1]  # Row 1, Column 2 (B2) - RMSE, not MSE!
            
            # Convert RMSE to numeric, handling any string values
            if isinstance(rmse_value, str):
                rmse_value = pd.to_numeric(rmse_value, errors='coerce')
            
            # Extract Sharpe ratio from Summary sheet
            df_summary = pd.read_excel(file_path, sheet_name='Summary')
            sharpe_value = df_summary.iloc[3, 1]  # Row 4, Column 2 (B4) - Sharpe Ratio
            
            # Convert Sharpe to numeric, handling any string values
            if isinstance(sharpe_value, str):
                sharpe_value = pd.to_numeric(sharpe_value, errors='coerce')
            
            print(f"  ✅ RMSE: {rmse_value}, Sharpe: {sharpe_value}")
            
            rmse_values.append(rmse_value)
            sharpe_values.append(sharpe_value)
            successful_stocks.append(stock)
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            failed_stocks.append(stock)
            rmse_values.append(np.nan)
            sharpe_values.append(np.nan)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Stock': stock_symbols,
        'RMSE': rmse_values,
        'Sharpe_Ratio': sharpe_values
    })
    
    # Summary
    print(f"\n=== EXTRACTION SUMMARY ===")
    print(f"Total stocks: {len(stock_symbols)}")
    print(f"Successful: {len(successful_stocks)}")
    print(f"Failed: {len(failed_stocks)}")
    
    if failed_stocks:
        print(f"Failed stocks: {', '.join(failed_stocks)}")
    
    # Save results to CSV
    results_df.to_csv('SET50_analysis_results_python.csv', index=False)
    print(f"\nResults saved to: SET50_analysis_results_python.csv")
    
    # Save results to Excel file
    try:
        with pd.ExcelWriter('sharpe_ratio.xlsx', engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='SET50_Analysis', index=False)
            
            # Get the workbook and worksheet to format
            workbook = writer.book
            worksheet = writer.sheets['SET50_Analysis']
            
            # Auto-adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"Results saved to: sharpe_ratio.xlsx")
        print(f"Excel file contains: Stock Name, RMSE, Sharpe Ratio columns")
        
    except Exception as e:
        print(f"Warning: Could not save to Excel: {str(e)}")
        print(f"CSV file is still available: SET50_analysis_results_python.csv")
    
    # Display results table
    print(f"\n=== RESULTS TABLE ===")
    print(results_df.to_string(index=False))
    
    # Create plots
    create_plots(results_df, successful_stocks)
    
    return results_df

def create_plots(results_df, successful_stocks):
    """Create 4 separate scientific-quality plots in black and white for publication"""
    
    # Filter out NaN values
    valid_data = results_df.dropna()
    
    if len(valid_data) == 0:
        print("No valid data to plot!")
        return
    
    # Set scientific plotting style
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['xtick.major.width'] = 1.0
    plt.rcParams['ytick.major.width'] = 1.0
    plt.rcParams['xtick.minor.width'] = 0.5
    plt.rcParams['ytick.minor.width'] = 0.5
    
    # Plot 1: RMSE Values by Stock
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    x_pos = np.arange(len(valid_data))
    bars1 = ax1.bar(x_pos, valid_data['RMSE'], color='white', edgecolor='black', linewidth=1.0, alpha=0.9)
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(valid_data['Stock'], rotation=90, fontsize=9)
    ax1.set_title('RMSE Values by Stock', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Stock Symbol', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add value labels on bars
    for bar, value in zip(bars1, valid_data['RMSE']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(valid_data['RMSE'])*0.02,
                f'{value:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('SET50_RMSE_Values.jpg', dpi=300, bbox_inches='tight', facecolor='white')
    print("Plot 1 saved: SET50_RMSE_Values.jpg")
    plt.close()
    
    # Plot 2: Sharpe Ratio Values by Stock
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    bars2 = ax2.bar(x_pos, valid_data['Sharpe_Ratio'], color='white', edgecolor='black', linewidth=1.0, alpha=0.9)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(valid_data['Stock'], rotation=90, fontsize=9)
    ax2.set_title('Sharpe Ratio Values by Stock', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Stock Symbol', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add value labels on bars
    for bar, value in zip(bars2, valid_data['Sharpe_Ratio']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(valid_data['Sharpe_Ratio'])*0.02,
                f'{value:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('SET50_Sharpe_Ratio_Values.jpg', dpi=300, bbox_inches='tight', facecolor='white')
    print("Plot 2 saved: SET50_Sharpe_Ratio_Values.jpg")
    plt.close()
    
    # Plot 3: Scatter Plot RMSE vs Sharpe Ratio
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    ax3.scatter(valid_data['RMSE'], valid_data['Sharpe_Ratio'], s=100, color='white', 
                edgecolor='black', linewidth=1.5, alpha=0.9, zorder=3)
    
    # Add stock labels
    for i, stock in enumerate(valid_data['Stock']):
        ax3.annotate(stock, (valid_data['RMSE'].iloc[i], valid_data['Sharpe_Ratio'].iloc[i]), 
                     fontsize=8, xytext=(8, 8), textcoords='offset points', 
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.8))
    
    ax3.set_xlabel('RMSE', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax3.set_title('RMSE vs Sharpe Ratio Relationship', fontsize=14, fontweight='bold', pad=20)
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('SET50_RMSE_vs_Sharpe_Scatter.jpg', dpi=300, bbox_inches='tight', facecolor='white')
    print("Plot 3 saved: SET50_RMSE_vs_Sharpe_Scatter.jpg")
    plt.close()
    
    # Plot 4: Combined Bar Plot
    fig4, ax4 = plt.subplots(figsize=(14, 8))
    width = 0.35
    
    bars3 = ax4.bar(x_pos - width/2, valid_data['RMSE'], width, label='RMSE', 
                     color='white', edgecolor='black', linewidth=1.0, alpha=0.9)
    bars4 = ax4.bar(x_pos + width/2, valid_data['Sharpe_Ratio'], width, label='Sharpe Ratio', 
                     color='lightgray', edgecolor='black', linewidth=1.0, alpha=0.9)
    
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(valid_data['Stock'], rotation=90, fontsize=9)
    ax4.set_title('RMSE and Sharpe Ratio Comparison', fontsize=14, fontweight='bold', pad=20)
    ax4.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Stock Symbol', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=11, frameon=True, fancybox=False, edgecolor='black')
    ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('SET50_RMSE_Sharpe_Comparison.jpg', dpi=300, bbox_inches='tight', facecolor='white')
    print("Plot 4 saved: SET50_RMSE_Sharpe_Comparison.jpg")
    plt.close()
    
    print("\n=== ALL PLOTS SAVED ===")
    print("1. SET50_RMSE_Values.jpg - RMSE values by stock")
    print("2. SET50_Sharpe_Ratio_Values.jpg - Sharpe ratio values by stock")
    print("3. SET50_RMSE_vs_Sharpe_Scatter.jpg - Scatter plot relationship")
    print("4. SET50_RMSE_Sharpe_Comparison.jpg - Combined comparison")
    print("\nAll plots are in scientific black and white format suitable for publication!")

if __name__ == "__main__":
    results = extract_set50_data()
