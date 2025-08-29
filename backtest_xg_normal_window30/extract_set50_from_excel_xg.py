import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_sharpe_ratio_data():
    """Read RMSE, Sharpe ratio, and Strategy Return data from sharpe_ratio.xlsx file"""
    
    try:
        # Read the Excel file
        df = pd.read_excel('sharpe_ratio_xg.xlsx', sheet_name='X')
        
        print("=== READING FROM SHARPE_RATIO.XLSX ===")
        print(f"Successfully loaded data with {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        
        # Display first few rows
        print(f"\n=== DATA PREVIEW ===")
        print(df.head().to_string(index=False))
        
        return df
        
    except FileNotFoundError:
        print("❌ Error: sharpe_ratio.xlsx file not found!")
        print("Please make sure the file exists in the current directory.")
        return None
    except Exception as e:
        print(f"❌ Error reading file: {str(e)}")
        return None

def create_plots(results_df):
    """Create 5 separate scientific-quality plots in black and white for publication"""
    
    # Filter out NaN values
    valid_data = results_df.dropna()
    
    if len(valid_data) == 0:
        print("No valid data to plot!")
        return
    
    print(f"\n=== CREATING PLOTS ===")
    print(f"Using {len(valid_data)} valid data points")
    
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
    
    # Plot 3: Strategy Return Values by Stock
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    bars3 = ax3.bar(x_pos, valid_data['Strategy_Return'], color='white', edgecolor='black', linewidth=1.0, alpha=0.9)
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(valid_data['Stock'], rotation=90, fontsize=9)
    ax3.set_title('Strategy Return Values by Stock', fontsize=14, fontweight='bold', pad=20)
    ax3.set_ylabel('Strategy Return (Decimal)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Stock Symbol', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Add value labels on bars (convert back to percentage for display)
    for bar, value in zip(bars3, valid_data['Strategy_Return']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(valid_data['Strategy_Return'])*0.02,
                f'{value*100:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('SET50_Strategy_Return_Values.jpg', dpi=300, bbox_inches='tight', facecolor='white')
    print("Plot 3 saved: SET50_Strategy_Return_Values.jpg")
    plt.close()
    
    # Plot 4: Strategy Return vs Sharpe Ratio Scatter Plot
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    ax4.scatter(valid_data['Strategy_Return'], valid_data['Sharpe_Ratio'], s=100, color='white', 
                edgecolor='black', linewidth=1.5, alpha=0.9, zorder=3)
    
    # Add stock labels
    for i, stock in enumerate(valid_data['Stock']):
        ax4.annotate(stock, (valid_data['Strategy_Return'].iloc[i], valid_data['Sharpe_Ratio'].iloc[i]), 
                     fontsize=8, xytext=(8, 8), textcoords='offset points', 
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.8))
    
    ax4.set_xlabel('Strategy Return (Decimal)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax4.set_title('Strategy Return vs Sharpe Ratio Relationship', fontsize=14, fontweight='bold', pad=20)
    ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('SET50_Strategy_Return_vs_Sharpe_Scatter.jpg', dpi=300, bbox_inches='tight', facecolor='white')
    print("Plot 4 saved: SET50_Strategy_Return_vs_Sharpe_Scatter.jpg")
    plt.close()
    
    # Plot 5: Combined Bar Plot (RMSE, Sharpe, Strategy Return)
    fig5, ax5 = plt.subplots(figsize=(16, 8))
    width = 0.25
    
    bars4 = ax5.bar(x_pos - width, valid_data['RMSE'], width, label='RMSE', 
                     color='white', edgecolor='black', linewidth=1.0, alpha=0.9)
    bars5 = ax5.bar(x_pos, valid_data['Sharpe_Ratio'], width, label='Sharpe Ratio', 
                     color='lightgray', edgecolor='black', linewidth=1.0, alpha=0.9)
    bars6 = ax5.bar(x_pos + width, valid_data['Strategy_Return'], width, label='Strategy Return', 
                     color='darkgray', edgecolor='black', linewidth=1.0, alpha=0.9)
    
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(valid_data['Stock'], rotation=90, fontsize=9)
    ax5.set_title('RMSE, Sharpe Ratio, and Strategy Return Comparison', fontsize=14, fontweight='bold', pad=20)
    ax5.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Stock Symbol', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=11, frameon=True, fancybox=False, edgecolor='black')
    ax5.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('SET50_Three_Metrics_Comparison.jpg', dpi=300, bbox_inches='tight', facecolor='white')
    print("Plot 5 saved: SET50_Three_Metrics_Comparison.jpg")
    plt.close()
    
    print("\n=== ALL PLOTS SAVED ===")
    print("1. SET50_RMSE_Values.jpg - RMSE values by stock")
    print("2. SET50_Sharpe_Ratio_Values.jpg - Sharpe ratio values by stock")
    print("3. SET50_Strategy_Return_Values.jpg - Strategy return values by stock")
    print("4. SET50_Strategy_Return_vs_Sharpe_Scatter.jpg - Strategy return vs Sharpe ratio")
    print("5. SET50_Three_Metrics_Comparison.jpg - Combined comparison of all three metrics")
    print("\nAll plots are in scientific black and white format suitable for publication!")

if __name__ == "__main__":
    # Read data from the Excel file
    results = read_sharpe_ratio_data()
    
    if results is not None:
        # Create all the plots
        create_plots(results)
    else:
        print("Cannot proceed without data. Please check the sharpe_ratio.xlsx file.")
