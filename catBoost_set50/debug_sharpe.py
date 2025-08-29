import pandas as pd
import os

def debug_sharpe_ratio():
    """Debug script to find Sharpe ratio values in Summary sheets"""
    
    base_path = r"D:\CatBoost\SET50"
    
    # Check a few stocks to understand the Summary sheet structure
    test_stocks = ['AOT', 'ADVANC', 'BANPU']
    
    for stock in test_stocks:
        print(f"\n=== DEBUGGING {stock} ===")
        file_path = os.path.join(base_path, stock, 'backtest_results_iter0.xlsx')
        
        try:
            # Read Summary sheet
            df_summary = pd.read_excel(file_path, sheet_name='Summary')
            print(f"Summary sheet shape: {df_summary.shape}")
            print(f"Headers: {list(df_summary.columns)}")
            print("\nFull Summary sheet:")
            print(df_summary.to_string(index=False))
            
            # Check what's in row 5, column 2 (B5)
            if df_summary.shape[0] >= 5 and df_summary.shape[1] >= 2:
                b5_value = df_summary.iloc[4, 1]  # Row 5, Column 2 (B5)
                print(f"\nValue at B5 (row 5, col 2): {b5_value} (type: {type(b5_value)})")
            
            # Check all values in the Summary sheet
            print(f"\nAll values in Summary sheet:")
            for i in range(df_summary.shape[0]):
                for j in range(df_summary.shape[1]):
                    value = df_summary.iloc[i, j]
                    print(f"  Row {i+1}, Col {j+1}: {value} (type: {type(value)})")
                    
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    debug_sharpe_ratio()
