import pandas as pd
import os
import glob
from pathlib import Path

def scan_excel_files():
    """Scan all Excel files in SET50 directories to understand their structure"""
    
    base_path = r"D:\CatBoost\SET50"
    stock_symbols = ['AOT', 'ADVANC', 'BANPU', 'BBL', 'BDMS', 'BEM', 'BH', 'BTS', 'CBG', 'CENTEL',
                     'COM_7', 'CPALL', 'CPF', 'CPN', 'DELTA', 'EA', 'EGCO', 'GLOBAL', 'GPSC', 'HMPRO',
                     'INTUCH', 'IVL', 'KBANK', 'KTB', 'KTC', 'LH', 'MINT', 'MTC', 'PTT', 'PTTEP',
                     'PTTGC', 'RATCH', 'SAWAD', 'SCC', 'TISCO', 'TOP', 'TTB', 'TU', 'WHA']
    
    print("=== SCANNING EXCEL FILES IN SET50 DIRECTORIES ===\n")
    
    results = {}
    
    for stock in stock_symbols:
        print(f"Processing {stock}...")
        
        # Construct file path
        file_path = os.path.join(base_path, stock, 'backtest_results_iter0.xlsx')
        
        if not os.path.exists(file_path):
            print(f"  ❌ File not found: {file_path}")
            results[stock] = {'status': 'file_not_found', 'path': file_path}
            continue
        
        try:
            # Get sheet names
            excel_file = pd.ExcelFile(file_path)
            sheets = excel_file.sheet_names
            print(f"  ✅ File found. Sheets: {sheets}")
            
            stock_data = {'status': 'success', 'path': file_path, 'sheets': sheets, 'data': {}}
            
            # Try to read each sheet
            for sheet_name in sheets:
                try:
                    if sheet_name == 'ModelMetrics':
                        df = pd.read_excel(file_path, sheet_name=sheet_name)
                        print(f"    📊 ModelMetrics sheet: {df.shape[0]} rows x {df.shape[1]} columns")
                        print(f"      Headers: {list(df.columns)}")
                        print(f"      First few rows:")
                        print(df.head(3).to_string())
                        
                        # Check if we can find RMSE data
                        if df.shape[1] >= 3:  # At least 3 columns
                            # Look for RMSE in row 2, column 3 (0-indexed: row 1, column 2)
                            if df.shape[0] >= 2:
                                rmse_value = df.iloc[1, 2]  # Row 2, Column 3 (C2)
                                print(f"      RMSE value at C2: {rmse_value}")
                                stock_data['data']['rmse'] = rmse_value
                            else:
                                print(f"      ⚠️ Not enough rows for RMSE extraction")
                        else:
                            print(f"      ⚠️ Not enough columns for RMSE extraction")
                            
                    elif sheet_name == 'Summary':
                        df = pd.read_excel(file_path, sheet_name=sheet_name)
                        print(f"    📊 Summary sheet: {df.shape[0]} rows x {df.shape[1]} columns")
                        print(f"      Headers: {list(df.columns)}")
                        print(f"      First few rows:")
                        print(df.head(3).to_string())
                        
                        # Check if we can find Sharpe ratio data
                        if df.shape[1] >= 2 and df.shape[0] >= 5:  # At least 2 columns and 5 rows
                            # Look for Sharpe ratio in row 5, column 2 (0-indexed: row 4, column 1)
                            sharpe_value = df.iloc[4, 1]  # Row 5, Column 2 (B5)
                            print(f"      Sharpe ratio at B5: {sharpe_value}")
                            stock_data['data']['sharpe'] = sharpe_value
                        else:
                            print(f"      ⚠️ Not enough rows/columns for Sharpe extraction")
                            
                    else:
                        df = pd.read_excel(file_path, sheet_name=sheet_name)
                        print(f"    📊 {sheet_name} sheet: {df.shape[0]} rows x {df.shape[1]} columns")
                        
                except Exception as e:
                    print(f"    ❌ Error reading {sheet_name} sheet: {str(e)}")
                    stock_data['data'][sheet_name] = f"Error: {str(e)}"
            
            results[stock] = stock_data
            print()
            
        except Exception as e:
            print(f"  ❌ Error processing file: {str(e)}")
            results[stock] = {'status': 'error', 'path': file_path, 'error': str(e)}
            print()
    
    # Summary
    print("=== SCANNING SUMMARY ===")
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    failed = len(results) - successful
    
    print(f"Total stocks: {len(stock_symbols)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print(f"\nFailed stocks:")
        for stock, data in results.items():
            if data['status'] != 'success':
                print(f"  {stock}: {data['status']}")
    
    return results

if __name__ == "__main__":
    results = scan_excel_files()
