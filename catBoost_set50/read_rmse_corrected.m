% Change to the current directory
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\')

% Define the base path and stock symbols (each stock has its own subdirectory)
base_path = 'D:\CatBoost\SET50\';
stock_symbols = {'AOT', 'ADVANC', 'BANPU', 'BBL', 'BDMS', 'BEM', 'BH', 'BTS', 'CBG', 'CENTEL', ...
                 'COM_7', 'CPALL', 'CPF', 'CPN', 'DELTA', 'EA', 'EGCO', 'GLOBAL', 'GPSC', 'HMPRO', ...
                 'INTUCH', 'IVL', 'KBANK', 'KTB', 'KTC', 'LH', 'MINT', 'MTC', 'PTT', 'PTTEP', ...
                 'PTTGC', 'RATCH', 'SAWAD', 'SCC', 'TISCO', 'TOP', 'TTB', 'TU', 'WHA'};

% Initialize arrays to store results
num_stocks = length(stock_symbols);
rmse_values = zeros(num_stocks, 1);
sharpe_values = zeros(num_stocks, 1);
successful_stocks = cell(num_stocks, 1);
failed_stocks = cell(0, 1);

fprintf('Starting to read Excel files for %d stocks...\n', num_stocks);
fprintf('Base path: %s\n\n', base_path);

% Loop through each stock
for i = 1:num_stocks
    stock = stock_symbols{i};
    fprintf('Processing stock %d/%d: %s\n', i, num_stocks, stock);
    
    try
        % Construct file path - each stock has its own subdirectory
        file_path = fullfile(base_path, stock, 'backtest_results_iter0.xlsx');
        
        % Check if file exists
        if ~exist(file_path, 'file')
            fprintf('  Warning: File not found: %s\n', file_path);
            failed_stocks{end+1} = stock;
            continue;
        end
        
        % Try to read ModelMetrics sheet for RMSE
        try
            % Read from ModelMetrics sheet
            [~, ~, model_data] = xlsread(file_path, 'ModelMetrics');
            
            if isempty(model_data)
                fprintf('  Warning: ModelMetrics sheet is empty\n');
                rmse_values(i) = NaN;
            else
                fprintf('  Successfully read ModelMetrics sheet\n');
                
                % Extract RMSE from cell B2 (Row 2, Column 2 - Out-of-Sample RMSE)
                % Based on Python debugging: Row 2, Column 2 (0-indexed: row 1, column 1)
                if size(model_data, 1) >= 2 && size(model_data, 2) >= 2
                    % Get the Out-of-Sample RMSE value (row 2, column 2)
                    rmse_cell_value = model_data{2, 2}; % Row 2, Column 2 (B2) - RMSE, not MSE!
                    
                    if isnumeric(rmse_cell_value) && ~isnan(rmse_cell_value)
                        rmse_values(i) = rmse_cell_value;
                        fprintf('  RMSE value from cell B2 (Out-of-Sample): %.6f\n', rmse_values(i));
                    else
                        fprintf('  Warning: RMSE value in cell B2 is not numeric: %s\n', num2str(rmse_cell_value));
                        rmse_values(i) = NaN;
                    end
                else
                    fprintf('  Warning: ModelMetrics sheet does not have expected structure\n');
                    fprintf('  Sheet size: %dx%d, expected at least 2x2\n', size(model_data, 1), size(model_data, 2));
                    rmse_values(i) = NaN;
                end
            end
        catch ME
            fprintf('  Error reading ModelMetrics sheet: %s\n', ME.message);
            rmse_values(i) = NaN;
        end
        
        % Try to read Summary sheet for Sharpe ratio
        try
            % Read from Summary sheet
            [~, ~, summary_data] = xlsread(file_path, 'Summary');
            
            if isempty(summary_data)
                fprintf('  Warning: Summary sheet is empty\n');
                sharpe_values(i) = NaN;
            else
                fprintf('  Successfully read Summary sheet\n');
                
                % Extract Sharpe ratio from cell B4 (Row 4, Column 2 - NOT Row 5!)
                % Based on Python debugging: Row 4, Column 2 (0-indexed: row 3, column 1)
                if size(summary_data, 1) >= 4 && size(summary_data, 2) >= 2
                    % Get the Sharpe ratio value (row 4, column 2)
                    sharpe_cell_value = summary_data{4, 2}; % Row 4, Column 2 (B4)
                    
                    if isnumeric(sharpe_cell_value) && ~isnan(sharpe_cell_value)
                        sharpe_values(i) = sharpe_cell_value;
                        fprintf('  Sharpe ratio from cell B4: %.6f\n', sharpe_values(i));
                    else
                        fprintf('  Warning: Sharpe ratio value in cell B4 is not numeric: %s\n', num2str(sharpe_cell_value));
                        sharpe_values(i) = NaN;
                    end
                else
                    fprintf('  Warning: Summary sheet does not have expected structure\n');
                    fprintf('  Sheet size: %dx%d, expected at least 4x2\n', size(summary_data, 1), size(summary_data, 2));
                    sharpe_values(i) = NaN;
                end
            end
        catch ME
            fprintf('  Error reading Summary sheet: %s\n', ME.message);
            sharpe_values(i) = NaN;
        end
        
        % Mark as successful if we got at least one value
        if ~isnan(rmse_values(i)) || ~isnan(sharpe_values(i))
            successful_stocks{i} = stock;
            fprintf('  Successfully processed %s\n', stock);
        else
            failed_stocks{end+1} = stock;
            fprintf('  Failed to process %s\n', stock);
        end
        
    catch ME
        fprintf('  Error processing %s: %s\n', stock, ME.message);
        failed_stocks{end+1} = stock;
        rmse_values(i) = NaN;
        sharpe_values(i) = NaN;
    end
    
    fprintf('\n');
end

% Display summary of results
fprintf('=== PROCESSING SUMMARY ===\n');
fprintf('Total stocks: %d\n', num_stocks);
fprintf('Successful: %d\n', sum(~cellfun(@isempty, successful_stocks)));
fprintf('Failed: %d\n', length(failed_stocks));

if ~isempty(failed_stocks)
    fprintf('Failed stocks: %s\n', strjoin(failed_stocks, ', '));
end

% Create results table
results_table = table(stock_symbols', rmse_values, sharpe_values, ...
                     'VariableNames', {'Stock', 'RMSE', 'Sharpe_Ratio'});

% Display results table
fprintf('\n=== RESULTS TABLE ===\n');
disp(results_table);

% Save results to workspace variables
assignin('base', 'stock_symbols', stock_symbols);
assignin('base', 'rmse_values', rmse_values);
assignin('base', 'sharpe_values', sharpe_values);
assignin('base', 'results_table', results_table);
assignin('base', 'successful_stocks', successful_stocks);
assignin('base', 'failed_stocks', failed_stocks);

% Create plots
fprintf('\n=== CREATING PLOTS ===\n');

% Filter out NaN values for plotting
valid_indices = ~isnan(rmse_values) & ~isnan(sharpe_values);
valid_stocks = stock_symbols(valid_indices);
valid_rmse = rmse_values(valid_indices);
valid_sharpe = sharpe_values(valid_indices);

if sum(valid_indices) > 0
    % Create figure with subplots
    figure('Position', [100, 100, 1200, 800]);
    
    % Plot 1: RMSE values
    subplot(2, 2, 1);
    bar(valid_rmse);
    set(gca, 'XTick', 1:length(valid_stocks), 'XTickLabel', valid_stocks);
    set(gca, 'XTickLabelRotation', 90, 'FontSize', 7);
    title('RMSE Values by Stock');
    ylabel('RMSE');
    grid on;
    
    % Plot 2: Sharpe ratio values
    subplot(2, 2, 2);
    bar(valid_sharpe);
    set(gca, 'XTick', 1:length(valid_stocks), 'XTickLabel', valid_stocks);
    set(gca, 'XTickLabelRotation', 90, 'FontSize', 7);
    title('Sharpe Ratio Values by Stock');
    ylabel('Sharpe Ratio');
    grid on;
    
    % Plot 3: Scatter plot of RMSE vs Sharpe
    subplot(2, 2, 3);
    scatter(valid_rmse, valid_sharpe, 50, 'filled');
    for j = 1:length(valid_stocks)
        text(valid_rmse(j), valid_sharpe(j), valid_stocks{j}, 'FontSize', 6);
    end
    xlabel('RMSE');
    ylabel('Sharpe Ratio');
    title('RMSE vs Sharpe Ratio');
    grid on;
    
    % Plot 4: Combined bar plot
    subplot(2, 2, 4);
    x_pos = 1:length(valid_stocks);
    bar(x_pos, [valid_rmse, valid_sharpe]);
    set(gca, 'XTick', x_pos, 'XTickLabel', valid_stocks);
    set(gca, 'XTickLabelRotation', 90, 'FontSize', 7);
    title('RMSE and Sharpe Ratio Comparison');
    ylabel('Value');
    legend('RMSE', 'Sharpe Ratio', 'Location', 'best');
    grid on;
    
    % Adjust layout
    sgtitle('SET50 Stocks Analysis - RMSE and Sharpe Ratio', 'FontSize', 16);
    
    fprintf('Plots created successfully!\n');
    fprintf('Valid data points: %d out of %d stocks\n', sum(valid_indices), num_stocks);
else
    fprintf('No valid data to plot!\n');
end

% Save results to file
try
    writetable(results_table, 'SET50_analysis_results_matlab.csv');
    fprintf('Results saved to SET50_analysis_results_matlab.csv\n');
catch ME
    fprintf('Warning: Could not save to CSV: %s\n', ME.message);
end

fprintf('\n=== ANALYSIS COMPLETE ===\n');
fprintf('Variables saved to workspace:\n');
fprintf('- stock_symbols: Stock symbols\n');
fprintf('- rmse_values: RMSE values array (from Excel files)\n');
fprintf('- sharpe_values: Sharpe ratio values array (from Excel files)\n');
fprintf('- results_table: Complete results table\n');
fprintf('- successful_stocks: Successfully processed stocks\n');
fprintf('- failed_stocks: Failed stocks\n');
