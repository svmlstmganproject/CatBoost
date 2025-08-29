% Updated MATLAB program to run Python scripts with different window sizes
% and plot all results with averages

clear all;
close all;
clc;

% Define window sizes to test
window_sizes = [5, 10, 30];
num_window_sizes = length(window_sizes);

% Initialize storage for all stocks and window sizes
all_stocks_data = struct();

% List of all 39 stocks
stock_names = {'ADVANC', 'AOT', 'BANPU', 'BBL', 'BEM', 'BDMS', 'BH', 'BTS', 'CBG', 'CENTEL', ...
               'COM_7', 'CPALL', 'CPF', 'CPN', 'DELTA', 'EA', 'EGCO', 'GLOBAL', 'GPSC', 'HMPRO', ...
               'INTUCH', 'IVL', 'KBANK', 'KTC', 'KTB', 'LH', 'MTC', 'MINT', 'PTTGC', 'PTT', ...
               'PTTEP', 'RATCH', 'SAWAD', 'SCC', 'TTB', 'TISCO', 'TOP', 'TU', 'WHA'};

num_stocks = length(stock_names);

fprintf('Starting analysis for %d stocks with %d window sizes...\n', num_stocks, num_window_sizes);

% Process each stock with different window sizes
for stock_idx = 1:num_stocks
    stock_name = stock_names{stock_idx};
    fprintf('\n=== Processing %s (%d/%d) ===\n', stock_name, stock_idx, num_stocks);
    
    % Initialize storage for this stock
    all_stocks_data.(stock_name) = struct();
    
    for ws_idx = 1:num_window_sizes
        window_size = window_sizes(ws_idx);
        fprintf('  Window size %d...\n', window_size);
        
        % Execute Python script with specific window size
        fprintf('    Executing Python script for %s with window size %d...\n', stock_name, window_size);
        cd(sprintf('D:\\CatBoost\\SET50\\%s', stock_name));
        
        % Run Python script with window size parameter
        python_cmd = sprintf('python test_cat3.py --window_size %d', window_size);
        [status, result] = system(python_cmd);
        
        if status == 0
            fprintf('    ✓ Python script executed successfully for %s (window size %d)\n', stock_name, window_size);
        else
            fprintf('    ✗ Error executing Python script for %s (window size %d): %s\n', stock_name, window_size, result);
        end
        
        % Return to original directory
        cd('D:\VDI_machine\back_up_VDI\catBoost_set50\');
        
        % Read the generated Excel file
        excel_file = sprintf('D:\\CatBoost\\SET50\\%s\\backtest_results_iter0.xlsx', stock_name);
        if exist(excel_file, 'file')
            try
                [~, ~, stock_data] = xlsread(excel_file, 'BacktestData');
                
                % Check if data was loaded successfully
                if isempty(stock_data) || size(stock_data, 1) < 2
                    fprintf('    ✗ Excel file %s is empty or has insufficient data\n', excel_file);
                    continue;
                end
                
                % Find the column index for 'cum_strategy_return'
                headers = stock_data(1, :);
                cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
                
                if ~isempty(cum_strategy_return_idx)
                    % Extract the data from the cum_strategy_return column (skip header row)
                    try
                        strategy_returns = cell2mat(stock_data(2:end, cum_strategy_return_idx));
                        
                        % Check if the data is valid
                        if ~isempty(strategy_returns) && all(isfinite(strategy_returns))
                            % Store the data
                            all_stocks_data.(stock_name).(sprintf('window_%d', window_size)) = strategy_returns;
                            fprintf('    ✓ Data loaded successfully for %s (window size %d)\n', stock_name, window_size);
                        else
                            fprintf('    ✗ Invalid data in %s (contains NaN or Inf)\n', excel_file);
                        end
                    catch ME
                        fprintf('    ✗ Error converting data to matrix: %s\n', ME.message);
                    end
                else
                    fprintf('    ✗ Column "cum_strategy_return" not found in %s\n', excel_file);
                end
            catch ME
                fprintf('    ✗ Error reading Excel file %s: %s\n', excel_file, ME.message);
            end
        else
            fprintf('    ✗ Excel file not found: %s\n', excel_file);
        end
    end
end

fprintf('\n=== All data loaded successfully! ===\n');

% Check if we have enough data to create plots
data_available = false;
for stock_idx = 1:num_stocks
    stock_name = stock_names{stock_idx};
    for ws_idx = 1:num_window_sizes
        window_size = window_sizes(ws_idx);
        if isfield(all_stocks_data.(stock_name), sprintf('window_%d', window_size))
            if ~isempty(all_stocks_data.(stock_name).(sprintf('window_%d', window_size)))
                data_available = true;
                break;
            end
        end
    end
    if data_available
        break;
    end
end

if ~data_available
    fprintf('No valid data available for plotting. Exiting.\n');
    return;
end

% Create comprehensive plots
try
    figure('Position', [100, 100, 1400, 1000]);

    % Plot 1: Individual stock performance for each window size
    subplot(2, 2, 1);
    hold on;
    colors = {'b', 'r', 'g'};
    legend_entries = {};

for ws_idx = 1:num_window_sizes
    window_size = window_sizes(ws_idx);
    color = colors{ws_idx};
    
    % Calculate average performance across all stocks for this window size
    all_returns = [];
    for stock_idx = 1:num_stocks
        stock_name = stock_names{stock_idx};
        if isfield(all_stocks_data.(stock_name), sprintf('window_%d', window_size))
            returns = all_stocks_data.(stock_name).(sprintf('window_%d', window_size));
            if ~isempty(returns)
                all_returns = [all_returns; returns];
            end
        end
    end
    
    if ~isempty(all_returns) && size(all_returns, 1) > 0
        % Calculate average across stocks
        try
            avg_returns = mean(all_returns, 1);
            if ~isempty(avg_returns) && all(isfinite(avg_returns))
                plot(avg_returns, color, 'LineWidth', 2);
                legend_entries{end+1} = sprintf('Window %d (Avg)', window_size);
            end
        catch ME
            fprintf('Error plotting window size %d: %s\n', window_size, ME.message);
        end
    end
end

title('Average Strategy Returns by Window Size (All Stocks)', 'FontSize', 14);
xlabel('Time Period', 'FontSize', 12);
ylabel('Cumulative Strategy Return', 'FontSize', 12);
legend(legend_entries, 'Location', 'best');
grid on;

% Plot 2: Individual stock comparison for window size 30 (original)
subplot(2, 2, 2);
hold on;
% Plot individual stocks with light gray color
for stock_idx = 1:num_stocks
    stock_name = stock_names{stock_idx};
    if isfield(all_stocks_data.(stock_name), 'window_30')
        returns = all_stocks_data.(stock_name).window_30;
        if ~isempty(returns) && all(isfinite(returns))
            try
                % Use a light gray color for individual stock lines
                plot(returns, 'Color', [0.8, 0.8, 0.8], 'LineWidth', 0.5);
            catch ME
                fprintf('Error plotting individual stock %s: %s\n', stock_name, ME.message);
            end
        end
    end
end

% Plot average line
all_window30_returns = [];
for stock_idx = 1:num_stocks
    stock_name = stock_names{stock_idx};
    if isfield(all_stocks_data.(stock_name), 'window_30')
        returns = all_stocks_data.(stock_name).window_30;
        if ~isempty(returns)
            all_window30_returns = [all_window30_returns; returns];
        end
    end
end

if ~isempty(all_window30_returns) && size(all_window30_returns, 1) > 0
    try
        avg_window30 = mean(all_window30_returns, 1);
        if ~isempty(avg_window30) && all(isfinite(avg_window30))
            plot(avg_window30, 'k', 'LineWidth', 3);
        end
    catch ME
        fprintf('Error plotting average line: %s\n', ME.message);
    end
end

title('Individual Stock Performance (Window Size 30)', 'FontSize', 14);
xlabel('Time Period', 'FontSize', 12);
ylabel('Cumulative Strategy Return', 'FontSize', 12);
legend('Individual Stocks', 'Average (All Stocks)', 'Location', 'best');
grid on;

% Plot 3: Window size comparison for average performance
subplot(2, 2, 3);
hold on;
window_performance = zeros(num_window_sizes, 1);
window_labels = {};

for ws_idx = 1:num_window_sizes
    window_size = window_sizes(ws_idx);
    
    % Calculate final performance for this window size
    all_final_returns = [];
    for stock_idx = 1:num_stocks
        stock_name = stock_names{stock_idx};
        if isfield(all_stocks_data.(stock_name), sprintf('window_%d', window_size))
            returns = all_stocks_data.(stock_name).(sprintf('window_%d', window_size));
            if ~isempty(returns)
                final_return = returns(end);
                all_final_returns = [all_final_returns; final_return];
            end
        end
    end
    
    if ~isempty(all_final_returns)
        avg_final_return = mean(all_final_returns);
        window_performance(ws_idx) = avg_final_return;
        window_labels{ws_idx} = sprintf('Window %d', window_size);
    end
end

bar(window_performance, 'FaceColor', [0.3, 0.6, 0.9]);
set(gca, 'XTickLabel', window_labels);
title('Average Final Performance by Window Size', 'FontSize', 14);
ylabel('Final Strategy Return', 'FontSize', 12);
grid on;

% Plot 4: Performance distribution across stocks
subplot(2, 2, 4);
hold on;

% Collect final returns for window size 30 (most common)
final_returns_window30 = [];
for stock_idx = 1:num_stocks
    stock_name = stock_names{stock_idx};
    if isfield(all_stocks_data.(stock_name), 'window_30')
        returns = all_stocks_data.(stock_name).window_30;
        if ~isempty(returns)
            final_returns_window30 = [final_returns_window30; returns(end)];
        end
    end
end

if ~isempty(final_returns_window30) && all(isfinite(final_returns_window30))
    try
        histogram(final_returns_window30, 15, 'FaceColor', [0.3, 0.6, 0.9], 'EdgeColor', 'k');
        title('Distribution of Final Returns (Window Size 30)', 'FontSize', 14);
        xlabel('Final Strategy Return', 'FontSize', 12);
        ylabel('Number of Stocks', 'FontSize', 12);
        
        % Add mean line
        mean_return = mean(final_returns_window30);
        ylims = ylim;
        line([mean_return, mean_return], ylims, 'Color', 'r', 'LineWidth', 2, 'LineStyle', '--');
        legend('Returns Distribution', sprintf('Mean: %.4f', mean_return), 'Location', 'best');
    catch ME
        fprintf('Error creating histogram: %s\n', ME.message);
    end
end

grid on;

% Overall title (use suptitle for older MATLAB versions, sgtitle for newer)
try
    sgtitle('SET50 CatBoost Strategy Analysis - Multi-Window Comparison', 'FontSize', 16, 'FontWeight', 'bold');
catch
    % Fallback for older MATLAB versions
    suptitle('SET50 CatBoost Strategy Analysis - Multi-Window Comparison');
end

    % Save the comprehensive plot
    saveas(gcf, 'SET50_comprehensive_analysis.png', 'png');
    saveas(gcf, 'SET50_comprehensive_analysis.fig', 'fig');
    
catch ME
    fprintf('Error creating plots: %s\n', ME.message);
    fprintf('Continuing with data analysis...\n');
end

fprintf('\n=== Analysis Complete! ===\n');
fprintf('Comprehensive plot saved as: SET50_comprehensive_analysis.png\n');
fprintf('Data structure "all_stocks_data" contains all results for further analysis\n');

% Display summary statistics
fprintf('\n=== Summary Statistics ===\n');
for ws_idx = 1:num_window_sizes
    window_size = window_sizes(ws_idx);
    fprintf('\nWindow Size %d:\n', window_size);
    
    % Calculate statistics for this window size
    all_final_returns = [];
    for stock_idx = 1:num_stocks
        stock_name = stock_names{stock_idx};
        if isfield(all_stocks_data.(stock_name), sprintf('window_%d', window_size))
            returns = all_stocks_data.(stock_name).(sprintf('window_%d', window_size));
            if ~isempty(returns)
                final_return = returns(end);
                all_final_returns = [all_final_returns; final_return];
            end
        end
    end
    
    if ~isempty(all_final_returns)
        fprintf('  Number of stocks: %d\n', length(all_final_returns));
        fprintf('  Average final return: %.4f\n', mean(all_final_returns));
        fprintf('  Median final return: %.4f\n', median(all_final_returns));
        fprintf('  Standard deviation: %.4f\n', std(all_final_returns));
        fprintf('  Min return: %.4f\n', min(all_final_returns));
        fprintf('  Max return: %.4f\n', max(all_final_returns));
    else
        fprintf('  No data available\n');
    end
end
