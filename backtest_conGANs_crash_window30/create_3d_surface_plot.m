% MATLAB Script to create 3D Surface Plot from sharpe_ratio.xlsx
% This script reads the Excel file and creates a 3D surface visualization

clear; clc; close all;

fprintf('=== CREATING 3D SURFACE PLOT FROM SHARPE_RATIO.XLSX ===\n\n');

% Read data from the Excel file
try
    fprintf('Reading data from sharpe_ratio.xlsx...\n');
    data = readtable('sharpe_ratio.xlsx');
    fprintf('Successfully read data with %d rows and %d columns\n', height(data), width(data));
    
    % Display column names
    fprintf('Columns found: %s\n', strjoin(data.Properties.VariableNames, ', '));
    
catch ME
    fprintf('Error reading Excel file: %s\n', ME.message);
    fprintf('Please ensure sharpe_ratio.xlsx exists in the current directory\n');
    return;
end

% Extract the three metrics
rmse = data.RMSE;
sharpe = data.Sharpe_Ratio;
strategy_return = data.Strategy_Return;
stock_names = data.Stock;

% Remove any NaN values
valid_indices = ~isnan(rmse) & ~isnan(sharpe) & ~isnan(strategy_return);
rmse_clean = rmse(valid_indices);
sharpe_clean = sharpe(valid_indices);
strategy_return_clean = strategy_return(valid_indices);
stock_names_clean = stock_names(valid_indices);

fprintf('Valid data points: %d out of %d stocks\n', sum(valid_indices), length(rmse));

% Create meshgrid for surface plotting
% We'll create a regular grid and interpolate the data
x_min = min(rmse_clean); x_max = max(rmse_clean);
y_min = min(sharpe_clean); y_max = max(sharpe_clean);

% Create grid points
x_grid = linspace(x_min, x_max, 50);
y_grid = linspace(y_min, y_max, 50);
[X, Y] = meshgrid(x_grid, y_grid);

% Interpolate the scattered data to create a surface
% Using scatteredInterpolant for better interpolation
F = scatteredInterpolant(rmse_clean, sharpe_clean, strategy_return_clean, 'linear', 'none');
Z = F(X, Y);

% Create the 3D surface plot
figure('Position', [100, 100, 1200, 800]);

% Main 3D surface plot
subplot(2, 2, [1, 3]);
surf(X, Y, Z, 'FaceAlpha', 0.8, 'EdgeColor', 'black', 'EdgeAlpha', 0.3);
hold on;

% Add the actual data points as 3D scatter plot
scatter3(rmse_clean, sharpe_clean, strategy_return_clean, 100, 'filled', 'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'black');

% Add stock labels to the scatter points
for i = 1:length(stock_names_clean)
    text(rmse_clean(i), sharpe_clean(i), strategy_return_clean(i), stock_names_clean{i}, ...
         'FontSize', 8, 'FontWeight', 'bold', 'HorizontalAlignment', 'left', ...
         'VerticalAlignment', 'bottom');
end

% Customize the plot
xlabel('RMSE', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Sharpe Ratio', 'FontSize', 12, 'FontWeight', 'bold');
zlabel('Strategy Return (Decimal)', 'FontSize', 12, 'FontWeight', 'bold');
title('3D Surface Plot: RMSE vs Sharpe Ratio vs Strategy Return', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
view(45, 30); % Set viewing angle
colorbar;
colormap('gray'); % Use grayscale for publication

% Add contour plot on the bottom
subplot(2, 2, 2);
contour(X, Y, Z, 15, 'LineColor', 'black', 'LineWidth', 1.5);
hold on;
scatter(rmse_clean, sharpe_clean, 50, 'filled', 'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'black');
xlabel('RMSE', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Sharpe Ratio', 'FontSize', 11, 'FontWeight', 'bold');
title('Contour Plot (Top View)', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

% Add statistics summary
subplot(2, 2, 4);
stats_data = [mean(rmse_clean), mean(sharpe_clean), mean(strategy_return_clean);
              std(rmse_clean), std(sharpe_clean), std(strategy_return_clean);
              min(rmse_clean), min(sharpe_clean), min(strategy_return_clean);
              max(rmse_clean), max(sharpe_clean), max(strategy_return_clean)];

stats_table = array2table(stats_data, 'VariableNames', {'RMSE', 'Sharpe_Ratio', 'Strategy_Return'}, ...
                         'RowNames', {'Mean', 'Std', 'Min', 'Max'});

% Display statistics
fprintf('\n=== STATISTICAL SUMMARY ===\n');
disp(stats_table);

% Create a simple text display for the subplot
text(0.1, 0.8, 'Statistical Summary:', 'FontSize', 12, 'FontWeight', 'bold', 'Units', 'normalized');
text(0.1, 0.7, sprintf('RMSE: Mean=%.3f, Std=%.3f', mean(rmse_clean), std(rmse_clean)), ...
     'FontSize', 10, 'Units', 'normalized');
text(0.1, 0.6, sprintf('Sharpe: Mean=%.3f, Std=%.3f', mean(sharpe_clean), std(sharpe_clean)), ...
     'FontSize', 10, 'Units', 'normalized');
text(0.1, 0.5, sprintf('Strategy Return: Mean=%.4f, Std=%.4f', mean(strategy_return_clean), std(strategy_return_clean)), ...
     'FontSize', 10, 'Units', 'normalized');
text(0.1, 0.4, sprintf('Total Stocks: %d', length(rmse_clean)), 'FontSize', 10, 'Units', 'normalized');

axis off; % Hide axes for the text display

% Adjust layout
sgtitle('SET50 Stocks: 3D Surface Analysis', 'FontSize', 16, 'FontWeight', 'bold');

% Save the plot
try
    saveas(gcf, 'SET50_3D_Surface_Plot.png', 'png');
    saveas(gcf, 'SET50_3D_Surface_Plot.fig', 'fig');
    fprintf('\n3D Surface plot saved as:\n');
    fprintf('- SET50_3D_Surface_Plot.png\n');
    fprintf('- SET50_3D_Surface_Plot.fig\n');
catch ME
    fprintf('Warning: Could not save plot: %s\n', ME.message);
end

fprintf('\n=== 3D SURFACE PLOT CREATION COMPLETE ===\n');
fprintf('The plot shows the relationship between:\n');
fprintf('- X-axis: RMSE (Root Mean Square Error)\n');
fprintf('- Y-axis: Sharpe Ratio\n');
fprintf('- Z-axis: Strategy Return (Decimal)\n');
fprintf('- Red dots: Actual stock data points\n');
fprintf('- Surface: Interpolated relationship\n');
