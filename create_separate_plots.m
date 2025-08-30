% MATLAB Script to create separate 3D Surface and Contour plots from sharpe_ratio.xlsx
% This script reads the Excel file and creates separate 3D surface and contour plots

clear; clc; close all;

fprintf('=== CREATING SEPARATE 3D SURFACE AND CONTOUR PLOTS FROM SHARPE_RATIO.XLSX ===\n\n');

% Read data from the Excel file
try
    fprintf('Reading data from sharpe_ratio_xg.xlsx...\n');
    data = readtable('sharpe_ratio_xg.xlsx');
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

% ===== PLOT 1: 3D SURFACE PLOT =====
figure('Position', [100, 100, 1000, 800]);

% Main 3D surface plot
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
xlabel('predictive RMSE', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Sharpe Ratio', 'FontSize', 12, 'FontWeight', 'bold');
zlabel('Cumulative Return', 'FontSize', 12, 'FontWeight', 'bold');
title('3D Surface Plot', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
view(20, 30); % Set viewing angle
colorbar;
colormap('jet'); % Use RGB jet colormap instead of grayscale

% Save the 3D surface plot
try
    saveas(gcf, 'SET50_3D_Surface_Plot.jpg', 'jpg');
    fprintf('\n3D Surface plot saved as: SET50_3D_Surface_Plot.jpg\n');
catch ME
    fprintf('Warning: Could not save 3D surface plot: %s\n', ME.message);
end

% Close the first figure
close(gcf);

% ===== PLOT 2: CONTOUR PLOT =====
figure('Position', [100, 100, 1000, 800]);

% Create contour plot with RGB colors
[C, h] = contour(X, Y, Z, 20, 'LineWidth', 2);
hold on;

% Add the actual data points
scatter(rmse_clean, sharpe_clean, 80, 'filled', 'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'black', 'LineWidth', 1.5);

% Add stock labels to the scatter points
for i = 1:length(stock_names_clean)
    text(rmse_clean(i), sharpe_clean(i), stock_names_clean{i}, ...
         'FontSize', 8, 'FontWeight', 'bold', 'HorizontalAlignment', 'left', ...
         'VerticalAlignment', 'bottom');
end

% Customize the contour plot
xlabel('predictive RMSE', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Sharpe Ratio', 'FontSize', 12, 'FontWeight', 'bold');
title('  Predictive RMSE vs Sharpe Ratio vs Cumulative Return', 'FontSize', 14, 'FontWeight', 'bold');
grid on;

% Add colorbar and use RGB colormap
colorbar;
colormap('jet'); % Use RGB jet colormap

% Add contour labels
clabel(C, h, 'FontSize', 8, 'Color', 'black');

% Save the contour plot
try
    saveas(gcf, 'SET50_Contour_Plot.jpg', 'jpg');
    fprintf('Contour plot saved as: SET50_Contour_Plot.jpg\n');
catch ME
    fprintf('Warning: Could not save contour plot: %s\n', ME.message);
end

% Close the second figure
close(gcf);

% ===== STATISTICAL SUMMARY =====
% Display statistics in console
fprintf('\n=== STATISTICAL SUMMARY ===\n');
fprintf('RMSE: Mean=%.3f, Std=%.3f, Min=%.3f, Max=%.3f\n', ...
        mean(rmse_clean), std(rmse_clean), min(rmse_clean), max(rmse_clean));
fprintf('Sharpe Ratio: Mean=%.3f, Std=%.3f, Min=%.3f, Max=%.3f\n', ...
        mean(sharpe_clean), std(sharpe_clean), min(sharpe_clean), max(sharpe_clean));
fprintf('Strategy Return: Mean=%.4f, Std=%.4f, Min=%.4f, Max=%.4f\n', ...
        mean(strategy_return_clean), std(strategy_return_clean), min(strategy_return_clean), max(strategy_return_clean));
fprintf('Total Stocks: %d\n', length(rmse_clean));

fprintf('\n=== PLOT CREATION COMPLETE ===\n');
fprintf('Two separate plots have been created:\n');
fprintf('1. SET50_3D_Surface_Plot.jpg - 3D surface plot with RGB colors\n');
fprintf('2. SET50_Contour_Plot.jpg - Contour plot with RGB colors\n');
fprintf('\nThe plots show the relationship between:\n');
fprintf('- X-axis: RMSE (Root Mean Square Error)\n');
fprintf('- Y-axis: Sharpe Ratio\n');
fprintf('- Z-axis/Contours: Strategy Return (Decimal)\n');
fprintf('- Red dots: Actual stock data points\n');
fprintf('- Surface/Contours: Interpolated relationship\n');
