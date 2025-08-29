%load advanc_window5_crash.txt
%load advanc_window10_crash.txt

% Execute Python script for ADVANC
fprintf('Executing Python script for ADVANC...\n');
cd('D:\CatBoost_normal\window10\SET50\ADVANC');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for ADVANC\n');
else
    fprintf('Error executing Python script for ADVANC: %s\n', result);
end
 cd('D:\VDI_machine\back_up_VDI\catBoost_set50\')
% Read ADVANC data from Excel file instead of text file
[~, ~, advanc_data] = xlsread('D:\CatBoost_normal\window10\SET50\ADVANC\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = advanc_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
advanc_window30_crash = cell2mat(advanc_data(2:end, cum_strategy_return_idx));



 
hold all
 
plot(advanc_window30_crash)

%load aot_window5_crash.txt
%load aot_window10_crash.txt
%load aot_window30_crash.txt

% Execute Python script for AOT
fprintf('Executing Python script for AOT...\n');
cd('D:\CatBoost_normal\window10\SET50\AOT');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for AOT\n');
else
    fprintf('Error executing Python script for AOT: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

[~, ~, aot_data] = xlsread('D:\CatBoost_normal\window10\SET50\AOT\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = aot_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
aot_window30_crash = cell2mat(aot_data(2:end, cum_strategy_return_idx));

plot(aot_window30_crash)


%load banpu_window5_crash.txt
%load banpu_window10_crash.txt

% Execute Python script for BANPU
fprintf('Executing Python script for BANPU...\n');
cd('D:\CatBoost_normal\window10\SET50\BANPU');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for BANPU\n');
else
    fprintf('Error executing Python script for BANPU: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read BANPU data from Excel file instead of text file
[~, ~, banpu_data] = xlsread('D:\CatBoost_normal\window10\SET50\BANPU\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = banpu_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
banpu_window30_crash = cell2mat(banpu_data(2:end, cum_strategy_return_idx));
 
plot(banpu_window30_crash)

%load bbl_window5_crash.txt
%load bbl_window10_crash.txt

% Execute Python script for BBL
fprintf('Executing Python script for BBL...\n');
cd('D:\CatBoost_normal\window10\SET50\BBL');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for BBL\n');
else
    fprintf('Error executing Python script for BBL: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read BBL data from Excel file instead of text file
[~, ~, bbl_data] = xlsread('D:\CatBoost_normal\window10\SET50\BBL\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = bbl_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
bbl_window30_crash = cell2mat(bbl_data(2:end, cum_strategy_return_idx));
 
plot(bbl_window30_crash)


%load bem_window5_crash.txt
%load bem_window10_crash.txt

% Execute Python script for BEM
fprintf('Executing Python script for BEM...\n');
cd('D:\CatBoost_normal\window10\SET50\BEM');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for BEM\n');
else
    fprintf('Error executing Python script for BEM: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read BEM data from Excel file instead of text file
[~, ~, bem_data] = xlsread('D:\CatBoost_normal\window10\SET50\BEM\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = bem_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
bem_window30_crash = cell2mat(bem_data(2:end, cum_strategy_return_idx));

 
plot(bem_window30_crash)

%load bdms_window5_crash.txt




%load bdms_window10_crash.txt

% Execute Python script for BDMS
fprintf('Executing Python script for BDMS...\n');
cd('D:\CatBoost_normal\window10\SET50\BDMS');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for BDMS\n');
else
    fprintf('Error executing Python script for BDMS: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read BDMS data from Excel file instead of text file
[~, ~, bdms_data] = xlsread('D:\CatBoost_normal\window10\SET50\BDMS\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = bdms_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
bdms_window30_crash = cell2mat(bdms_data(2:end, cum_strategy_return_idx));
 
plot(bdms_window30_crash)


%load bh_window5_crash.txt
%load bh_window10_crash.txt

% Execute Python script for BH
fprintf('Executing Python script for BH...\n');
cd('D:\CatBoost_normal\window10\SET50\BH');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for BH\n');
else
    fprintf('Error executing Python script for BH: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read BH data from Excel file instead of text file
[~, ~, bh_data] = xlsread('D:\CatBoost_normal\window10\SET50\BH\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = bh_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
bh_window30_crash = cell2mat(bh_data(2:end, cum_strategy_return_idx));

 
plot(bh_window30_crash)


%load bem_window5_crash.txt
%load bem_window10_crash.txt
% Read BEM data from Excel file instead of text file (duplicate section)
% Data already loaded above, skipping reload
plot(bem_window30_crash)

%load bts_window5_crash.txt
%load bts_window10_crash.txt

% Execute Python script for BTS
fprintf('Executing Python script for BTS...\n');
cd('D:\CatBoost_normal\window10\SET50\BTS');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for BTS\n');
else
    fprintf('Error executing Python script for BTS: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read BTS data from Excel file instead of text file
[~, ~, bts_data] = xlsread('D:\CatBoost_normal\window10\SET50\BTS\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = bts_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
bts_window30_crash = cell2mat(bts_data(2:end, cum_strategy_return_idx));
 
plot(bts_window30_crash)

%load cbg_window5_crash.txt
%load cbg_window10_crash.txt

% Execute Python script for CBG
fprintf('Executing Python script for CBG...\n');
cd('D:\CatBoost_normal\window10\SET50\CBG');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for CBG\n');
else
    fprintf('Error executing Python script for CBG: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read CBG data from Excel file instead of text file
[~, ~, cbg_data] = xlsread('D:\CatBoost_normal\window10\SET50\CBG\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = cbg_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
cbg_window30_crash = cell2mat(cbg_data(2:end, cum_strategy_return_idx));

 
plot(cbg_window30_crash)


%load centel_window5_crash.txt
%load centel_window10_crash.txt

% Execute Python script for CENTEL
fprintf('Executing Python script for CENTEL...\n');
cd('D:\CatBoost_normal\window10\SET50\CENTEL');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for CENTEL\n');
else
    fprintf('Error executing Python script for CENTEL: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read CENTEL data from Excel file instead of text file
[~, ~, centel_data] = xlsread('D:\CatBoost_normal\window10\SET50\CENTEL\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = centel_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
centel_window30_crash = cell2mat(centel_data(2:end, cum_strategy_return_idx));
 
plot(centel_window30_crash)

%%11

%load com7_window5_crash.txt
%load com7_window10_crash.txt

% Execute Python script for COM7
fprintf('Executing Python script for COM7...\n');
cd('D:\CatBoost_normal\window10\SET50\COM_7');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for COM7\n');
else
    fprintf('Error executing Python script for COM7: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read COM7 data from Excel file instead of text file
[~, ~, com7_data] = xlsread('D:\CatBoost_normal\window10\SET50\COM_7\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = com7_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
com7_window30_crash = cell2mat(com7_data(2:end, cum_strategy_return_idx));
 
plot(com7_window30_crash)

%%12
% 
% 
%load cpall_window5_crash.txt
%load cpall_window10_crash.txt

% Execute Python script for CPALL
fprintf('Executing Python script for CPALL...\n');
cd('D:\CatBoost_normal\window10\SET50\CPALL');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for CPALL\n');
else
    fprintf('Error executing Python script for CPALL: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read CPALL data from Excel file instead of text file
[~, ~, cpall_data] = xlsread('D:\CatBoost_normal\window10\SET50\CPALL\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = cpall_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
cpall_window30_crash = cell2mat(cpall_data(2:end, cum_strategy_return_idx));
 
plot(cpall_window30_crash)
 

%%13
% 
% 
%load cpf_window5_crash.txt
%load cpf_window10_crash.txt

% Execute Python script for CPF
fprintf('Executing Python script for CPF...\n');
cd('D:\CatBoost_normal\window10\SET50\CPF');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for CPF\n');
else
    fprintf('Error executing Python script for CPF: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read CPF data from Excel file instead of text file
[~, ~, cpf_data] = xlsread('D:\CatBoost_normal\window10\SET50\CPF\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = cpf_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
cpf_window30_crash = cell2mat(cpf_data(2:end, cum_strategy_return_idx));
 
plot(cpf_window30_crash)


%%14
% 
% 
%load cpn_window5_crash.txt
%load cpn_window10_crash.txt

% Execute Python script for CPN
fprintf('Executing Python script for CPN...\n');
cd('D:\CatBoost_normal\window10\SET50\CPN');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for CPN\n');
else
    fprintf('Error executing Python script for CPN: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read CPN data from Excel file instead of text file
[~, ~, cpn_data] = xlsread('D:\CatBoost_normal\window10\SET50\CPN\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = cpn_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
cpn_window30_crash = cell2mat(cpn_data(2:end, cum_strategy_return_idx));
 
plot(cpn_window30_crash)

%%15
% 
% 
%load delta_window5_crash.txt
%load delta_window10_crash.txt

% Execute Python script for DELTA
fprintf('Executing Python script for DELTA...\n');
cd('D:\CatBoost_normal\window10\SET50\DELTA');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for DELTA\n');
else
    fprintf('Error executing Python script for DELTA: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read DELTA data from Excel file instead of text file
[~, ~, delta_data] = xlsread('D:\CatBoost_normal\window10\SET50\DELTA\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = delta_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
delta_window30_crash = cell2mat(delta_data(2:end, cum_strategy_return_idx));
 
plot(delta_window30_crash)

%%16
% 
% 
%load ea_window5_crash.txt
%load ea_window10_crash.txt

% Execute Python script for EA
fprintf('Executing Python script for EA...\n');
cd('D:\CatBoost_normal\window10\SET50\EA');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for EA\n');
else
    fprintf('Error executing Python script for EA: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read EA data from Excel file instead of text file
[~, ~, ea_data] = xlsread('D:\CatBoost_normal\window10\SET50\EA\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = ea_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
ea_window30_crash = cell2mat(ea_data(2:end, cum_strategy_return_idx));
 
plot(ea_window30_crash)

%%17
% 
% 
%load egco_window5_crash.txt
%load egco_window10_crash.txt

% Execute Python script for EGCO
fprintf('Executing Python script for EGCO...\n');
cd('D:\CatBoost_normal\window10\SET50\EGCO');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for EGCO\n');
else
    fprintf('Error executing Python script for EGCO: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read EGCO data from Excel file instead of text file
[~, ~, egco_data] = xlsread('D:\CatBoost_normal\window10\SET50\EGCO\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = egco_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
egco_window30_crash = cell2mat(egco_data(2:end, cum_strategy_return_idx));
 
plot(egco_window30_crash)

%%18
% 
% 
%load global_window5_crash.txt
%load global_window10_crash.txt

% Execute Python script for GLOBAL
fprintf('Executing Python script for GLOBAL...\n');
cd('D:\CatBoost_normal\window10\SET50\GLOBAL');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for GLOBAL\n');
else
    fprintf('Error executing Python script for GLOBAL: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read GLOBAL data from Excel file instead of text file
[~, ~, global_data] = xlsread('D:\CatBoost_normal\window10\SET50\GLOBAL\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = global_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
global_window30_crash = cell2mat(global_data(2:end, cum_strategy_return_idx));
 
plot(global_window30_crash)

%%19
% 
% 
%load gpsc_window5_crash.txt
%load gpsc_window10_crash.txt

% Execute Python script for GPSC
fprintf('Executing Python script for GPSC...\n');
cd('D:\CatBoost_normal\window10\SET50\GPSC');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for GPSC\n');
else
    fprintf('Error executing Python script for GPSC: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read GPSC data from Excel file instead of text file
[~, ~, gpsc_data] = xlsread('D:\CatBoost_normal\window10\SET50\GPSC\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = gpsc_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
gpsc_window30_crash = cell2mat(gpsc_data(2:end, cum_strategy_return_idx));
 
plot(gpsc_window30_crash)

%%20
% 
% 
%load hmpro_window5_crash.txt
%load hmpro_window10_crash.txt

% Execute Python script for HMPRO
fprintf('Executing Python script for HMPRO...\n');
cd('D:\CatBoost_normal\window10\SET50\HMPRO');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for HMPRO\n');
else
    fprintf('Error executing Python script for HMPRO: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read HMPRO data from Excel file instead of text file
[~, ~, hmpro_data] = xlsread('D:\CatBoost_normal\window10\SET50\HMPRO\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = hmpro_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
hmpro_window30_crash = cell2mat(hmpro_data(2:end, cum_strategy_return_idx));
 
plot(hmpro_window30_crash)

%%21
% 
% 
%load intuch_window5_crash.txt
%load intuch_window10_crash.txt

% Execute Python script for INTUCH
fprintf('Executing Python script for INTUCH...\n');
cd('D:\CatBoost_normal\window10\SET50\INTUCH');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for INTUCH\n');
else
    fprintf('Error executing Python script for INTUCH: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read INTUCH data from Excel file instead of text file
[~, ~, intuch_data] = xlsread('D:\CatBoost_normal\window10\SET50\INTUCH\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = intuch_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
intuch_window30_crash = cell2mat(intuch_data(2:end, cum_strategy_return_idx));
 
plot(intuch_window30_crash)

%%22
% 
% 
%load ivl_window5_crash.txt
%load ivl_window10_crash.txt

% Execute Python script for IVL
fprintf('Executing Python script for IVL...\n');
cd('D:\CatBoost_normal\window10\SET50\IVL');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for IVL\n');
else
    fprintf('Error executing Python script for IVL: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read IVL data from Excel file instead of text file
[~, ~, ivl_data] = xlsread('D:\CatBoost_normal\window10\SET50\IVL\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = ivl_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
ivl_window30_crash = cell2mat(ivl_data(2:end, cum_strategy_return_idx));
 
plot(ivl_window30_crash)

%%23
% 
% 
%load kbank_window5_crash.txt
%load kbank_window10_crash.txt

% Execute Python script for KBANK
fprintf('Executing Python script for KBANK...\n');
cd('D:\CatBoost_normal\window10\SET50\KBANK');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for KBANK\n');
else
    fprintf('Error executing Python script for KBANK: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read KBANK data from Excel file instead of text file
[~, ~, kbank_data] = xlsread('D:\CatBoost_normal\window10\SET50\KBANK\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = kbank_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
kbank_window30_crash = cell2mat(kbank_data(2:end, cum_strategy_return_idx));
 
plot(kbank_window30_crash)

%%24
% 
% 
%load ktc_window5_crash.txt
%load ktc_window10_crash.txt

% Execute Python script for KTC
fprintf('Executing Python script for KTC...\n');
cd('D:\CatBoost_normal\window10\SET50\KTC');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for KTC\n');
else
    fprintf('Error executing Python script for KTC: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read KTC data from Excel file instead of text file
[~, ~, ktc_data] = xlsread('D:\CatBoost_normal\window10\SET50\KTC\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = ktc_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
ktc_window30_crash = cell2mat(ktc_data(2:end, cum_strategy_return_idx));
 
plot(ktc_window30_crash)

%%25
% 
% 
%load ktb_window5_crash.txt
%load ktb_window10_crash.txt

% Execute Python script for KTB
fprintf('Executing Python script for KTB...\n');
cd('D:\CatBoost_normal\window10\SET50\KTB');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for KTB\n');
else
    fprintf('Error executing Python script for KTB: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read KTB data from Excel file instead of text file
[~, ~, ktb_data] = xlsread('D:\CatBoost_normal\window10\SET50\KTB\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = ktb_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
ktb_window30_crash = cell2mat(ktb_data(2:end, cum_strategy_return_idx));
 
plot(ktb_window30_crash)

%%26
% 
% 
%load lh_window5_crash.txt
%load lh_window10_crash.txt

% Execute Python script for LH
fprintf('Executing Python script for LH...\n');
cd('D:\CatBoost_normal\window10\SET50\LH');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for LH\n');
else
    fprintf('Error executing Python script for LH: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read LH data from Excel file instead of text file
[~, ~, lh_data] = xlsread('D:\CatBoost_normal\window10\SET50\LH\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = lh_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
lh_window30_crash = cell2mat(lh_data(2:end, cum_strategy_return_idx));
 
plot(lh_window30_crash)

%%27
% 
% 
%load mtc_window5_crash.txt
%load mtc_window10_crash.txt

% Execute Python script for MTC
fprintf('Executing Python script for MTC...\n');
cd('D:\CatBoost_normal\window10\SET50\MTC');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for MTC\n');
else
    fprintf('Error executing Python script for MTC: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read MTC data from Excel file instead of text file
[~, ~, mtc_data] = xlsread('D:\CatBoost_normal\window10\SET50\MTC\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = mtc_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
mtc_window30_crash = cell2mat(mtc_data(2:end, cum_strategy_return_idx));
 
plot(mtc_window30_crash)

%%28
% 
% 
%load mint_window5_crash.txt
%load mint_window10_crash.txt

% Execute Python script for MINT
fprintf('Executing Python script for MINT...\n');
cd('D:\CatBoost_normal\window10\SET50\MINT');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for MINT\n');
else
    fprintf('Error executing Python script for MINT: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read MINT data from Excel file instead of text file
[~, ~, mint_data] = xlsread('D:\CatBoost_normal\window10\SET50\MINT\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = mint_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
mint_window30_crash = cell2mat(mint_data(2:end, cum_strategy_return_idx));
 
plot(mint_window30_crash)

%%29
% 
% 
%load pttgc_window5_crash.txt
%load pttgc_window10_crash.txt

% Execute Python script for PTTGC
fprintf('Executing Python script for PTTGC...\n');
cd('D:\CatBoost_normal\window10\SET50\PTTGC');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for PTTGC\n');
else
    fprintf('Error executing Python script for PTTGC: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read PTTGC data from Excel file instead of text file
[~, ~, pttgc_data] = xlsread('D:\CatBoost_normal\window10\SET50\PTTGC\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = pttgc_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
pttgc_window30_crash = cell2mat(pttgc_data(2:end, cum_strategy_return_idx));
 
plot(pttgc_window30_crash)

%%30
% 
% 
%load ptt_window5_crash.txt
%load ptt_window10_crash.txt

% Execute Python script for PTT
fprintf('Executing Python script for PTT...\n');
cd('D:\CatBoost_normal\window10\SET50\PTT');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for PTT\n');
else
    fprintf('Error executing Python script for PTT: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read PTT data from Excel file instead of text file
[~, ~, ptt_data] = xlsread('D:\CatBoost_normal\window10\SET50\PTT\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = ptt_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
ptt_window30_crash = cell2mat(ptt_data(2:end, cum_strategy_return_idx));
 
plot(ptt_window30_crash)

%%31
% 
% 
%load pttep_window5_crash.txt
%load pttep_window10_crash.txt

% Execute Python script for PTTEP
fprintf('Executing Python script for PTTEP...\n');
cd('D:\CatBoost_normal\window10\SET50\PTTEP');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for PTTEP\n');
else
    fprintf('Error executing Python script for PTTEP: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read PTTEP data from Excel file instead of text file
[~, ~, pttep_data] = xlsread('D:\CatBoost_normal\window10\SET50\PTTEP\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = pttep_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
pttep_window30_crash = cell2mat(pttep_data(2:end, cum_strategy_return_idx));
 
plot(pttep_window30_crash)
%%32
% 
% 
%load ratch_window5_crash.txt
%load ratch_window10_crash.txt

% Execute Python script for RATCH
fprintf('Executing Python script for RATCH...\n');
cd('D:\CatBoost_normal\window10\SET50\RATCH');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for RATCH\n');
else
    fprintf('Error executing Python script for RATCH: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read RATCH data from Excel file instead of text file
[~, ~, ratch_data] = xlsread('D:\CatBoost_normal\window10\SET50\RATCH\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = ratch_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
ratch_window30_crash = cell2mat(ratch_data(2:end, cum_strategy_return_idx));
 
plot(ratch_window30_crash)
%%33
% 
% 
%load sawad_window5_crash.txt
%load sawad_window10_crash.txt

% Execute Python script for SAWAD
fprintf('Executing Python script for SAWAD...\n');
cd('D:\CatBoost_normal\window10\SET50\SAWAD');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for SAWAD\n');
else
    fprintf('Error executing Python script for SAWAD: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read SAWAD data from Excel file instead of text file
[~, ~, sawad_data] = xlsread('D:\CatBoost_normal\window10\SET50\SAWAD\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = sawad_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
sawad_window30_crash = cell2mat(sawad_data(2:end, cum_strategy_return_idx));
 
plot(sawad_window30_crash)

%%34
% 
% 
%load scc_window5_crash.txt
%load scc_window10_crash.txt

% Execute Python script for SCC
fprintf('Executing Python script for SCC...\n');
cd('D:\CatBoost_normal\window10\SET50\SCC');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for SCC\n');
else
    fprintf('Error executing Python script for SCC: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read SCC data from Excel file instead of text file
[~, ~, scc_data] = xlsread('D:\CatBoost_normal\window10\SET50\SCC\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = scc_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
scc_window30_crash = cell2mat(scc_data(2:end, cum_strategy_return_idx));
 
plot(scc_window30_crash)

%%35
% 
% 
%load ttb_window5_crash.txt
%load ttb_window10_crash.txt

% Execute Python script for TTB
fprintf('Executing Python script for TTB...\n');
cd('D:\CatBoost_normal\window10\SET50\TTB');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for TTB\n');
else
    fprintf('Error executing Python script for TTB: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read TTB data from Excel file instead of text file
[~, ~, ttb_data] = xlsread('D:\CatBoost_normal\window10\SET50\TTB\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = ttb_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
ttb_window30_crash = cell2mat(ttb_data(2:end, cum_strategy_return_idx));
 
plot(ttb_window30_crash)

%%36
% 
% 
%load tisco_window5_crash.txt
%load tisco_window10_crash.txt

% Execute Python script for TISCO
fprintf('Executing Python script for TISCO...\n');
cd('D:\CatBoost_normal\window10\SET50\TISCO');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for TISCO\n');
else
    fprintf('Error executing Python script for TISCO: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read TISCO data from Excel file instead of text file
[~, ~, tisco_data] = xlsread('D:\CatBoost_normal\window10\SET50\TISCO\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = tisco_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
tisco_window30_crash = cell2mat(tisco_data(2:end, cum_strategy_return_idx));
 
plot(tisco_window30_crash)

%%37
% 
% 
%load top_window5_crash.txt
%load top_window10_crash.txt

% Execute Python script for TOP
fprintf('Executing Python script for TOP...\n');
cd('D:\CatBoost_normal\window10\SET50\TOP');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for TOP\n');
else
    fprintf('Error executing Python script for TOP: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read TOP data from Excel file instead of text file
[~, ~, top_data] = xlsread('D:\CatBoost_normal\window10\SET50\TOP\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = top_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
top_window30_crash = cell2mat(top_data(2:end, cum_strategy_return_idx));
 
plot(top_window30_crash)

%%38
% 
% 
%load tu_window5_crash.txt
%load tu_window10_crash.txt

% Execute Python script for TU
fprintf('Executing Python script for TU...\n');
cd('D:\CatBoost_normal\window10\SET50\TU');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for TU\n');
else
    fprintf('Error executing Python script for TU: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read TU data from Excel file instead of text file
[~, ~, tu_data] = xlsread('D:\CatBoost_normal\window10\SET50\TU\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = tu_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
tu_window30_crash = cell2mat(tu_data(2:end, cum_strategy_return_idx));
 
plot(tu_window30_crash)

%%39
% 
% 
%load wha_window5_crash.txt
%load wha_window10_crash.txt

% Execute Python script for WHA
fprintf('Executing Python script for WHA...\n');
cd('D:\CatBoost_normal\window10\SET50\WHA');
[status, result] = system('python test_cat3.py');
if status == 0
    fprintf('Python script executed successfully for WHA\n');
else
    fprintf('Error executing Python script for WHA: %s\n', result);
end
cd('D:\VDI_machine\back_up_VDI\catBoost_set50\'); % Return to original directory

% Read WHA data from Excel file instead of text file
[~, ~, wha_data] = xlsread('D:\CatBoost_normal\window10\SET50\WHA\backtest_results_iter0.xlsx', 'BacktestData');
% Find the column index for 'cum_strategy_return'
headers = wha_data(1, :);
cum_strategy_return_idx = find(strcmp(headers, 'cum_strategy_return'));
% Extract the data from the cum_strategy_return column (skip header row)
wha_window30_crash = cell2mat(wha_data(2:end, cum_strategy_return_idx));
 
plot(wha_window30_crash)



all_window30_crash=wha_window30_crash+advanc_window30_crash+aot_window30_crash+banpu_window30_crash+bem_window30_crash+bbl_window30_crash+bdms_window30_crash+bh_window30_crash+bts_window30_crash+cbg_window30_crash+centel_window30_crash+com7_window30_crash+cpall_window30_crash+cpf_window30_crash+cpn_window30_crash+delta_window30_crash+ea_window30_crash+egco_window30_crash+global_window30_crash+gpsc_window30_crash+hmpro_window30_crash+intuch_window30_crash+ivl_window30_crash+kbank_window30_crash+ktc_window30_crash+ktb_window30_crash+lh_window30_crash+mtc_window30_crash+mint_window30_crash+pttgc_window30_crash+ptt_window30_crash+pttep_window30_crash+ratch_window30_crash+sawad_window30_crash+scc_window30_crash+ttb_window30_crash+tisco_window30_crash+top_window30_crash+tu_window30_crash

all_window30_crash_avg=all_window30_crash/39 

   %all_crash_avg  =(all_window5_crash_avg+all_window10_crash_avg+all_window30_crash_avg)/3
   plot(all_window30_crash_avg,'LineWidth',3,'Color','Black')