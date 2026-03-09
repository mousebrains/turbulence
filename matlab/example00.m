% example00.m — Test script for chi calculation from a .p file
% Mar-2026, Claude and Pat Welch, pat@mousebrains.com

%% Paths
myPath = fileparts(mfilename("fullpath"));
rootPath = fullfile(myPath, '..');
addpath(fullfile(rootPath, 'odas'));    % ODAS library
addpath(myPath);                        % chi functions

%% 1. Convert .p file to MATLAB workspace
fname = fullfile(rootPath, 'VMP', 'ARCTERX_Thompson_2025_SN479_0006.p');
fprintf('Reading %s\n', fname);
d = odas_p2mat(fname);

%% 2. Detect profiles
W = gradient(d.P_slow, 1/d.fs_slow);   % dP/dt from slow pressure
profiles = get_profile(d.P_slow, W, 0.5, 0.3, 'down', 7, d.fs_slow);
n_prof = size(profiles, 2);
fprintf('Found %d profiles\n', n_prof);

if n_prof == 0
    error('probar:noProfiles', 'No profiles detected');
end

%% 3. Extract first profile
ratio = round(d.fs_fast / d.fs_slow);
s_slow = profiles(1,1);
e_slow = profiles(2,1);
s_fast = (s_slow - 1) * ratio + 1;
e_fast = e_slow * ratio;

fprintf('Profile 1: slow [%d:%d], fast [%d:%d]\n', ...
    s_slow, e_slow, s_fast, e_fast);

%% 4. Temperature gradient — use ODAS pre-computed gradT1
gradT1 = d.gradT1(s_fast:e_fast);

%% Diagnostics
fprintf('JAC_T range: [%.2f, %.2f] °C\n', ...
    min(d.JAC_T(s_slow:e_slow)), max(d.JAC_T(s_slow:e_slow)));
fprintf('P_fast range: [%.1f, %.1f] dbar\n', ...
    min(d.P_fast(s_fast:e_fast)), max(d.P_fast(s_fast:e_fast)));
fprintf('speed range: [%.3f, %.3f] m/s\n', ...
    min(d.speed_fast(s_fast:e_fast)), max(d.speed_fast(s_fast:e_fast)));

%% 5. Method 2 — MLE fit (no epsilon needed)
fft_len = 256;
diss_len = 3 * fft_len;
olap = floor(diss_len / 2);

fprintf('Computing chi — Method 2 ...\n');
results_m2 = get_chi(gradT1, d.P_fast(s_fast:e_fast), ...
    d.JAC_T(s_slow:e_slow), d.speed_fast(s_fast:e_fast), d.fs_fast, ...
    fft_length=fft_len, diss_length=diss_len, overlap=olap);

fprintf('Method 2: %d estimates, chi range [%.2e, %.2e] K^2/s\n', ...
    numel(results_m2.chi), min(results_m2.chi), max(results_m2.chi));

%% 6. Compute epsilon from shear probes (for Method 1)
fprintf('Computing epsilon from shear probes ...\n');
info = struct( ...
    'fft_length',  256, ...
    'diss_length', 768, ...
    'overlap',     384, ...
    'fs_fast',     d.fs_fast, ...
    'fs_slow',     d.fs_slow, ...
    'speed',       d.speed_fast(s_fast:e_fast), ...
    'T',           d.T1_fast(s_fast:e_fast), ...
    'P',           d.P_fast(s_fast:e_fast), ...
    'goodman',     true);

diss = get_diss_odas(d.sh1(s_fast:e_fast), ...
    [d.Ax(s_fast:e_fast), d.Ay(s_fast:e_fast)], info);

fprintf('Epsilon range: [%.2e, %.2e] W/kg\n', min(diss.e), max(diss.e));

%% 7. Method 1 — chi from epsilon
fprintf('Computing chi — Method 1 ...\n');
results_m1 = get_chi(gradT1, d.P_fast(s_fast:e_fast), ...
    d.JAC_T(s_slow:e_slow), d.speed_fast(s_fast:e_fast), d.fs_fast, ...
    method=1, epsilon=diss.e, ...
    fft_length=fft_len, diss_length=diss_len, overlap=olap);

fprintf('Method 1: %d estimates, chi range [%.2e, %.2e] K^2/s\n', ...
    numel(results_m1.chi), min(results_m1.chi), max(results_m1.chi));

%% 8. Compute salinity
P_slow = d.P_slow(s_slow:e_slow);
T_slow = d.JAC_T(s_slow:e_slow);
C_slow = d.JAC_C(s_slow:e_slow);
sal_info = struct('fs', d.fs_slow, 'speed', mean(d.speed_slow(s_slow:e_slow)));
S = salinity_JAC(P_slow, T_slow, C_slow, sal_info);

%% 9. Plot
figure('Name', 'Profile 1');

% Left panel — chi and epsilon
ax1 = subplot(1,2,1);
semilogx(results_m2.chi, results_m2.P_mean, 'b-', 'LineWidth', 1.5);
hold on;
semilogx(results_m1.chi, results_m1.P_mean, 'r-', 'LineWidth', 1.5);
semilogx(diss.e, diss.P, 'k--', 'LineWidth', 1);
hold off;
set(ax1, 'YDir', 'reverse');
xlabel('\chi [K^2/s]  /  \epsilon [W/kg]');
ylabel('Pressure [dbar]');
legend('\chi Method 2 (MLE)', '\chi Method 1 (from \epsilon)', ...
    '\epsilon shear', 'Location', 'best');
grid on;

% Right panel — temperature (bottom x) and salinity (top x)
ax2 = subplot(1,2,2);
plot(T_slow, P_slow, 'g-', 'LineWidth', 1.5);
set(ax2, 'YDir', 'reverse', 'XColor', 'g');
xlabel('Temperature [°C]');
ylabel('Pressure [dbar]');
grid on;

ax3 = axes('Position', ax2.Position, ...
    'XAxisLocation', 'top', 'YDir', 'reverse', ...
    'Color', 'none', 'YTickLabel', []);
hold(ax3, 'on');
plot(ax3, S, P_slow, 'm-', 'LineWidth', 1.5);
hold(ax3, 'off');
set(ax3, 'XColor', 'm');
xlabel(ax3, 'Salinity [PSU]');
linkaxes([ax2 ax3], 'y');

% Link pressure axes across both panels
linkaxes([ax1 ax2], 'y');

% Title on left panel only to avoid overlapping salinity axis
title(ax1, fname, 'Interpreter', 'none');

fprintf('Done.\n');
