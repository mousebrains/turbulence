% generate_validation_mats.m
% Generate .mat files containing epsilon, profile, and speed data
% from the ODAS MATLAB Library for validation against rsi-python.
%
% Usage: Run from the turbulence/ directory with ODAS on the path:
%   addpath('odas');
%   run('matlab/generate_validation_mats.m');
%
% Output: VMP/*_validation.mat for each .p file
%
% Mar-2026, Claude and Pat Welch, pat@mousebrains.com

%% Configuration — match Python defaults
fft_length  = 256;
diss_length = 2 * fft_length;   % = 512
overlap_val = diss_length / 2;  % = 256
f_AA        = 98;
fit_order   = 3;
P_min       = 0.5;   % minimum pressure for profile detection [dbar]
W_min       = 0.3;   % minimum fall rate [m/s]
direction   = 'down';
min_duration = 7.0;  % minimum profile duration [s]
despike_thresh = 8;
despike_smooth = 0.5; % [Hz]

vmp_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'VMP');
p_files = dir(fullfile(vmp_dir, '*.p'));
n_files = length(p_files);

fprintf('Found %d .p files in %s\n', n_files, vmp_dir);

% Pre-build path arrays for parfor (struct indexing inside parfor is fragile)
p_paths   = cell(n_files, 1);
out_paths = cell(n_files, 1);
for fi = 1:n_files
    p_paths{fi}   = fullfile(p_files(fi).folder, p_files(fi).name);
    [~, stem, ~]   = fileparts(p_paths{fi});
    out_paths{fi}  = fullfile(p_files(fi).folder, [stem '_validation.mat']);
end

%% Start a process-based parallel pool (threads can't save/load)
pool = gcp('nocreate');
if isempty(pool) || ~strcmpi(pool.Cluster.Profile, 'Processes')
    delete(gcp('nocreate'));
    parpool('Processes');
end

parfor fi = 1:n_files
    p_path  = p_paths{fi};
    out_path = out_paths{fi};

    fprintf('\n=== [%d/%d] %s ===\n', fi, n_files, p_path);

    try
        %% 1. Load and convert data using odas_p2mat
        result = odas_p2mat(p_path);

        fs_fast = result.fs_fast;
        fs_slow = result.fs_slow;
        ratio   = round(fs_fast / fs_slow);

        % Extract channels
        t_fast     = result.t_fast;
        t_slow     = result.t_slow;
        P_slow     = result.P_slow;
        T1_slow    = result.T1_slow;
        speed_slow = result.speed_slow;
        speed_fast = result.speed_fast;
        W_slow_vec = result.W_slow;

        % Shear probes (already in du/dz, divided by speed^2)
        sh1 = result.sh1;
        sh2 = result.sh2;

        % Accelerometers (this instrument has Ax and Ay only)
        Ax = result.Ax;
        Ay = result.Ay;

        % Temperature fast (for viscosity)
        T1_fast = result.T1_fast;

        % Pressure fast
        P_fast = result.P_fast;

        %% 2. Profile detection
        profiles = get_profile(P_slow, W_slow_vec, P_min, W_min, ...
                               direction, min_duration, fs_slow);

        if isempty(profiles)
            fprintf('  No profiles found, skipping %s\n', p_path);
        else

        n_profiles = size(profiles, 2);
        fprintf('  Found %d profiles\n', n_profiles);

        %% 3. Compute epsilon for each profile
        % Store results in cell arrays (profiles may differ in length)
        eps_all      = cell(n_profiles, 1);
        K_max_all    = cell(n_profiles, 1);
        FM_all       = cell(n_profiles, 1);
        mad_all      = cell(n_profiles, 1);
        method_all   = cell(n_profiles, 1);
        nu_all       = cell(n_profiles, 1);
        speed_all    = cell(n_profiles, 1);
        P_mean_all   = cell(n_profiles, 1);
        T_mean_all   = cell(n_profiles, 1);
        t_mean_all   = cell(n_profiles, 1);
        K_all        = cell(n_profiles, 1);
        F_all        = cell(n_profiles, 1);
        spec_sh_all  = cell(n_profiles, 1);
        dof_spec_all = zeros(n_profiles, 1);
        prof_idx_all = profiles;  % 2 x n_profiles (slow indices)

        for pi = 1:n_profiles
            s_slow = profiles(1, pi);
            e_slow = profiles(2, pi);

            % Convert to fast indices
            s_fast = (s_slow - 1) * ratio + 1;
            e_fast = min(e_slow * ratio, length(t_fast));

            range_fast = s_fast:e_fast;
            N_range = length(range_fast);

            % Shear matrix (already in du/dz from odas_p2mat)
            SH = [sh1(range_fast), sh2(range_fast)];

            % Accelerometer matrix
            A = [Ax(range_fast), Ay(range_fast)];

            % Despike shear
            for si = 1:size(SH, 2)
                SH(:, si) = despike(SH(:, si), despike_thresh, ...
                                    despike_smooth, fs_fast, round(fs_fast/2));
            end

            % Build info structure for get_diss_odas
            info = struct();
            info.fft_length  = fft_length;
            info.diss_length = diss_length;
            info.overlap     = overlap_val;
            info.fs_fast     = fs_fast;
            info.fs_slow     = fs_slow;
            info.speed       = speed_fast(range_fast);
            info.T           = T1_fast(range_fast);
            info.P           = P_fast(range_fast);
            info.t_fast      = t_fast(range_fast);
            info.f_AA        = f_AA;
            info.fit_order   = fit_order;
            info.goodman     = true;

            % Compute dissipation
            diss = get_diss_odas(SH, A, info);

            eps_all{pi}      = diss.e;       % [n_shear x n_est]
            K_max_all{pi}    = diss.K_max;
            FM_all{pi}       = diss.FM;
            mad_all{pi}      = diss.mad;
            method_all{pi}   = diss.method;
            nu_all{pi}       = diss.nu;      % column vector
            speed_all{pi}    = diss.speed;   % column vector
            P_mean_all{pi}   = diss.P;       % column vector
            T_mean_all{pi}   = diss.T;       % column vector
            t_mean_all{pi}   = diss.t;       % column vector
            K_all{pi}        = diss.K;       % wavenumber vector
            F_all{pi}        = diss.F;       % frequency vector
            spec_sh_all{pi}  = diss.sh_clean; % cleaned spectra
            dof_spec_all(pi) = diss.dof_spec;

            n_est = size(diss.e, 2);
            P_range = diss.P;
            fprintf('  Profile %d: %d estimates, P=%.0f-%.0f dbar\n', ...
                    pi, n_est, min(P_range), max(P_range));
        end

        %% 4. Save validation .mat file
        parsave_validation(out_path, ...
            fs_fast, fs_slow, profiles, n_profiles, ...
            eps_all, K_max_all, FM_all, mad_all, method_all, ...
            nu_all, speed_all, P_mean_all, T_mean_all, t_mean_all, ...
            K_all, F_all, spec_sh_all, dof_spec_all, prof_idx_all, ...
            fft_length, diss_length, overlap_val, f_AA, fit_order, ...
            P_min, W_min, min_duration);

        fprintf('  Saved %s\n', out_path);

        end % if ~isempty(profiles)

    catch ME
        fprintf('  ERROR processing %s: %s\n', p_path, ME.message);
        if ~isempty(ME.stack)
            fprintf('  %s\n', ME.stack(1).name);
        end
    end
end

fprintf('\nDone.\n');


function parsave_validation(out_path, ...
    fs_fast, fs_slow, profiles, n_profiles, ...
    eps_all, K_max_all, FM_all, mad_all, method_all, ...
    nu_all, speed_all, P_mean_all, T_mean_all, t_mean_all, ...
    K_all, F_all, spec_sh_all, dof_spec_all, prof_idx_all, ...
    fft_length, diss_length, overlap_val, f_AA, fit_order, ...
    P_min, W_min, min_duration)
% PARSAVE_VALIDATION  Save validation data inside a parfor loop.
%   Wraps save() in a function so parfor's transparency analysis is satisfied.
    save(out_path, ...
         'fs_fast', 'fs_slow', ...
         'profiles', 'n_profiles', ...
         'eps_all', 'K_max_all', 'FM_all', 'mad_all', 'method_all', ...
         'nu_all', 'speed_all', 'P_mean_all', 'T_mean_all', 't_mean_all', ...
         'K_all', 'F_all', 'spec_sh_all', 'dof_spec_all', ...
         'prof_idx_all', ...
         'fft_length', 'diss_length', 'overlap_val', 'f_AA', 'fit_order', ...
         'P_min', 'W_min', 'min_duration', ...
         '-v7.3');
end
