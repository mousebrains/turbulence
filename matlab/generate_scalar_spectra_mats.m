% generate_scalar_spectra_mats.m
% Generate .mat files containing scalar (temperature gradient) spectra
% from the ODAS MATLAB Library for validation against rsi-python chi.
%
% Uses get_scalar_spectra_odas.m as the independent reference for the
% scalar spectrum pipeline (first-difference correction, bilinear
% correction, Goodman scalar cleaning).
%
% Usage: Run from the turbulence/ directory with ODAS on the path:
%   addpath('odas');
%   run('matlab/generate_scalar_spectra_mats.m');
%
% Output: VMP/*_scalar_spectra.mat for each .p file
%
% Mar-2026, Claude and Pat Welch, pat@mousebrains.com

%% Configuration — match Python defaults
fft_length  = 256;
spec_length = 2 * fft_length;   % = 512 (diss_length)
overlap_val = spec_length / 2;  % = 256
f_AA        = 98;
P_min       = 0.5;
W_min       = 0.3;
direction   = 'down';
min_duration = 7.0;

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
    out_paths{fi}  = fullfile(p_files(fi).folder, [stem '_scalar_spectra.mat']);
end

%% Start a process-based parallel pool (threads can't save/load)
pool = gcp('nocreate');
if isempty(pool) || ~strcmpi(pool.Cluster.Profile, 'Processes')
    delete(gcp('nocreate'));
    parpool('Processes');
end

parfor fi = 1:n_files
    p_path   = p_paths{fi};
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
        P_slow     = result.P_slow;
        speed_fast = result.speed_fast;
        W_slow_vec = result.W_slow;
        P_fast     = result.P_fast;

        % Temperature gradient channels (pre-emphasized, from odas_p2mat)
        gradT1 = result.gradT1;
        gradT2 = result.gradT2;

        % Accelerometers for Goodman cleaning
        Ax = result.Ax;
        Ay = result.Ay;

        % Get differentiator gains from setupstr
        setupfilestr = result.setupfilestr;
        diff_gain_T1 = str2double(setupstr(setupfilestr, 'T1_dT1', 'diff_gain'));
        diff_gain_T2 = str2double(setupstr(setupfilestr, 'T2_dT2', 'diff_gain'));
        diff_gains = [diff_gain_T1, diff_gain_T2];

        %% 2. Profile detection
        profiles = get_profile(P_slow, W_slow_vec, P_min, W_min, ...
                               direction, min_duration, fs_slow);

        if isempty(profiles)
            fprintf('  No profiles found, skipping %s\n', p_path);
        else

        n_profiles = size(profiles, 2);
        fprintf('  Found %d profiles\n', n_profiles);

        %% 3. Compute scalar spectra for each profile
        scalar_spec_all  = cell(n_profiles, 1);
        K_scalar_all     = cell(n_profiles, 1);
        F_scalar_all     = cell(n_profiles, 1);
        speed_scalar_all = cell(n_profiles, 1);
        P_scalar_all     = cell(n_profiles, 1);
        t_scalar_all     = cell(n_profiles, 1);

        for pi = 1:n_profiles
            s_slow = profiles(1, pi);
            e_slow = profiles(2, pi);

            % Convert to fast indices
            s_fast = (s_slow - 1) * ratio + 1;
            e_fast = min(e_slow * ratio, length(t_fast));

            range_fast = s_fast:e_fast;

            % Scalar gradient matrix (two thermistors)
            gradT_mat = [gradT1(range_fast), gradT2(range_fast)];

            % Accelerometer matrix (reference signals for Goodman)
            A = [Ax(range_fast), Ay(range_fast)];

            % Call get_scalar_spectra_odas
            sp = get_scalar_spectra_odas(...
                gradT_mat, A, ...
                P_fast(range_fast), ...
                t_fast(range_fast), ...
                speed_fast(range_fast), ...
                'diff_gain', diff_gains, ...
                'fft_length', fft_length, ...
                'spec_length', spec_length, ...
                'overlap', overlap_val, ...
                'fs', fs_fast, ...
                'f_AA', f_AA, ...
                'gradient_method', 'first_difference', ...
                'goodman', true);

            scalar_spec_all{pi}  = sp.scalar_spec;  % (n_freq, n_therm, n_est)
            K_scalar_all{pi}     = sp.K;             % (n_freq, n_est)
            F_scalar_all{pi}     = sp.F;             % (n_freq, n_est)
            speed_scalar_all{pi} = sp.speed;         % (n_est, 1)
            P_scalar_all{pi}     = sp.P;             % (n_est, 1)
            t_scalar_all{pi}     = sp.t;             % (n_est, 1)

            n_est = size(sp.scalar_spec, 3);
            fprintf('  Profile %d: %d estimates\n', pi, n_est);
        end

        %% 4. Save
        parsave_scalar(out_path, ...
            fs_fast, fs_slow, profiles, n_profiles, ...
            scalar_spec_all, K_scalar_all, F_scalar_all, ...
            speed_scalar_all, P_scalar_all, t_scalar_all, ...
            diff_gains, fft_length, spec_length, overlap_val, f_AA);

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


function parsave_scalar(out_path, ...
    fs_fast, fs_slow, profiles, n_profiles, ...
    scalar_spec_all, K_scalar_all, F_scalar_all, ...
    speed_scalar_all, P_scalar_all, t_scalar_all, ...
    diff_gains, fft_length, spec_length, overlap_val, f_AA)
% PARSAVE_SCALAR  Save scalar spectra data inside a parfor loop.
%   Wraps save() in a function so parfor's transparency analysis is satisfied.
    save(out_path, ...
         'fs_fast', 'fs_slow', ...
         'profiles', 'n_profiles', ...
         'scalar_spec_all', 'K_scalar_all', 'F_scalar_all', ...
         'speed_scalar_all', 'P_scalar_all', 't_scalar_all', ...
         'diff_gains', ...
         'fft_length', 'spec_length', 'overlap_val', 'f_AA', ...
         '-v7.3');
end
