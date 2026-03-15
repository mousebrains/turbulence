% generate_validation_nc.m
% Generate CF-compliant NetCDF files containing epsilon, profile, and speed
% data from the ODAS MATLAB Library for validation against microstructure-tpw.
%
% Usage: Run from the turbulence/ directory with ODAS on the path:
%   addpath('odas');
%   run('matlab/generate_validation_nc.m');
%
% Output: VMP/*_validation.nc for each .p file
%
% Each file has one group per profile (/profile_001, /profile_002, ...),
% plus root-level scalar attributes for configuration parameters.
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
HP_cut         = 0.25; % high-pass cutoff [Hz]

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
    out_paths{fi}  = fullfile(p_files(fi).folder, [stem '_validation.nc']);
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

        %% 4. Save validation NetCDF file
        parsave_validation_nc(out_path, ...
            fs_fast, fs_slow, profiles, n_profiles, ...
            eps_all, K_max_all, FM_all, mad_all, method_all, ...
            nu_all, speed_all, P_mean_all, T_mean_all, t_mean_all, ...
            K_all, F_all, spec_sh_all, dof_spec_all, prof_idx_all, ...
            fft_length, diss_length, overlap_val, f_AA, fit_order, ...
            P_min, W_min, min_duration, HP_cut);

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


function parsave_validation_nc(out_path, ...
    fs_fast, fs_slow, profiles, n_profiles, ...
    eps_all, K_max_all, FM_all, mad_all, method_all, ...
    nu_all, speed_all, P_mean_all, T_mean_all, t_mean_all, ...
    K_all, F_all, spec_sh_all, dof_spec_all, prof_idx_all, ...
    fft_length, diss_length, overlap_val, f_AA, fit_order, ...
    P_min, W_min, min_duration, HP_cut)
% PARSAVE_VALIDATION_NC  Write validation data to CF-compliant NetCDF4.
%   Each profile is stored in its own group: /profile_001, /profile_002, ...
%   Root group contains global attributes and configuration parameters.

    % Delete existing file (netcdf4 doesn't overwrite cleanly)
    if isfile(out_path)
        delete(out_path);
    end

    % --- Root group: global attributes ---
    ncid = netcdf.create(out_path, bitor(netcdf.getConstant('NETCDF4'), ...
                                         netcdf.getConstant('CLOBBER')));

    netcdf.putAtt(ncid, netcdf.getConstant('NC_GLOBAL'), ...
        'Conventions', 'CF-1.8');
    netcdf.putAtt(ncid, netcdf.getConstant('NC_GLOBAL'), ...
        'title', 'ODAS epsilon validation data');
    netcdf.putAtt(ncid, netcdf.getConstant('NC_GLOBAL'), ...
        'institution', 'Oregon State University');
    netcdf.putAtt(ncid, netcdf.getConstant('NC_GLOBAL'), ...
        'source', 'ODAS MATLAB Library (get_diss_odas)');
    netcdf.putAtt(ncid, netcdf.getConstant('NC_GLOBAL'), ...
        'history', [datestr(now, 'yyyy-mm-ddTHH:MM:SS') ...
        ' generate_validation_nc.m']);

    % Configuration as global attributes
    netcdf.putAtt(ncid, netcdf.getConstant('NC_GLOBAL'), ...
        'fs_fast', fs_fast);
    netcdf.putAtt(ncid, netcdf.getConstant('NC_GLOBAL'), ...
        'fs_slow', fs_slow);
    netcdf.putAtt(ncid, netcdf.getConstant('NC_GLOBAL'), ...
        'n_profiles', int32(n_profiles));
    netcdf.putAtt(ncid, netcdf.getConstant('NC_GLOBAL'), ...
        'fft_length', int32(fft_length));
    netcdf.putAtt(ncid, netcdf.getConstant('NC_GLOBAL'), ...
        'diss_length', int32(diss_length));
    netcdf.putAtt(ncid, netcdf.getConstant('NC_GLOBAL'), ...
        'overlap', int32(overlap_val));
    netcdf.putAtt(ncid, netcdf.getConstant('NC_GLOBAL'), ...
        'f_AA', f_AA);
    netcdf.putAtt(ncid, netcdf.getConstant('NC_GLOBAL'), ...
        'fit_order', int32(fit_order));
    netcdf.putAtt(ncid, netcdf.getConstant('NC_GLOBAL'), ...
        'P_min', P_min);
    netcdf.putAtt(ncid, netcdf.getConstant('NC_GLOBAL'), ...
        'W_min', W_min);
    netcdf.putAtt(ncid, netcdf.getConstant('NC_GLOBAL'), ...
        'min_duration', min_duration);
    netcdf.putAtt(ncid, netcdf.getConstant('NC_GLOBAL'), ...
        'HP_cut', HP_cut);

    % Profile slow-sample indices (2 x n_profiles) in root group
    dim_two = netcdf.defDim(ncid, 'bounds', 2);
    dim_prof = netcdf.defDim(ncid, 'profile', n_profiles);
    vid = netcdf.defVar(ncid, 'profile_indices', 'NC_INT', [dim_two, dim_prof]);
    netcdf.putAtt(ncid, vid, 'long_name', 'slow-sample start/end indices per profile');
    netcdf.putAtt(ncid, vid, 'comment', 'row 0 = start, row 1 = end (1-based MATLAB indices)');
    netcdf.putVar(ncid, vid, int32(prof_idx_all));

    vid = netcdf.defVar(ncid, 'dof_spec', 'NC_DOUBLE', dim_prof);
    netcdf.putAtt(ncid, vid, 'long_name', 'spectral degrees of freedom');
    netcdf.putVar(ncid, vid, dof_spec_all);

    % --- Per-profile groups ---
    for pi = 1:n_profiles
        grp_name = sprintf('profile_%03d', pi);
        gid = netcdf.defGrp(ncid, grp_name);

        eps_mat   = eps_all{pi};      % (n_shear, n_est)
        K_max_mat = K_max_all{pi};
        FM_mat    = FM_all{pi};
        mad_mat   = mad_all{pi};
        meth_mat  = method_all{pi};
        nu_vec    = nu_all{pi};       % (n_est, 1)
        spd_vec   = speed_all{pi};
        P_vec     = P_mean_all{pi};
        T_vec     = T_mean_all{pi};
        t_vec     = t_mean_all{pi};
        K_mat     = K_all{pi};        % (n_freq, n_est) or (n_freq, 1)
        F_mat     = F_all{pi};
        sh_clean  = spec_sh_all{pi};  % (n_freq, n_sh, n_sh, n_est)

        n_shear = size(eps_mat, 1);
        n_est   = size(eps_mat, 2);
        n_freq  = size(K_mat, 1);

        % Dimensions
        d_est   = netcdf.defDim(gid, 'estimate', n_est);
        d_freq  = netcdf.defDim(gid, 'frequency', n_freq);
        d_shear = netcdf.defDim(gid, 'shear_probe', n_shear);

        % --- Coordinate variables ---
        vid = netcdf.defVar(gid, 'P_mean', 'NC_DOUBLE', d_est);
        netcdf.putAtt(gid, vid, 'long_name', 'mean pressure per estimate');
        netcdf.putAtt(gid, vid, 'units', 'dbar');
        netcdf.putAtt(gid, vid, 'standard_name', 'sea_water_pressure');
        netcdf.putVar(gid, vid, P_vec(:)');

        vid = netcdf.defVar(gid, 'T_mean', 'NC_DOUBLE', d_est);
        netcdf.putAtt(gid, vid, 'long_name', 'mean temperature per estimate');
        netcdf.putAtt(gid, vid, 'units', 'degC');
        netcdf.putAtt(gid, vid, 'standard_name', 'sea_water_temperature');
        netcdf.putVar(gid, vid, T_vec(:)');

        vid = netcdf.defVar(gid, 't_mean', 'NC_DOUBLE', d_est);
        netcdf.putAtt(gid, vid, 'long_name', 'mean time per estimate');
        netcdf.putAtt(gid, vid, 'units', 'seconds since 1970-01-01T00:00:00Z');
        netcdf.putAtt(gid, vid, 'calendar', 'standard');
        netcdf.putVar(gid, vid, t_vec(:)');

        % --- Data variables ---
        vid = netcdf.defVar(gid, 'epsilon', 'NC_DOUBLE', [d_shear, d_est]);
        netcdf.putAtt(gid, vid, 'long_name', 'TKE dissipation rate');
        netcdf.putAtt(gid, vid, 'units', 'W kg-1');
        netcdf.putVar(gid, vid, eps_mat);

        vid = netcdf.defVar(gid, 'K_max', 'NC_DOUBLE', [d_shear, d_est]);
        netcdf.putAtt(gid, vid, 'long_name', 'maximum resolved wavenumber');
        netcdf.putAtt(gid, vid, 'units', 'cpm');
        netcdf.putVar(gid, vid, K_max_mat);

        vid = netcdf.defVar(gid, 'FM', 'NC_DOUBLE', [d_shear, d_est]);
        netcdf.putAtt(gid, vid, 'long_name', 'figure of merit (obs/Nasmyth variance ratio)');
        netcdf.putAtt(gid, vid, 'units', '1');
        netcdf.putVar(gid, vid, FM_mat);

        vid = netcdf.defVar(gid, 'mad', 'NC_DOUBLE', [d_shear, d_est]);
        netcdf.putAtt(gid, vid, 'long_name', 'mean absolute deviation of spectral fit');
        netcdf.putAtt(gid, vid, 'units', '1');
        netcdf.putVar(gid, vid, mad_mat);

        vid = netcdf.defVar(gid, 'method', 'NC_INT', [d_shear, d_est]);
        netcdf.putAtt(gid, vid, 'long_name', 'fit method index');
        netcdf.putVar(gid, vid, int32(meth_mat));

        vid = netcdf.defVar(gid, 'nu', 'NC_DOUBLE', d_est);
        netcdf.putAtt(gid, vid, 'long_name', 'kinematic viscosity');
        netcdf.putAtt(gid, vid, 'units', 'm2 s-1');
        netcdf.putVar(gid, vid, nu_vec(:)');

        vid = netcdf.defVar(gid, 'speed', 'NC_DOUBLE', d_est);
        netcdf.putAtt(gid, vid, 'long_name', 'mean profiling speed per estimate');
        netcdf.putAtt(gid, vid, 'units', 'm s-1');
        netcdf.putVar(gid, vid, spd_vec(:)');

        % Wavenumber vector (n_freq x n_est — varies with speed)
        vid = netcdf.defVar(gid, 'K', 'NC_DOUBLE', [d_freq, d_est]);
        netcdf.putAtt(gid, vid, 'long_name', 'wavenumber');
        netcdf.putAtt(gid, vid, 'units', 'cpm');
        netcdf.putVar(gid, vid, K_mat);

        % Frequency vector
        vid = netcdf.defVar(gid, 'F', 'NC_DOUBLE', [d_freq, d_est]);
        netcdf.putAtt(gid, vid, 'long_name', 'frequency');
        netcdf.putAtt(gid, vid, 'units', 'Hz');
        netcdf.putVar(gid, vid, F_mat);

        % Cleaned shear spectra: ODAS returns (n_freq, n_shear, n_shear, n_est)
        if ~isempty(sh_clean)
            d_sh2 = netcdf.defDim(gid, 'shear_probe_2', n_shear);
            vid = netcdf.defVar(gid, 'spec_shear_clean', 'NC_DOUBLE', ...
                [d_freq, d_shear, d_sh2, d_est]);
            netcdf.putAtt(gid, vid, 'long_name', ...
                'Goodman-cleaned shear cross-spectral matrix');
            netcdf.putAtt(gid, vid, 'units', 's-2 cpm-1');
            netcdf.putVar(gid, vid, sh_clean);
        end
    end

    netcdf.close(ncid);
end
