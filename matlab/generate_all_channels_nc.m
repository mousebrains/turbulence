% generate_all_channels_nc.m
% Generate NetCDF files containing ALL odas_p2mat channel data for
% validation of microstructure-tpw channel conversion against ODAS MATLAB.
%
% The existing generate_odas_p2mat_nc.m only exports P, T1, T2, sh1, sh2,
% Ax, Ay, speed, and W.  This script exports every channel that odas_p2mat
% returns, including scalars (Chlorophyll, Turbidity, DO, JAC_C, JAC_T,
% U_EM, inclinometers, voltage, etc.).
%
% Usage: Run from the turbulence/ directory with ODAS on the path:
%   addpath('odas');
%   run('matlab/generate_all_channels_nc.m');
%
% By default processes both VMP/ and MR/ directories.  Edit the dirs cell
% array below to restrict.
%
% Output: <dir>/*_allch.nc for each .p file
%
% Mar-2026, Claude and Pat Welch, pat@mousebrains.com

dirs = {'VMP', 'MR'};

base_dir = fullfile(fileparts(mfilename('fullpath')), '..');

p_paths   = {};
out_paths = {};

for di = 1:length(dirs)
    d = fullfile(base_dir, dirs{di});
    if ~isfolder(d)
        fprintf('Directory %s not found, skipping\n', d);
        continue;
    end
    p_files = dir(fullfile(d, '*.p'));
    for fi = 1:length(p_files)
        pp = fullfile(p_files(fi).folder, p_files(fi).name);
        [~, stem, ~] = fileparts(pp);
        op = fullfile(p_files(fi).folder, [stem '_allch.nc']);
        p_paths{end+1}   = pp;    %#ok<SAGROW>
        out_paths{end+1} = op;    %#ok<SAGROW>
    end
end

n_files = length(p_paths);
fprintf('Found %d .p files to process\n', n_files);

%% Start a process-based parallel pool
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
        result = odas_p2mat(p_path);
        write_all_channels_nc(out_path, result);
        fprintf('  Saved %s\n', out_path);
    catch ME
        fprintf('  ERROR processing %s: %s\n', p_path, ME.message);
        if ~isempty(ME.stack)
            fprintf('    in %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
        end
    end
end

fprintf('\nDone.\n');


function write_all_channels_nc(out_path, result)
% WRITE_ALL_CHANNELS_NC  Write every odas_p2mat channel to NetCDF4.
%
%   All fields of the result struct are inspected.  Vectors whose length
%   matches n_fast are written on the fast_sample dimension; vectors whose
%   length matches n_slow are written on slow_sample.  Scalar fields are
%   written as global attributes.

    if isfile(out_path)
        delete(out_path);
    end

    ncid = netcdf.create(out_path, bitor(netcdf.getConstant('NETCDF4'), ...
                                         netcdf.getConstant('CLOBBER')));

    fs_fast = result.fs_fast;
    fs_slow = result.fs_slow;
    n_fast  = length(result.t_fast);
    n_slow  = length(result.t_slow);

    % --- Global attributes ---
    NC_GLOBAL = netcdf.getConstant('NC_GLOBAL');
    netcdf.putAtt(ncid, NC_GLOBAL, 'Conventions', 'CF-1.8');
    netcdf.putAtt(ncid, NC_GLOBAL, 'title', ...
        'ODAS odas_p2mat all-channel data for Python validation');
    netcdf.putAtt(ncid, NC_GLOBAL, 'source', ...
        'ODAS MATLAB Library (odas_p2mat)');
    netcdf.putAtt(ncid, NC_GLOBAL, 'history', ...
        [datestr(now, 'yyyy-mm-ddTHH:MM:SS') ' generate_all_channels_nc.m']);
    netcdf.putAtt(ncid, NC_GLOBAL, 'fs_fast', fs_fast);
    netcdf.putAtt(ncid, NC_GLOBAL, 'fs_slow', fs_slow);
    netcdf.putAtt(ncid, NC_GLOBAL, 'n_fast', int32(n_fast));
    netcdf.putAtt(ncid, NC_GLOBAL, 'n_slow', int32(n_slow));

    % --- Dimensions ---
    d_fast = netcdf.defDim(ncid, 'fast_sample', n_fast);
    d_slow = netcdf.defDim(ncid, 'slow_sample', n_slow);

    % --- Write every field in the result struct ---
    fields = fieldnames(result);
    n_written = 0;

    for i = 1:length(fields)
        fname = fields{i};
        val   = result.(fname);

        % Skip non-numeric or multi-dimensional arrays
        if ~isnumeric(val)
            continue;
        end

        % Scalars -> global attributes
        if isscalar(val)
            netcdf.putAtt(ncid, NC_GLOBAL, fname, double(val));
            continue;
        end

        % Determine dimension from vector length
        nv = numel(val);
        if nv == n_fast
            dim = d_fast;
        elseif nv == n_slow
            dim = d_slow;
        else
            % Skip vectors that don't match either dimension
            % (e.g. header arrays, config vectors)
            fprintf('  Skipping %s (length %d, not fast=%d or slow=%d)\n', ...
                    fname, nv, n_fast, n_slow);
            continue;
        end

        vid = netcdf.defVar(ncid, fname, 'NC_DOUBLE', dim);
        netcdf.putAtt(ncid, vid, 'long_name', fname);
        netcdf.putVar(ncid, vid, double(val(:))');
        n_written = n_written + 1;
    end

    netcdf.close(ncid);
    fprintf('  Wrote %d channel variables\n', n_written);
end
