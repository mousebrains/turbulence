% generate_odas_p2mat_nc.m
% Generate CF-compliant NetCDF files containing odas_p2mat channel data
% for validation of rsi-python PFile conversion against ODAS MATLAB Library.
%
% Usage: Run from the turbulence/ directory with ODAS on the path:
%   addpath('odas');
%   run('matlab/generate_odas_p2mat_nc.m');
%
% Output: VMP/*_p2mat.nc for each .p file
%
% Each file contains sampling rates as global attributes and all converted
% channels (pressure, temperature, shear, accelerometers, speed, fall rate)
% as root-level variables with CF-compliant metadata.
%
% Mar-2026, Claude and Pat Welch, pat@mousebrains.com

vmp_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'VMP');
p_files = dir(fullfile(vmp_dir, '*.p'));
n_files = length(p_files);

fprintf('Found %d .p files in %s\n', n_files, vmp_dir);

% Pre-build path arrays for parfor
p_paths   = cell(n_files, 1);
out_paths = cell(n_files, 1);
for fi = 1:n_files
    p_paths{fi}   = fullfile(p_files(fi).folder, p_files(fi).name);
    [~, stem, ~]   = fileparts(p_paths{fi});
    out_paths{fi}  = fullfile(p_files(fi).folder, [stem '_p2mat.nc']);
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
        result = odas_p2mat(p_path);

        parsave_p2mat_nc(out_path, result);

        fprintf('  Saved %s\n', out_path);
    catch ME
        fprintf('  ERROR processing %s: %s\n', p_path, ME.message);
        if ~isempty(ME.stack)
            fprintf('  %s\n', ME.stack(1).name);
        end
    end
end

fprintf('\nDone.\n');


function parsave_p2mat_nc(out_path, result)
% PARSAVE_P2MAT_NC  Write odas_p2mat channel data to CF-compliant NetCDF4.
%   Root group contains global attributes and all channel variables.

    if isfile(out_path)
        delete(out_path);
    end

    ncid = netcdf.create(out_path, bitor(netcdf.getConstant('NETCDF4'), ...
                                         netcdf.getConstant('CLOBBER')));

    fs_fast = result.fs_fast;
    fs_slow = result.fs_slow;

    % --- Global attributes ---
    netcdf.putAtt(ncid, netcdf.getConstant('NC_GLOBAL'), ...
        'Conventions', 'CF-1.8');
    netcdf.putAtt(ncid, netcdf.getConstant('NC_GLOBAL'), ...
        'title', 'ODAS odas_p2mat channel conversion data');
    netcdf.putAtt(ncid, netcdf.getConstant('NC_GLOBAL'), ...
        'institution', 'Oregon State University');
    netcdf.putAtt(ncid, netcdf.getConstant('NC_GLOBAL'), ...
        'source', 'ODAS MATLAB Library (odas_p2mat)');
    netcdf.putAtt(ncid, netcdf.getConstant('NC_GLOBAL'), ...
        'history', [datestr(now, 'yyyy-mm-ddTHH:MM:SS') ...
        ' generate_odas_p2mat_nc.m']);
    netcdf.putAtt(ncid, netcdf.getConstant('NC_GLOBAL'), ...
        'fs_fast', fs_fast);
    netcdf.putAtt(ncid, netcdf.getConstant('NC_GLOBAL'), ...
        'fs_slow', fs_slow);

    % --- Dimensions ---
    n_fast = length(result.t_fast);
    n_slow = length(result.t_slow);

    d_fast = netcdf.defDim(ncid, 'fast_sample', n_fast);
    d_slow = netcdf.defDim(ncid, 'slow_sample', n_slow);

    % --- Time vectors ---
    vid = netcdf.defVar(ncid, 't_fast', 'NC_DOUBLE', d_fast);
    netcdf.putAtt(ncid, vid, 'long_name', 'time at fast sampling rate');
    netcdf.putAtt(ncid, vid, 'units', 'seconds since 1970-01-01T00:00:00Z');
    netcdf.putAtt(ncid, vid, 'calendar', 'standard');
    netcdf.putVar(ncid, vid, result.t_fast(:)');

    vid = netcdf.defVar(ncid, 't_slow', 'NC_DOUBLE', d_slow);
    netcdf.putAtt(ncid, vid, 'long_name', 'time at slow sampling rate');
    netcdf.putAtt(ncid, vid, 'units', 'seconds since 1970-01-01T00:00:00Z');
    netcdf.putAtt(ncid, vid, 'calendar', 'standard');
    netcdf.putVar(ncid, vid, result.t_slow(:)');

    % --- Slow-rate channels ---
    vid = netcdf.defVar(ncid, 'P_slow', 'NC_DOUBLE', d_slow);
    netcdf.putAtt(ncid, vid, 'long_name', 'pressure at slow sampling rate');
    netcdf.putAtt(ncid, vid, 'units', 'dbar');
    netcdf.putAtt(ncid, vid, 'standard_name', 'sea_water_pressure');
    netcdf.putVar(ncid, vid, result.P_slow(:)');

    vid = netcdf.defVar(ncid, 'T1_slow', 'NC_DOUBLE', d_slow);
    netcdf.putAtt(ncid, vid, 'long_name', 'temperature T1 at slow sampling rate');
    netcdf.putAtt(ncid, vid, 'units', 'degC');
    netcdf.putAtt(ncid, vid, 'standard_name', 'sea_water_temperature');
    netcdf.putVar(ncid, vid, result.T1_slow(:)');

    vid = netcdf.defVar(ncid, 'T2_slow', 'NC_DOUBLE', d_slow);
    netcdf.putAtt(ncid, vid, 'long_name', 'temperature T2 at slow sampling rate');
    netcdf.putAtt(ncid, vid, 'units', 'degC');
    netcdf.putAtt(ncid, vid, 'standard_name', 'sea_water_temperature');
    netcdf.putVar(ncid, vid, result.T2_slow(:)');

    vid = netcdf.defVar(ncid, 'W_slow', 'NC_DOUBLE', d_slow);
    netcdf.putAtt(ncid, vid, 'long_name', 'vertical velocity (fall rate) at slow sampling rate');
    netcdf.putAtt(ncid, vid, 'units', 'm s-1');
    netcdf.putVar(ncid, vid, result.W_slow(:)');

    vid = netcdf.defVar(ncid, 'speed_slow', 'NC_DOUBLE', d_slow);
    netcdf.putAtt(ncid, vid, 'long_name', 'profiling speed at slow sampling rate');
    netcdf.putAtt(ncid, vid, 'units', 'm s-1');
    netcdf.putVar(ncid, vid, result.speed_slow(:)');

    % --- Fast-rate channels ---
    vid = netcdf.defVar(ncid, 'P_fast', 'NC_DOUBLE', d_fast);
    netcdf.putAtt(ncid, vid, 'long_name', 'pressure at fast sampling rate');
    netcdf.putAtt(ncid, vid, 'units', 'dbar');
    netcdf.putAtt(ncid, vid, 'standard_name', 'sea_water_pressure');
    netcdf.putVar(ncid, vid, result.P_fast(:)');

    vid = netcdf.defVar(ncid, 'T1_fast', 'NC_DOUBLE', d_fast);
    netcdf.putAtt(ncid, vid, 'long_name', 'temperature T1 at fast sampling rate');
    netcdf.putAtt(ncid, vid, 'units', 'degC');
    netcdf.putAtt(ncid, vid, 'standard_name', 'sea_water_temperature');
    netcdf.putVar(ncid, vid, result.T1_fast(:)');

    vid = netcdf.defVar(ncid, 'T2_fast', 'NC_DOUBLE', d_fast);
    netcdf.putAtt(ncid, vid, 'long_name', 'temperature T2 at fast sampling rate');
    netcdf.putAtt(ncid, vid, 'units', 'degC');
    netcdf.putAtt(ncid, vid, 'standard_name', 'sea_water_temperature');
    netcdf.putVar(ncid, vid, result.T2_fast(:)');

    vid = netcdf.defVar(ncid, 'sh1', 'NC_DOUBLE', d_fast);
    netcdf.putAtt(ncid, vid, 'long_name', 'shear probe 1 (du/dz, divided by speed^2)');
    netcdf.putAtt(ncid, vid, 'units', 's-1');
    netcdf.putVar(ncid, vid, result.sh1(:)');

    vid = netcdf.defVar(ncid, 'sh2', 'NC_DOUBLE', d_fast);
    netcdf.putAtt(ncid, vid, 'long_name', 'shear probe 2 (du/dz, divided by speed^2)');
    netcdf.putAtt(ncid, vid, 'units', 's-1');
    netcdf.putVar(ncid, vid, result.sh2(:)');

    vid = netcdf.defVar(ncid, 'Ax', 'NC_DOUBLE', d_fast);
    netcdf.putAtt(ncid, vid, 'long_name', 'accelerometer X');
    netcdf.putAtt(ncid, vid, 'units', 'm s-2');
    netcdf.putVar(ncid, vid, result.Ax(:)');

    vid = netcdf.defVar(ncid, 'Ay', 'NC_DOUBLE', d_fast);
    netcdf.putAtt(ncid, vid, 'long_name', 'accelerometer Y');
    netcdf.putAtt(ncid, vid, 'units', 'm s-2');
    netcdf.putVar(ncid, vid, result.Ay(:)');

    vid = netcdf.defVar(ncid, 'speed_fast', 'NC_DOUBLE', d_fast);
    netcdf.putAtt(ncid, vid, 'long_name', 'profiling speed at fast sampling rate');
    netcdf.putAtt(ncid, vid, 'units', 'm s-1');
    netcdf.putVar(ncid, vid, result.speed_fast(:)');

    netcdf.close(ncid);
end
