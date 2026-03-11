% generate_for_tests.m
% Generate all NetCDF validation files needed by the Python test suite.
%
% Runs the three generation scripts in sequence:
%   1. generate_odas_p2mat_nc.m      — channel conversion data (*_p2mat.nc)
%   2. generate_validation_nc.m      — epsilon/dissipation data (*_validation.nc)
%   3. generate_scalar_spectra_nc.m  — scalar spectra data (*_scalar_spectra.nc)
%
% Usage: Run from the turbulence/ directory with ODAS on the path:
%   addpath('odas');
%   run('matlab/generate_for_tests.m');
%
% Mar-2026, Claude and Pat Welch, pat@mousebrains.com

script_dir = fileparts(mfilename('fullpath'));

fprintf('=== Step 1/3: Channel conversion (odas_p2mat) ===\n');
run(fullfile(script_dir, 'generate_odas_p2mat_nc.m'));

fprintf('\n=== Step 2/3: Epsilon validation (get_diss_odas) ===\n');
run(fullfile(script_dir, 'generate_validation_nc.m'));

fprintf('\n=== Step 3/3: Scalar spectra validation (get_scalar_spectra_odas) ===\n');
run(fullfile(script_dir, 'generate_scalar_spectra_nc.m'));

fprintf('\n=== All validation files generated ===\n');
