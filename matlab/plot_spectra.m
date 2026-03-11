% Mar-2026, Claude and Pat Welch, pat@mousebrains.com
function plot_spectra(fname, options)
%PLOT_SPECTRA  Plot epsilon and chi spectra for one depth window.
%
%   plot_spectra(fname)
%   plot_spectra(fname, profile=2, P_range=[80 120])
%
%   Creates a two-panel figure showing the shear (epsilon) and temperature
%   gradient (chi) spectra for a single depth window, together with the
%   theoretical model fits.  The depth window is the portion of the chosen
%   profile that falls inside P_range.
%
%   Inputs:
%       fname   - Path to a Rockland .p file
%
%   Name-Value Options:
%       profile        - Profile number (default: 1)
%       P_range        - [P_min P_max] pressure range [dbar] (default: full)
%       fft_length     - FFT segment length [samples] (default: 256)
%       spectrum_model - "batchelor" or "kraichnan" (default: "kraichnan")
%       f_AA           - Anti-aliasing frequency [Hz] (default: 98)

    arguments
        fname          (1,:) char
        options.profile        (1,1) double {mustBePositive, mustBeInteger} = 1
        options.P_range        (1,:) double = []
        options.fft_length     (1,1) double {mustBePositive, mustBeInteger} = 256
        options.spectrum_model (1,1) string {mustBeMember( ...
            options.spectrum_model, ["batchelor","kraichnan"])} = "kraichnan"
        options.f_AA           (1,1) double {mustBePositive} = 98
    end

    %% Paths — add ODAS and matlab/ helpers
    myPath   = fileparts(mfilename("fullpath"));
    rootPath = fullfile(myPath, '..');
    addpath(fullfile(rootPath, 'odas'));
    addpath(myPath);

    %% Read .p file
    fprintf('Reading %s\n', fname);
    d = odas_p2mat(fname);

    %% Detect profiles
    W_slow   = gradient(d.P_slow, 1/d.fs_slow);
    profiles = get_profile(d.P_slow, W_slow, 0.5, 0.3, 'down', 7, d.fs_slow);
    n_prof   = size(profiles, 2);
    fprintf('Found %d profiles\n', n_prof);

    if n_prof == 0
        error('plot_spectra:noProfiles', 'No profiles detected');
    end
    if options.profile > n_prof
        error('plot_spectra:badProfile', ...
            'Requested profile %d but only %d found', options.profile, n_prof);
    end

    %% Extract chosen profile
    ratio  = round(d.fs_fast / d.fs_slow);
    s_slow = profiles(1, options.profile);
    e_slow = profiles(2, options.profile);
    s_fast = (s_slow - 1) * ratio + 1;
    e_fast = e_slow * ratio;

    P_prof  = d.P_fast(s_fast:e_fast);
    sp_prof = d.speed_fast(s_fast:e_fast);
    T_prof  = d.JAC_T(s_slow:e_slow);

    %% Restrict to P_range if given
    if isempty(options.P_range)
        sel = true(size(P_prof));
    else
        sel = P_prof >= options.P_range(1) & P_prof <= options.P_range(2);
    end

    if sum(sel) < 2 * options.fft_length
        error('plot_spectra:tooShort', ...
            'Only %d samples in P_range — need at least %d', ...
            sum(sel), 2 * options.fft_length);
    end

    idx      = find(sel);
    i_start  = idx(1);
    i_end    = idx(end);

    % Corresponding slow indices for T interpolation
    i_s_slow = max(1, floor(i_start / ratio));
    i_s_end  = min(numel(T_prof), ceil(i_end / ratio));

    fprintf('Profile %d, P = %.1f–%.1f dbar (%d fast samples)\n', ...
        options.profile, P_prof(i_start), P_prof(i_end), i_end - i_start + 1);

    %% Compute epsilon (shear spectra)
    fft_len    = options.fft_length;
    diss_len   = 2 * fft_len;   % single window for spectral display
    olap       = 0;

    sh1_seg = d.sh1(s_fast + i_start - 1 : s_fast + i_end - 1);
    sh2_seg = d.sh2(s_fast + i_start - 1 : s_fast + i_end - 1);

    info_eps = struct( ...
        'fft_length',  fft_len, ...
        'diss_length', diss_len, ...
        'overlap',     floor(diss_len / 2), ...
        'fs_fast',     d.fs_fast, ...
        'fs_slow',     d.fs_slow, ...
        'speed',       sp_prof(i_start:i_end), ...
        'T',           d.T1_fast(s_fast + i_start - 1 : s_fast + i_end - 1), ...
        'P',           P_prof(i_start:i_end), ...
        'f_AA',        options.f_AA, ...
        'goodman',     true);

    diss = get_diss_odas([sh1_seg, sh2_seg], ...
        [d.Ax(s_fast+i_start-1:s_fast+i_end-1), ...
         d.Ay(s_fast+i_start-1:s_fast+i_end-1)], info_eps);

    %% Pick the spectral window closest to the midpoint pressure
    if ~isempty(options.P_range)
        P_mid = mean(options.P_range);
    else
        P_mid = mean(P_prof);
    end
    [~, w_eps] = min(abs(diss.P - P_mid));

    %% Compute chi (temperature gradient spectra)
    gradT1_seg = d.gradT1(s_fast + i_start - 1 : s_fast + i_end - 1);
    gradT2_seg = d.gradT2(s_fast + i_start - 1 : s_fast + i_end - 1);
    P_seg      = P_prof(i_start:i_end);
    sp_seg     = sp_prof(i_start:i_end);
    T_seg      = T_prof(i_s_slow:i_s_end);

    % Method 1: chi from epsilon
    results_m1 = get_chi([gradT1_seg, gradT2_seg], P_seg, T_seg, sp_seg, ...
        d.fs_fast, ...
        method=1, epsilon=diss.e, ...
        fft_length=fft_len, diss_length=diss_len, ...
        overlap=floor(diss_len/2), ...
        spectrum_model=options.spectrum_model, ...
        f_AA=options.f_AA);

    % Method 2: MLE fit
    results_m2 = get_chi([gradT1_seg, gradT2_seg], P_seg, T_seg, sp_seg, ...
        d.fs_fast, ...
        method=2, ...
        fft_length=fft_len, diss_length=diss_len, ...
        overlap=floor(diss_len/2), ...
        spectrum_model=options.spectrum_model, ...
        f_AA=options.f_AA);

    % Pick window closest to midpoint pressure
    [~, w_chi] = min(abs(results_m2.P_mean - P_mid));

    %% FP07 noise floor (use the estimate_noise from chi_method1)
    W_mean = results_m2.speed(w_chi);
    T_mean = results_m2.T_mean(w_chi);
    F_chi  = results_m2.F;
    K_chi  = results_m2.K(:, w_chi);
    tau0   = fp07_time_constant(W_mean);
    H2     = fp07_transfer(F_chi, tau0);

    diff_gain = 0.94;
    noise_K = estimate_noise_spectra(F_chi, T_mean, W_mean, d.fs_fast, diff_gain);

    %% Epsilon spectra: AA wavenumber
    K_AA_eps = (0.9 * options.f_AA) / diss.speed(w_eps);

    %% Create figure
    [~, base_name, ~] = fileparts(fname);
    fig = figure('Name', sprintf('Spectra — Profile %d', options.profile), ...
        'Units', 'normalized', 'Position', [0.1 0.15 0.85 0.45]);
    sgtitle(sprintf('%s — Profile %d — P = %.1f–%.1f dbar', ...
        strrep(base_name, '_', '\_'), options.profile, ...
        P_prof(i_start), P_prof(i_end)), 'FontSize', 12);

    %% Left panel — Epsilon (shear) spectra
    ax1 = subplot(1, 2, 1);
    hold on;
    colors_sh = {'b', 'r'};
    n_shear = size(diss.e, 1);

    for si = 1:n_shear
        K_eps = diss.K(:, w_eps);
        % Shear spectrum (diagonal of cleaned cross-spectrum)
        sh_spec = real(diss.sh_clean(:, si, si, w_eps));
        nas     = diss.Nasmyth_spec(:, si, w_eps);

        valid = K_eps > 0;
        loglog(K_eps(valid), sh_spec(valid), colors_sh{si}, ...
            'LineWidth', 0.8, 'DisplayName', sprintf('sh%d', si));
        loglog(K_eps(valid), nas(valid), colors_sh{si}, ...
            'LineWidth', 1.2, 'LineStyle', '--', ...
            'DisplayName', sprintf('Nasmyth ε=%.1e', diss.e(si, w_eps)));
    end

    % AA line
    xline(K_AA_eps, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5, ...
        'HandleVisibility', 'off');

    hold off;
    set(ax1, 'XScale', 'log', 'YScale', 'log');
    xlim([0.5, 300]);
    xlabel('Wavenumber [cpm]');
    ylabel('\Phi_{shear} [s^{-2} cpm^{-1}]');
    legend('Location', 'southwest', 'FontSize', 7);
    grid on; grid minor;
    title(sprintf('ε spectra — W = %.2f m/s', diss.speed(w_eps)), ...
        'FontSize', 10);

    %% Right panel — Chi (temperature gradient) spectra
    ax2 = subplot(1, 2, 2);
    hold on;
    colors_chi = {'r', [1 0.5 0]};
    n_probes_chi = size(results_m2.chi, 1);
    probe_names = {'T1', 'T2'};

    K_AA_chi = (0.9 * options.f_AA) / W_mean;

    for ci = 1:n_probes_chi
        c = colors_chi{ci};

        % Observed spectrum
        obs = results_m2.spec_gradT(:, ci, w_chi);
        valid = K_chi > 0 & obs > 0 & isfinite(obs);
        loglog(K_chi(valid), obs(valid), 'Color', c, ...
            'LineWidth', 0.8, 'DisplayName', probe_names{ci});

        % Method 1 fit (from epsilon) — dashed
        m1_spec = results_m1.spec_batch(:, ci, w_chi) .* H2;
        chi_m1  = results_m1.chi(ci, w_chi);
        vm1 = K_chi > 0 & m1_spec > 0 & isfinite(m1_spec);
        if any(vm1) && isfinite(chi_m1)
            loglog(K_chi(vm1), m1_spec(vm1), 'Color', c, ...
                'LineWidth', 1.2, 'LineStyle', '--', ...
                'DisplayName', sprintf('M1 χ=%.1e', chi_m1));
        end

        % Method 2 MLE fit — dash-dot
        m2_spec = results_m2.spec_batch(:, ci, w_chi) .* H2;
        chi_m2  = results_m2.chi(ci, w_chi);
        vm2 = K_chi > 0 & m2_spec > 0 & isfinite(m2_spec);
        if any(vm2) && isfinite(chi_m2)
            loglog(K_chi(vm2), m2_spec(vm2), 'Color', c, ...
                'LineWidth', 1.2, 'LineStyle', '-.', ...
                'DisplayName', sprintf('M2 χ=%.1e', chi_m2));
        end
    end

    % Noise floor
    valid_n = K_chi > 0 & noise_K > 0;
    loglog(K_chi(valid_n), noise_K(valid_n), ':', 'Color', [0.5 0.5 0.5], ...
        'LineWidth', 0.6, 'DisplayName', 'Noise');

    % AA line
    xline(K_AA_chi, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5, ...
        'HandleVisibility', 'off');

    hold off;
    set(ax2, 'XScale', 'log', 'YScale', 'log');
    xlim([0.5, 300]);
    ylim([1e-11, inf]);
    xlabel('Wavenumber [cpm]');
    ylabel('\Phi_T [(K/m)^2 cpm^{-1}]');
    legend('Location', 'southwest', 'FontSize', 7);
    grid on; grid minor;
    title('χ spectra  (-- M1 from ε,  -· M2 MLE)', 'FontSize', 10);

    fprintf('Done. Window at P ≈ %.1f dbar\n', results_m2.P_mean(w_chi));
end


function noise_K = estimate_noise_spectra(F, T_mean, speed, fs, diff_gain)
%ESTIMATE_NOISE_SPECTRA  FP07 electronics noise in wavenumber units.
%   Simplified noise model from RSI TN-040 (same as chi_method1.m).

    E_n  = 4e-9;    % first-stage voltage noise [V/sqrt(Hz)]
    fc   = 18.7;    % first-stage flicker knee [Hz]
    E_n2 = 8e-9;    % second-stage voltage noise [V/sqrt(Hz)]
    fc_2 = 42;      % second-stage flicker knee [Hz]
    R_0  = 3000;    % thermistor resistance [Ohm]
    gain = 6;       % first-stage gain
    f_AA = 110;     % AA filter cutoff [Hz]
    adc_fs   = 4.096;
    adc_bits = 16;
    gamma_RSI = 3;
    T_K  = 295;     % operating temp [K]
    K_B  = 1.382e-23;

    delta_s = adc_fs / 2^adc_bits;
    fN = fs / 2;

    % Stage 1: amplifier + Johnson noise
    V1 = 2 * E_n^2 .* sqrt(1 + (F./fc).^2) ./ (F./fc);
    V1(~isfinite(V1)) = max(V1(isfinite(V1)));
    phi_R = 4 * K_B * R_0 * T_K;
    Noise_1 = gain^2 * (V1 + phi_R);

    % Stage 2: pre-emphasis + second amplifier
    G_2 = 1 + (2*pi*diff_gain.*F).^2;
    V2 = 2 * E_n2^2 .* sqrt(1 + (F./fc_2).^2) ./ (F./fc_2);
    V2(~isfinite(V2)) = max(V2(isfinite(V2)));
    Noise_2 = G_2 .* (Noise_1 + V2);

    % Anti-aliasing filter (two-stage 4th-order Butterworth)
    G_AA = 1 ./ (1 + (F./f_AA).^8).^2;
    Noise_3 = Noise_2 .* G_AA;

    % ADC quantization noise
    Noise_4 = Noise_3 + gamma_RSI * delta_s^2 / (12 * fN);

    % Convert to counts^2/Hz
    Noise_counts = Noise_4 ./ delta_s^2;

    % High-pass from pre-emphasis deconvolution
    w = 2*pi*diff_gain.*F;
    G_HP = (1/diff_gain)^2 .* w.^2 ./ (1 + w.^2);
    Noise_counts = Noise_counts .* G_HP;

    % Convert to physical units
    T_kelvin = T_mean + 273.15;
    e_b = 0.68;
    b = 1.0;
    beta_1 = 3000.0;
    eta = (b/2) * 2^adc_bits * gain * e_b / adc_fs;
    R_ratio = 1.0;
    scale_factor = T_kelvin^2 * (1 + R_ratio)^2 / (2 * eta * beta_1 * R_ratio);

    noise_f = Noise_counts .* scale_factor^2;  % (K/m)^2/Hz
    noise_K = noise_f .* speed;                % (K/m)^2/cpm
end
