% Mar-2026, Claude and Pat Welch, pat@mousebrains.com
function result = chi_method1(spec_obs, K, epsilon, nu, speed, options)
%CHI_METHOD1  Compute chi given known epsilon (Dillon & Caldwell 1980).
%
%   result = chi_method1(spec_obs, K, epsilon, nu, speed)
%   result = chi_method1(..., T_mean=10, f_AA=98, fs=512, ...)
%
%   Given the observed temperature gradient spectrum and epsilon from shear
%   probes, computes chi by:
%     1. Computing kB from epsilon
%     2. Integrating the observed spectrum up to the noise floor / AA limit
%     3. Correcting for FP07 rolloff and unresolved variance beyond K_max
%
%   Inputs:
%       spec_obs - Observed temperature gradient spectrum [(K/m)^2/cpm]
%       K        - Wavenumber vector [cpm]
%       epsilon  - TKE dissipation rate [W/kg]
%       nu       - Kinematic viscosity [m^2/s]
%       speed    - Profiling speed [m/s]
%
%   Name-Value Options:
%       T_mean         - Mean temperature [deg C] (default: 10)
%       f_AA           - Anti-aliasing frequency [Hz] (default: 98)
%       fs             - Sampling rate [Hz] (default: 512)
%       diff_gain      - Pre-emphasis differentiator gain [s] (default: 0.94)
%       spectrum_model - "kraichnan" (default) or "batchelor"
%       fp07_model     - "single_pole" (default) or "double_pole"
%       kappa_T        - Thermal diffusivity [m^2/s] (default: 1.4e-7)
%       noise_K        - Pre-computed noise spectrum [(K/m)^2/cpm]; if empty,
%                        uses gradT_noise_odas via the ODAS library
%
%   Output:
%       result - Struct with fields:
%           chi         - Thermal variance dissipation rate [K^2/s]
%           kB          - Batchelor wavenumber [cpm]
%           K_max       - Maximum integration wavenumber [cpm]
%           spec_batch  - Fitted Batchelor spectrum [(K/m)^2/cpm]
%           fom         - Figure of merit (obs/model variance ratio)
%           K_max_ratio - K_max / kB (spectral resolution)
%
%   References:
%       Dillon & Caldwell, 1980: J. Geophys. Res., 85, 1910-1916.

    arguments
        spec_obs       (:,1) double
        K              (:,1) double
        epsilon        (1,1) double {mustBePositive}
        nu             (1,1) double {mustBePositive}
        speed          (1,1) double {mustBePositive}
        options.T_mean         (1,1) double = 10
        options.f_AA           (1,1) double {mustBePositive} = 98
        options.fs             (1,1) double {mustBePositive} = 512
        options.diff_gain      (1,1) double {mustBePositive} = 0.94
        options.spectrum_model (1,1) string {mustBeMember(options.spectrum_model, ...
            ["batchelor", "kraichnan"])} = "kraichnan"
        options.fp07_model     (1,1) string {mustBeMember(options.fp07_model, ...
            ["single_pole", "double_pole"])} = "single_pole"
        options.kappa_T        (1,1) double {mustBePositive} = 1.4e-7
        options.noise_K        (:,1) double = double.empty(0,1)
    end

    kappa_T = options.kappa_T;
    grad_func = spectrum_function(options.spectrum_model);

    % Batchelor wavenumber from epsilon
    kB = batchelor_wavenumber(epsilon, nu, kappa_T=kappa_T);

    % Default NaN result
    nan_result = struct("chi", NaN, "kB", kB, "K_max", NaN, ...
        "spec_batch", zeros(size(K)), "fom", NaN, "K_max_ratio", NaN);

    if kB < 1
        result = nan_result;
        return
    end

    % FP07 transfer function
    tau0 = fp07_time_constant(speed);
    F = K .* speed;
    H2 = fp07_transfer(F, tau0, model=options.fp07_model);

    % Noise spectrum
    if isempty(options.noise_K)
        noise_K = estimate_noise(F, options.T_mean, speed, ...
            options.fs, options.diff_gain);
    else
        noise_K = options.noise_K;
    end

    % Integration limits: above noise and within AA
    K_AA = options.f_AA / speed;
    above_noise = spec_obs > 2 * noise_K;
    valid = above_noise & (K > 0) & (K <= K_AA);

    if sum(valid) < 3
        valid = (K > 0) & (K <= K_AA);
    end
    if sum(valid) < 3
        result = nan_result;
        return
    end

    valid_idx = find(valid);
    K_max = K(valid_idx(end));

    % Integrate observed spectrum
    obs_var = trapz(K(valid), spec_obs(valid));

    chi_trial = 6 * kappa_T * obs_var;
    if chi_trial <= 0
        result = nan_result;
        return
    end

    % Correction for FP07 rolloff and unresolved variance
    K_fine = linspace(0, max(K_max * 5, kB * 5), 10000)';
    K_fine(1) = K_fine(2) * 0.01;
    spec_batch_fine = grad_func(K_fine, kB, chi_trial);
    V_total = trapz(K_fine, spec_batch_fine);

    F_fine = K_fine .* speed;
    H2_fine = fp07_transfer(F_fine, tau0, model=options.fp07_model);
    mask_resolved = K_fine <= K_max;
    V_resolved = trapz(K_fine(mask_resolved), ...
        spec_batch_fine(mask_resolved) .* H2_fine(mask_resolved));

    if V_resolved <= 0 || V_total <= 0
        result = nan_result;
        return
    end

    correction = V_total / V_resolved;
    chi = 6 * kappa_T * obs_var * correction;

    % Fitted Batchelor spectrum for output
    spec_batch = grad_func(K, kB, chi);

    % Figure of merit
    mod_v = trapz(K(valid), spec_batch(valid));
    if mod_v > 0 && isfinite(chi)
        fom = obs_var / mod_v;
    else
        fom = NaN;
    end

    K_max_ratio = K_max / kB;

    result = struct("chi", chi, "kB", kB, "K_max", K_max, ...
        "spec_batch", spec_batch, "fom", fom, "K_max_ratio", K_max_ratio);
end


function grad_func = spectrum_function(model)
    switch model
        case "batchelor"
            grad_func = @batchelor_gradT;
        case "kraichnan"
            grad_func = @kraichnan_gradT;
    end
end


function noise_K = estimate_noise(F, T_mean, speed, fs, diff_gain)
%ESTIMATE_NOISE  Simplified FP07 electronics noise in wavenumber units.
%
%   This is a standalone approximation of the noise model from
%   gradT_noise_odas.m / RSI TN-040.  If the ODAS library is on your path,
%   prefer using gradT_noise_odas directly for accurate, instrument-specific
%   noise estimates.

    % Default electronics parameters (RSI TN-040)
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

    % Convert to physical units using simplified scale factor
    T_kelvin = T_mean + 273.15;
    e_b = 0.68;
    b = 1.0;
    beta_1 = 3000.0;
    eta = (b/2) * 2^adc_bits * gain * e_b / adc_fs;
    R_ratio = 1.0;
    scale_factor = T_kelvin^2 * (1 + R_ratio)^2 / (2 * eta * beta_1 * R_ratio);

    noise_f = Noise_counts .* scale_factor^2;  % (K/m)^2/Hz

    % Convert frequency spectrum to wavenumber spectrum
    noise_K = noise_f .* speed;  % (K/m)^2/cpm
end
