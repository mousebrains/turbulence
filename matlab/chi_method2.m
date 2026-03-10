% Mar-2026, Claude and Pat Welch, pat@mousebrains.com
function result = chi_method2(spec_obs, K, nu, speed, options)
%CHI_METHOD2  Compute chi via MLE Batchelor spectrum fitting (no epsilon required).
%
%   result = chi_method2(spec_obs, K, nu, speed)
%   result = chi_method2(..., T_mean=10, f_AA=98, fs=512, ...)
%
%   Maximum likelihood spectral fitting to estimate kB from the observed
%   temperature gradient spectrum (Ruddick et al. 2000).  Epsilon and chi
%   are recovered from the fitted kB.
%
%   Uses a coarse-to-fine grid search over kB, minimizing the negative
%   log-likelihood for chi-squared distributed spectral estimates:
%       NLL = sum( log(S_model) + S_obs / S_model )
%
%   Inputs:
%       spec_obs - Observed temperature gradient spectrum [(K/m)^2/cpm]
%       K        - Wavenumber vector [cpm]
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
%       noise_K        - Pre-computed noise spectrum [(K/m)^2/cpm]
%       n_iterations   - Number of refinement iterations (default: 3)
%
%   Output:
%       result - Struct with fields:
%           chi         - Thermal variance dissipation rate [K^2/s]
%           epsilon     - TKE dissipation rate recovered from kB [W/kg]
%           kB          - Best-fit Batchelor wavenumber [cpm]
%           K_max       - Maximum fit wavenumber [cpm]
%           spec_batch  - Fitted Batchelor spectrum [(K/m)^2/cpm]
%           fom         - Figure of merit (obs/model variance ratio)
%           K_max_ratio - K_max / kB (spectral resolution)
%
%   References:
%       Ruddick, Anis & Thompson, 2000: J. Atmos. Oceanic Technol., 17, 1541-1555.
%       Peterson & Fer, 2014: Methods in Oceanography, 10, 44-69.

    arguments
        spec_obs       (:,1) double
        K              (:,1) double
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
        options.n_iterations   (1,1) double {mustBePositive, mustBeInteger} = 3
    end

    kappa_T = options.kappa_T;
    grad_func = spectrum_function(options.spectrum_model);

    nan_result = struct("chi", NaN, "epsilon", NaN, "kB", NaN, ...
        "K_max", NaN, "spec_batch", zeros(size(K)), ...
        "fom", NaN, "K_max_ratio", NaN);

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

    % Fitting range
    K_AA = options.f_AA / speed;
    above_noise = spec_obs > 2 * noise_K;
    fit_mask = above_noise & (K > 0) & (K <= K_AA);
    if sum(fit_mask) < 6
        fit_mask = (K > 0) & (K <= K_AA);
    end
    if sum(fit_mask) < 6
        result = nan_result;
        return
    end

    fit_idx = find(fit_mask);
    K_fit = K(fit_idx);
    spec_fit = spec_obs(fit_idx);
    H2_fit = H2(fit_idx);
    noise_fit = noise_K(fit_idx);
    K_max_fit = K_fit(end);

    % Initial chi estimate from observed spectrum minus noise
    spec_above = max(spec_obs - noise_K, 0);
    valid_init = (K > 0) & (K <= K_AA);
    chi_obs = 6 * kappa_T * trapz(K(valid_init), spec_above(valid_init));
    chi_obs = max(chi_obs, 1e-14);

    % Iterative fitting (Peterson & Fer 2014)
    kB_best = NaN;
    for iter = 1:options.n_iterations
        % MLE grid search for kB
        [kB_best, nll_best] = mle_grid_search( ...
            K_fit, spec_fit, H2_fit, noise_fit, chi_obs, grad_func);

        if isnan(kB_best)
            result = nan_result;
            return
        end

        % Refine integration limits
        k_star = 0.04 * kB_best * sqrt(kappa_T / nu);
        k_l = max(K(2), 3 * k_star);
        k_u = K_max_fit;

        refined_mask = (K >= k_l) & (K <= k_u) & (K > 0);
        if sum(refined_mask) < 3
            break
        end

        % Update chi_obs with refined limits + unresolved variance
        obs_refined = trapz(K(refined_mask), ...
            max(spec_obs(refined_mask) - noise_K(refined_mask), 0));

        % Add unresolved variance below k_l and above k_u
        K_fine = linspace(K(2) * 0.01, kB_best * 5, 10000)';
        spec_fine = grad_func(K_fine, kB_best, chi_obs);
        below = K_fine < k_l;
        above = K_fine > k_u;
        unresolved = 0;
        if any(below)
            unresolved = unresolved + trapz(K_fine(below), spec_fine(below));
        end
        if any(above)
            unresolved = unresolved + trapz(K_fine(above), spec_fine(above));
        end

        chi_obs = 6 * kappa_T * (obs_refined + unresolved);
        chi_obs = max(chi_obs, 1e-14);
    end

    % Recover epsilon from best-fit kB
    epsilon = (2*pi * kB_best)^4 * nu * kappa_T^2;

    % Final chi by integrating full Batchelor spectrum
    K_fine = linspace(K(2) * 0.01, kB_best * 5, 10000)';
    spec_fine = grad_func(K_fine, kB_best, chi_obs);
    chi = 6 * kappa_T * trapz(K_fine, spec_fine);

    % Fitted spectrum for output
    spec_batch = grad_func(K, kB_best, chi);

    % Figure of merit
    mod_v = trapz(K_fit, spec_batch(fit_idx));
    obs_v = trapz(K_fit, spec_fit);
    if mod_v > 0 && isfinite(chi)
        fom = obs_v / mod_v;
    else
        fom = NaN;
    end

    K_max_ratio = K_max_fit / kB_best;

    result = struct("chi", chi, "epsilon", epsilon, "kB", kB_best, ...
        "K_max", K_max_fit, "spec_batch", spec_batch, ...
        "fom", fom, "K_max_ratio", K_max_ratio);
end


function [kB_best, nll_best] = mle_grid_search(K_fit, spec_fit, H2_fit, noise_fit, chi_obs, grad_func)
%MLE_GRID_SEARCH  Coarse-to-fine grid search for kB via maximum likelihood.

    % Coarse search: 100 points, log-spaced 1 to ~31623 cpm
    kB_coarse = logspace(0, 4.5, 100)';
    nll_coarse = inf(size(kB_coarse));

    for i = 1:numel(kB_coarse)
        spec_model = grad_func(K_fit, kB_coarse(i), chi_obs) .* H2_fit + noise_fit;
        spec_model = max(spec_model, 1e-30);
        nll = sum(log(spec_model) + spec_fit ./ spec_model);
        if isfinite(nll)
            nll_coarse(i) = nll;
        end
    end

    if all(isinf(nll_coarse))
        kB_best = NaN;
        nll_best = NaN;
        return
    end

    [~, idx] = min(nll_coarse);
    best_coarse = kB_coarse(idx);

    % Fine search: 100 points around coarse best
    kB_lo = max(best_coarse * 0.5, 1.0);
    kB_hi = best_coarse * 2.0;
    kB_fine = linspace(kB_lo, kB_hi, 100)';
    nll_fine = inf(size(kB_fine));

    for i = 1:numel(kB_fine)
        spec_model = grad_func(K_fit, kB_fine(i), chi_obs) .* H2_fit + noise_fit;
        spec_model = max(spec_model, 1e-30);
        nll = sum(log(spec_model) + spec_fit ./ spec_model);
        if isfinite(nll)
            nll_fine(i) = nll;
        end
    end

    if all(isinf(nll_fine))
        kB_best = best_coarse;
        nll_best = min(nll_coarse);
    else
        [nll_best, idx] = min(nll_fine);
        kB_best = kB_fine(idx);
    end
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
%   See chi_method1.m for the full implementation.

    E_n = 4e-9; fc = 18.7; E_n2 = 8e-9; fc_2 = 42;
    R_0 = 3000; gain = 6; f_AA = 110;
    adc_fs = 4.096; adc_bits = 16; gamma_RSI = 3;
    T_K = 295; K_B = 1.382e-23;

    delta_s = adc_fs / 2^adc_bits;
    fN = fs / 2;

    V1 = 2*E_n^2 .* sqrt(1+(F./fc).^2) ./ (F./fc);
    V1(~isfinite(V1)) = max(V1(isfinite(V1)));
    Noise_1 = gain^2 * (V1 + 4*K_B*R_0*T_K);

    G_2 = 1 + (2*pi*diff_gain.*F).^2;
    V2 = 2*E_n2^2 .* sqrt(1+(F./fc_2).^2) ./ (F./fc_2);
    V2(~isfinite(V2)) = max(V2(isfinite(V2)));
    Noise_2 = G_2 .* (Noise_1 + V2);

    Noise_3 = Noise_2 ./ (1 + (F./f_AA).^8).^2;
    Noise_4 = Noise_3 + gamma_RSI * delta_s^2 / (12*fN);
    Noise_counts = Noise_4 ./ delta_s^2;

    w = 2*pi*diff_gain.*F;
    Noise_counts = Noise_counts .* (1/diff_gain)^2 .* w.^2 ./ (1+w.^2);

    T_kelvin = T_mean + 273.15;
    eta = 0.5 * 2^adc_bits * gain * 0.68 / adc_fs;
    sf = T_kelvin^2 * 4 / (2 * eta * 3000);
    noise_K = Noise_counts .* sf^2 .* speed;
end
