% Mar-2026, Claude and Pat Welch, pat@mousebrains.com
function results = get_chi(gradT, P_fast, T_slow, speed, fs_fast, options)
%GET_CHI  Compute chi (thermal variance dissipation) for a profile.
%
%   results = get_chi(gradT, P_fast, T_slow, speed, fs_fast)
%   results = get_chi(..., method=1, epsilon=[], fft_length=512, ...)
%
%   Computes a depth profile of chi from temperature gradient data using
%   windowed spectral analysis.  Supports two methods:
%     Method 1: Chi from known epsilon (requires epsilon input)
%     Method 2: MLE Batchelor spectrum fitting (no epsilon needed)
%
%   Inputs:
%       gradT   - Temperature gradient signal [K/m], matrix with one
%                 column per thermistor (e.g., from make_gradT_odas)
%       P_fast  - Pressure at fast sampling rate [dbar]
%       T_slow  - Temperature at slow sampling rate [deg C]
%       speed   - Profiling speed [m/s], scalar or vector (length = rows of gradT)
%       fs_fast - Fast sampling rate [Hz]
%
%   Name-Value Options:
%       method         - 1 (from epsilon) or 2 (MLE fit). Default: 2
%       epsilon        - Epsilon vector, one per analysis window [W/kg].
%                        Required for method=1, ignored for method=2.
%       fft_length     - FFT segment length [samples] (default: 512)
%       diss_length    - Analysis window length [samples] (default: 3*fft_length)
%       overlap        - Window overlap [samples] (default: diss_length/2)
%       f_AA           - Anti-aliasing frequency [Hz] (default: 98)
%       diff_gain      - Differentiator gain [s] (default: 0.94)
%       spectrum_model - "batchelor" or "kraichnan" (default: "batchelor")
%       fp07_model     - "single_pole" or "double_pole" (default: "single_pole")
%       salinity       - Salinity [PSU] for viscosity (default: 35)
%
%   Output:
%       results - Struct with fields (n_probes x n_estimates):
%           chi         - Thermal variance dissipation rate [K^2/s]
%           epsilon_T   - Epsilon from temperature (method 2) or input [W/kg]
%           kB          - Batchelor wavenumber [cpm]
%           K_max       - Maximum integration wavenumber [cpm]
%           fom         - Figure of merit
%           K_max_ratio - K_max / kB
%           speed       - Mean speed per window [m/s]
%           nu          - Kinematic viscosity per window [m^2/s]
%           P_mean      - Mean pressure per window [dbar]
%           T_mean      - Mean temperature per window [deg C]
%           K           - Wavenumber vector [cpm] (per window)
%           F           - Frequency vector [Hz]
%           spec_gradT  - Observed spectra (n_freq x n_probes x n_estimates)
%           spec_batch  - Fitted spectra (n_freq x n_probes x n_estimates)
%           fs_fast     - Sampling rate [Hz]
%           fft_length  - FFT length used
%           diss_length - Window length used

    arguments
        gradT          (:,:) double
        P_fast         (:,1) double
        T_slow         (:,1) double
        speed          (:,:) double {mustBePositive}
        fs_fast        (1,1) double {mustBePositive}
        options.method         (1,1) double {mustBeMember(options.method, [1 2])} = 2
        options.epsilon        (:,:) double = double.empty(0,0)
        options.fft_length     (1,1) double {mustBePositive, mustBeInteger} = 512
        options.diss_length    (1,1) double = -1  % sentinel, set below
        options.overlap        (1,1) double = -1  % sentinel, set below
        options.f_AA           (1,1) double {mustBePositive} = 98
        options.diff_gain      (1,1) double {mustBePositive} = 0.94
        options.spectrum_model (1,1) string {mustBeMember(options.spectrum_model, ...
            ["batchelor", "kraichnan"])} = "batchelor"
        options.fp07_model     (1,1) string {mustBeMember(options.fp07_model, ...
            ["single_pole", "double_pole"])} = "single_pole"
        options.salinity       (1,1) double {mustBeNonnegative} = 35
    end

    fft_length = options.fft_length;

    if options.diss_length < 0
        diss_length = 3 * fft_length;
    else
        diss_length = options.diss_length;
    end

    if options.overlap < 0
        overlap = floor(diss_length / 2);
    else
        overlap = options.overlap;
    end

    f_AA_eff = 0.9 * options.f_AA;  % 10% safety margin

    N = size(gradT, 1);
    n_probes = size(gradT, 2);

    if isscalar(speed)
        speed = speed * ones(N, 1);
    end

    % Interpolate T_slow to fast rate for mean temperature
    n_slow = numel(T_slow);
    ratio = round(N / n_slow);
    t_slow_idx = (0:n_slow-1)' * ratio + ratio/2;
    t_fast_idx = (0:N-1)';
    T_fast = interp1(t_slow_idx, T_slow, t_fast_idx, "linear", "extrap");

    % Number of analysis windows
    step = diss_length - overlap;
    n_est = 1 + floor((N - diss_length) / step);
    n_freq = fft_length / 2 + 1;

    % Pre-allocate output
    chi_out     = NaN(n_probes, n_est);
    eps_out     = NaN(n_probes, n_est);
    kB_out      = NaN(n_probes, n_est);
    Kmax_out    = NaN(n_probes, n_est);
    fom_out     = NaN(n_probes, n_est);
    Kmr_out     = NaN(n_probes, n_est);
    speed_out   = NaN(1, n_est);
    nu_out      = NaN(1, n_est);
    P_out       = NaN(1, n_est);
    T_out       = NaN(1, n_est);
    K_out       = NaN(n_freq, n_est);
    spec_obs_out  = NaN(n_freq, n_probes, n_est);
    spec_fit_out  = NaN(n_freq, n_probes, n_est);

    F = (0:n_freq-1)' * fs_fast / fft_length;

    for j = 1:n_est
        i_start = 1 + (j-1) * step;
        i_end   = i_start + diss_length - 1;
        sel     = i_start:i_end;

        W_mean = mean(speed(sel));
        T_mean = mean(T_fast(sel));
        P_mean = mean(P_fast(sel));

        speed_out(j) = W_mean;
        T_out(j)     = T_mean;
        P_out(j)     = P_mean;

        % Kinematic viscosity
        nu = visc35(T_mean);  % uses ODAS visc35 or our simple version
        nu_out(j) = nu;

        K = F ./ W_mean;
        K_out(:, j) = K;

        for p = 1:n_probes
            seg = gradT(sel, p);

            % Welch spectrum (cosine window, 50% overlap)
            [Pxx, ~] = pwelch(seg, hanning(fft_length), fft_length/2, ...
                fft_length, fs_fast, "onesided");

            % Convert to wavenumber spectrum
            spec_obs = Pxx .* W_mean;
            spec_obs_out(:, p, j) = spec_obs;

            if options.method == 1
                % Method 1: chi from known epsilon
                if isempty(options.epsilon)
                    error("get_chi:noEpsilon", ...
                        "epsilon is required for method=1");
                end
                eps_val = options.epsilon(p, j);
                if ~isfinite(eps_val) || eps_val <= 0
                    continue
                end

                r = chi_method1(spec_obs, K, eps_val, nu, W_mean, ...
                    T_mean=T_mean, f_AA=f_AA_eff, fs=fs_fast, ...
                    diff_gain=options.diff_gain, ...
                    spectrum_model=options.spectrum_model, ...
                    fp07_model=options.fp07_model);

                chi_out(p, j) = r.chi;
                eps_out(p, j) = eps_val;
                kB_out(p, j)  = r.kB;
                Kmax_out(p, j) = r.K_max;
                fom_out(p, j)  = r.fom;
                Kmr_out(p, j)  = r.K_max_ratio;
                spec_fit_out(:, p, j) = r.spec_batch;

            else
                % Method 2: MLE fit
                r = chi_method2(spec_obs, K, nu, W_mean, ...
                    T_mean=T_mean, f_AA=f_AA_eff, fs=fs_fast, ...
                    diff_gain=options.diff_gain, ...
                    spectrum_model=options.spectrum_model, ...
                    fp07_model=options.fp07_model);

                chi_out(p, j) = r.chi;
                eps_out(p, j) = r.epsilon;
                kB_out(p, j)  = r.kB;
                Kmax_out(p, j) = r.K_max;
                fom_out(p, j)  = r.fom;
                Kmr_out(p, j)  = r.K_max_ratio;
                spec_fit_out(:, p, j) = r.spec_batch;
            end
        end
    end

    results = struct( ...
        "chi",         chi_out, ...
        "epsilon_T",   eps_out, ...
        "kB",          kB_out, ...
        "K_max",       Kmax_out, ...
        "fom",         fom_out, ...
        "K_max_ratio", Kmr_out, ...
        "speed",       speed_out, ...
        "nu",          nu_out, ...
        "P_mean",      P_out, ...
        "T_mean",      T_out, ...
        "K",           K_out, ...
        "F",           F, ...
        "spec_gradT",  spec_obs_out, ...
        "spec_batch",  spec_fit_out, ...
        "fs_fast",     fs_fast, ...
        "fft_length",  fft_length, ...
        "diss_length", diss_length);
end


function nu = visc35(T)
%VISC35  Kinematic viscosity of seawater at S=35 [m^2/s].
%   Sharqawy, Lienhard & Zubair, 2010.
    nu = 1e-6 .* (1.7910 - 6.144e-2.*T + 1.4510e-3.*T.^2 ...
        - 1.6826e-5.*T.^3 - 1.5290e-7.*T.^4);
end
