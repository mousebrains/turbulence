% Mar-2026, Claude and Pat Welch, pat@mousebrains.com
function S = kraichnan_gradT(k, kB, chi, options)
%KRAICHNAN_GRADT  Kraichnan temperature gradient spectrum [(K/m)^2 / cpm].
%
%   S = kraichnan_gradT(k, kB, chi)
%   S = kraichnan_gradT(k, kB, chi, kappa_T=1.4e-7, q=5.26)
%
%   S(k) = chi*q/(3*kappa_T*kB^2) * k * (1 + sqrt(6q)*y) * exp(-sqrt(6q)*y)
%     y = k / kB
%
%   Exponential rolloff (vs Gaussian for Batchelor). Integrates to chi/(6*kappa_T).
%
%   References:
%       Bogucki, Domaradzki & Yeung, 1997: J. Fluid Mech., 343, 111-130.

    arguments
        k       (:,1) double
        kB      (1,1) double {mustBePositive}
        chi     (1,1) double {mustBePositive}
        options.kappa_T (1,1) double {mustBePositive} = 1.4e-7
        options.q       (1,1) double {mustBePositive} = 5.26
    end

    y = k ./ kB;
    sq6q = sqrt(6 * options.q);
    S = chi .* options.q ./ (3 .* options.kappa_T .* kB.^2) ...
        .* k .* (1 + sq6q .* y) .* exp(-sq6q .* y);
    S(~isfinite(S)) = 0;
end
