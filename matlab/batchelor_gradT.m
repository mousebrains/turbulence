% Mar-2026, Claude and Pat Welch, pat@mousebrains.com
function S = batchelor_gradT(k, kB, chi, options)
%BATCHELOR_GRADT  Batchelor temperature gradient spectrum [(K/m)^2 / cpm].
%
%   S = batchelor_gradT(k, kB, chi)
%   S = batchelor_gradT(k, kB, chi, kappa_T=1.4e-7, q=3.7)
%
%   S(k) = sqrt(q/2) * chi / (kB * kappa_T) * f(alpha)
%     alpha = sqrt(2*q) * k / kB
%     f(alpha) = alpha * (exp(-alpha^2/2) - alpha*sqrt(pi/2)*erfc(alpha/sqrt(2)))
%
%   Integrates to chi / (6 * kappa_T).
%
%   References:
%       Dillon & Caldwell, 1980: J. Geophys. Res., 85, 1910-1916.
%       Oakey, 1982: J. Phys. Oceanogr., 12, 256-271.

    arguments
        k       (:,1) double
        kB      (1,1) double {mustBePositive}
        chi     (1,1) double {mustBePositive}
        options.kappa_T (1,1) double {mustBePositive} = 1.4e-7
        options.q       (1,1) double {mustBePositive} = 3.7
    end

    q = options.q;
    alpha = sqrt(2*q) .* k ./ kB;
    f = alpha .* (exp(-alpha.^2 / 2) ...
        - alpha .* sqrt(pi/2) .* erfc(alpha ./ sqrt(2)));
    S = sqrt(q/2) .* chi ./ (kB .* options.kappa_T) .* f;
end
