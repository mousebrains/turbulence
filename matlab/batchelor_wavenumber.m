% Mar-2026, Claude and Pat Welch, pat@mousebrains.com
function kB = batchelor_wavenumber(epsilon, nu, options)
%BATCHELOR_WAVENUMBER  Batchelor wavenumber [cpm].
%
%   kB = batchelor_wavenumber(epsilon, nu)
%   kB = batchelor_wavenumber(epsilon, nu, kappa_T=1.4e-7)
%
%   kB = (1/(2*pi)) * (epsilon / (nu * kappa_T^2))^(1/4)
%
%   References:
%       Oakey, N.S., 1982: J. Phys. Oceanogr., 12, 256-271.

    arguments
        epsilon     (:,:) double {mustBePositive}
        nu          (1,1) double {mustBePositive}
        options.kappa_T (1,1) double {mustBePositive} = 1.4e-7
    end

    kB = (1 / (2*pi)) .* (epsilon ./ (nu .* options.kappa_T.^2)).^0.25;
end
