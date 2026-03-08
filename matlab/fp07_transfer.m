% Mar-2026, Claude and Pat Welch, pat@mousebrains.com
function H2 = fp07_transfer(f, tau0, options)
%FP07_TRANSFER  FP07 thermistor transfer function |H(f)|^2.
%
%   H2 = fp07_transfer(f, tau0)
%   H2 = fp07_transfer(f, tau0, model="single_pole")
%
%   References:
%       Lueck, Hertzman & Osborn, 1977 (single pole)
%       Gregg & Meagher, 1980 (double pole)

    arguments
        f       (:,1) double
        tau0    (1,1) double {mustBePositive}
        options.model (1,1) string {mustBeMember(options.model, ...
            ["single_pole", "double_pole"])} = "single_pole"
    end

    switch options.model
        case "single_pole"
            H2 = 1 ./ (1 + (2*pi*f*tau0).^2);
        case "double_pole"
            H2 = 1 ./ (1 + (2*pi*f*tau0).^2).^2;
    end
end
