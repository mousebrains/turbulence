% Mar-2026, Claude and Pat Welch, pat@mousebrains.com
function tau0 = fp07_time_constant(speed, options)
%FP07_TIME_CONSTANT  Speed-dependent FP07 thermistor time constant [s].
%
%   tau0 = fp07_time_constant(speed)
%   tau0 = fp07_time_constant(speed, model="lueck")
%
%   Models:
%       "lueck"    : tau = 0.01 * (1/speed)^0.5  (Lueck et al. 1977)
%       "peterson" : tau = 0.012 * speed^(-0.32)  (Peterson & Fer 2014)
%       "goto"     : tau = 0.003                  (Goto et al. 2016)

    arguments
        speed   (:,:) double {mustBePositive}
        options.model (1,1) string {mustBeMember(options.model, ...
            ["lueck", "peterson", "goto"])} = "lueck"
    end

    switch options.model
        case "lueck"
            tau0 = 0.01 .* (1.0 ./ speed).^0.5;
        case "peterson"
            tau0 = 0.012 .* speed.^(-0.32);
        case "goto"
            tau0 = 0.003 + zeros(size(speed));
    end
end
