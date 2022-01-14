EXPERIMENT.analysis.MD1.model = eye(3);
EXPERIMENT.analysis.MD1.labels = {'topic', 'perm', 'model'};

EXPERIMENT.analysis.MD1.nested = [[0, 0, 0]; [1, 0, 0]; [0, 0, 0]];

EXPERIMENT.analysis.MD1.compute = @(data, FACTORS)...
  anovan(...
    data, ...
    EXPERIMENT.analysis.getSelectedFactors(EXPERIMENT.analysis.MD1.labels, FACTORS), ... %groups labels
    'model', EXPERIMENT.analysis.MD1.model, ...
    'VarNames', EXPERIMENT.analysis.MD1.labels, ...
    'nested', EXPERIMENT.analysis.MD1.nested, ...
    'sstype', EXPERIMENT.analysis.anova.sstype, ...
    'alpha', EXPERIMENT.analysis.alpha.threshold, ...
    'display', 'off'...
  );

