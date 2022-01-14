EXPERIMENT.analysis.MD3.model = [eye(4); [1 0 1 0]];
EXPERIMENT.analysis.MD3.labels = {'topic', 'utterance', 'perm', 'model'};

EXPERIMENT.analysis.MD3.nested = [[0, 0, 0, 0]; [1, 0, 0, 0]; [1, 0, 0, 0]; [0, 0, 0, 0]];

EXPERIMENT.analysis.MD3.compute = @(data, FACTORS)...
  anovan(...
    data, ...
    EXPERIMENT.analysis.getSelectedFactors(EXPERIMENT.analysis.MD3.labels, FACTORS), ... %groups labels
    'model', EXPERIMENT.analysis.MD3.model, ...
    'VarNames', EXPERIMENT.analysis.MD3.labels, ...
    'nested', EXPERIMENT.analysis.MD3.nested, ...
    'sstype', EXPERIMENT.analysis.anova.sstype, ...
    'alpha', EXPERIMENT.analysis.alpha.threshold, ...
    'display', 'off'...
  );

