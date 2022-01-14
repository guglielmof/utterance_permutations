EXPERIMENT.analysis.MD2.model = [eye(4); [0 0 1 1]];
EXPERIMENT.analysis.MD2.labels = {'topic', 'utterance', 'perm', 'model'};

EXPERIMENT.analysis.MD2.nested = [[0, 0, 0, 0]; [1, 0, 0, 0]; [1, 0, 0, 0]; [0, 0, 0, 0]];

EXPERIMENT.analysis.MD2.compute = @(data, FACTORS)...
  anovan(...
    data, ...
    EXPERIMENT.analysis.getSelectedFactors(EXPERIMENT.analysis.MD2.labels, FACTORS), ... %groups labels
    'model', EXPERIMENT.analysis.MD2.model, ...
    'VarNames', EXPERIMENT.analysis.MD2.labels, ...
    'nested', EXPERIMENT.analysis.MD2.nested, ...
    'sstype', EXPERIMENT.analysis.anova.sstype, ...
    'alpha', EXPERIMENT.analysis.alpha.threshold, ...
    'display', 'off'...
  );

