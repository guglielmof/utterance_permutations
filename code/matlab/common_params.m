addpath(genpath('anova_models'));
addpath(genpath('utils'));


EXPERIMENT.analysis.getSelectedFactors = @(labels, FACTORS) getSelectedFactors(labels, FACTORS);


models;


function [selectedFactors] = getSelectedFactors(labels, FACTORS)
    selectedFactors = cell(1, length(labels));
    for ln=1:length(labels)
        lb = labels{ln};
        selectedFactors{ln} = FACTORS.(lb);
    end
end