function [] = computeANOVA(TAG)
     
    

    common_params;
    
    %{
    measureTable = readtable(sprintf("../../data/measures/full_valid_100.csv"), "delimiter", ",");
    measureTable(:, 'model') = measureTable(:, 'name');
    measureTable(:, 'name') = [];
    

    measure = measureTable;
    %}
   
    
    
    measureTable = readtable(sprintf("../../data/measures/full_valid_100.csv"), "delimiter", ",");
    
    measureTable(:, 'Var1') = [];
    %measureTable(:, 'Unnamed_0') = [];
    %measureTable(:, 'AP') = [];
    %measureTable(:, 'P_1') = [];
    %measureTable(:, 'P_3') = [];
    %measureTable(:, 'RR_10') = [];
    %measureTable(:, 'nDCG_10') = [];
    
    
    tp = split(measureTable{:, 'qid'}, "_");
    %measureTable(:, 'topic') = tp(:, 1);
    %measureTable(:, 'utterance') = tp(:, 2);    
    measureTable(:, 'model') = measureTable(:, 'name');
    measureTable(:, 'name') = [];
    
    

    %%%% pass from row data to means
    redMeasure = measureTable(:, ["nDCG_3", "perm", "topic", "model", "qtype"]);
    redMeasure = grpstats(redMeasure, ["perm", "topic", "model", "qtype"]);
    measure = redMeasure;
    measure(:, 'nDCG_3') = measure(:, 'mean_nDCG_3');
    measure(:, 'mean_nDCG_3') = [];
    measure(:, 'GroupCount') = [];
    %%%%

    FILTERS = struct();
    %"first_query"
    FILTERS.model = ["RM3_seq", "RM3_prev","linear_prev", "allennlp_simple", "context_query"];
    FILTERS.perm = [0];
    FILTERS.qtype = ["original"];
    
    measure = filterMeasure(measure, FILTERS);

    
    

    for nVar = 1:length(measure.Properties.VariableNames)
        f = measure.Properties.VariableNames{nVar};
        FACTORS.(f) = measure{:, f};
    end
    
    [~, tbl, stats] = EXPERIMENT.analysis.(TAG).compute(FACTORS.nDCG_3, FACTORS);
    
    
    
    fl = getFactorLabels(TAG);
    soa = computeSOA(height(measure), tbl, fl);
    
    disp(tbl);
    disp(soa.omega2p);

    disp(getLatexANOVATable(TAG, tbl, soa));
    
    if strcmp(TAG, 'MD0')
        [c, m, h] = multcompare(stats, 'dim', 2);
    end
    
end