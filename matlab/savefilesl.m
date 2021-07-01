
% Create output CSVs
for lambdai = 1:sla
    lambda = lambdavals(lambdai);

    gamma = gammavals';
    orth_l2_norm = sols(:, lambdai, 1);
    par_l2_norm = sols(:, lambdai, 2);
    proj_l1_norm = sols(:, lambdai, 3) * sqrt(n) .* sqrt(gamma);
    results = table(gamma, orth_l2_norm, par_l2_norm, proj_l1_norm);

    current_file_name = sprintf('theory_predictions_large_lambda%.5f.csv', lambda);
    writetable(results, current_file_name);
end
