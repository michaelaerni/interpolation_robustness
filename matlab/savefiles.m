

n =1000

gamma = gammavals';
orth_l2_norm = sols_max_m(:,  1);
par_l2_norm = sols_max_m(:,  2);
proj_l1_norm = sols_max_m(:,  3) * sqrt(n) .* sqrt(gamma);
results = table(gamma, orth_l2_norm, par_l2_norm, proj_l1_norm);

current_file_name = sprintf('theory_predictions_large_0.0_eps_0.05_new.csv');
writetable(results, current_file_name);
