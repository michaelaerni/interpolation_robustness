sols = zeros(1, 6);
n = 1000;

% Prepare output directory
[current_dir, ~, ~] = fileparts(mfilename('fullpath'));
log_dir = fullfile(current_dir, '..', '..', 'logs');
output_dir = 'logistic_regression_theory_regularized_large_parallel';
mkdir(log_dir, output_dir);
output_file = sprintf('%s_%d.mat', datestr(now, 'yyyy-mm-dd_HHMMSS'), randi(100000));
output_path = fullfile(log_dir, output_dir, output_file)


eps = 0.05*sqrt(1000)*sqrt(gamma);
sigma = 0.0;
nmc = 500000;
xmin = ones(6,1);
%x0(5:7) = 0.1;
%New_eval_p2(x0,eps,gamma,sigma,nmc)
xmin = ones(6,1);
difmin = 1000000;


options = optimset('TolFun',1e-15,'MaxFunEvals',1e5,'Maxiter',200,'Display','iter');
options.TolX = 1e-10;
options.FiniteDifferenceType =  'central';
options.FinDiffRelStep = 0.1;
options.Algorithm =  'levenberg-marquardt';

lb = [0.0 0.0 0.0 0.0 0.0  0.0];
ub = [20 20 20 20 20 20];
sol = lsqnonlin(@(x) New_eval(x,eps,gamma,sigma,nmc,lambda),xmin,lb,ub, options)



options.Algorithm =  'trust-region-reflective';
%sol = fsolve(@(x) New_eval(x,eps,gamma,sigma,nmc,lambda),xmin, options)
lb = [0.0 0.0 0.0 0.0 0.0  0.0];
ub = [20 20 20 20 20 20];
sol = lsqnonlin(@(x) New_eval(x,eps,gamma,sigma,nmc,lambda),sol,lb,ub, options)

options.FinDiffRelStep = 0.01;

lb = [0.0 0.0 0.0 0.0 0.0  0.0];
ub = [20 20 20 20 20 20];
sol = lsqnonlin(@(x) New_eval(x,eps,gamma,sigma,nmc,lambda),sol,lb,ub, options)


orth_l2_norm = sol(1);
par_l2_norm = sols(2);
proj_l1_norm = sols(3) * sqrt(n) .* sqrt(gamma);
results = table(gamma, orth_l2_norm, par_l2_norm, proj_l1_norm);

current_file_name = sprintf('theory_predictions_large_lambda%.5f_gamma%.5f.csv', lambda, gamma);
current_file_path = fullfile(log_dir, output_dir, current_file_name)
writetable(results, current_file_path);










function Z = get_rand_sigma(sigma, nmc)
    Z = randn(2, nmc);
    Z(1,:) = abs(Z(1,:));
    pd = makedist('Binomial','N',1,'p',1-sigma);
    flips = (random(pd,1,nmc) - 0.5)*2;
    Z(1,:) = Z(1,:) .* flips;
end

function res = mureau(theta,t,mu)
    res =1/(2*t) *(theta - mu)^2 + log(1+exp(-theta));
end


function [ls,Z]= Prox(vo,vp,delta, taul, r, mu, eps, gamma, sigma, nmc, lambda)
    Z = get_rand_sigma(sigma, nmc);

    sz = size(Z);

    ls = zeros(sz(2),1);
    t = taul/r;
    mu = Z(1,:)*vp + Z(2,:)*vo -eps*delta;
    sol = ones(1,sz(2))*3;

    for i = 1:1000
        %fx = 1/(2*t) *(sol - mu).^2 + log(1+exp(-sol));
        gx = - (2*mu - 2*sol)/(2*t) - exp(-sol)./(exp(-sol) + 1);
        sol = sol -0.1 * gx;
    end
    ls = arrayfun(@(x) max(x,0), sol);
end

function ls = Mt(    vo,vp,delta, taul, r, mu, eps, gamma, sigma, nmc, lambda, prox_v, Z)

    mu = Z(1,:)*vp + Z(2,:)*vo -eps*delta;
    t = taul/r;
    ls = -1/(2*t^2) * ( mu - prox_v).^2;
end
function ls = Mv(    vo,vp,delta, taul, r, mu, eps, gamma, sigma, nmc, lambda, prox_v, Z)
    mu = Z(1,:)*vp + Z(2,:)*vo -eps*delta;
    t = taul/r;
    ls = 1/t * ( mu - prox_v);

end


function res = g(mu, r,  eps, gamma, sigma, nmc, lambda)
    alpha = sqrt(gamma)* r;
    tmu = mu/alpha;
    a = sqrt(2)/sqrt(pi);

    res = (mu^2+alpha^2) -(mu^2+alpha^2)*erf(tmu/sqrt(2))- alpha*a*mu*exp(-tmu^2/2);
end

function res = grf(mu, r,  eps, gamma, sigma, nmc, lambda)
    a = sqrt(2)/sqrt(pi);
    res = 2*gamma*r - 2*gamma*r*erf((2^(1/2)*mu)/(2*gamma^(1/2)*r)) - a*gamma^(1/2)*mu*exp(-mu^2/(2*gamma*r^2)) - (a*mu^3*exp(-mu^2/(2*gamma*r^2)))/(gamma^(1/2)*r^2) + (2^(1/2)*mu*exp(-mu^2/(2*gamma*r^2))*(mu^2 + gamma*r^2))/(gamma^(1/2)*r^2*pi^(1/2));
end
function res = gmuf(mu, r,  eps, gamma, sigma, nmc, lambda)
    a = sqrt(2)/sqrt(pi);
    res = 2*mu - 2*mu*erf((2^(1/2)*mu)/(2*gamma^(1/2)*r)) - a*gamma^(1/2)*r*exp(-mu^2/(2*gamma*r^2)) + (a*mu^2*exp(-mu^2/(2*gamma*r^2)))/(gamma^(1/2)*r) - (2^(1/2)*exp(-mu^2/(2*gamma*r^2))*(mu^2 + gamma*r^2))/(gamma^(1/2)*r*pi^(1/2));
end



function ret = New_eval(x, eps, gamma, sigma, nmc, lambda)
    vo = real(x(1));
    vp = real(x(2));
    delta = real(x(3));
    taul = real(x(4));
    r = real(x(5));
    mu = real(x(6));

    [prox_v, Z] = Prox(vo,vp,delta, taul, r, 0.0, eps, gamma, sigma, nmc, lambda);
    mv = Mv(vo,vp,delta, taul, r, 0.0, eps, gamma, sigma, nmc, lambda, prox_v, Z);
    mt = Mt(vo,vp,delta, taul, r, 0.0, eps, gamma, sigma, nmc, lambda, prox_v, Z);



    emtaul = mean(mt) * 1/r;
    emr = -mean(mt) *taul/r^2;
    emdelta = mean(mv);
    emZls = mv .* Z(2,:);
    emZsigmals = mv .*Z(1,:);

    emZ = mean(emZls);
    emZsigma = mean(emZsigmals);

    %options = optimset('TolFun',1e-15,'MaxFunEvals',1e5,'Maxiter',50,'Display','off');
    %mu  = fmincon(@(mu) computemu(mu, r, eps, gamma, sigma, nmc, lambda,vo,delta), 2.0,[],[],[],[],0.0,5,[], options);

    gr =  real(grf(mu, r, eps, gamma, sigma, nmc, lambda));

    gmu =  real(gmuf(mu, r, eps, gamma, sigma, nmc, lambda));

    gval = max(1e-8, real(g(mu, r, eps, gamma, sigma, nmc, lambda)));

    ret = zeros(1,6);
    ret(1,2) = emZsigma + 2*lambda*vp;
    ret(1,1) = (emZ + 2*lambda*vo - sqrt(gval));
    ret(1,3) = (-emdelta *eps - mu);
    ret(1,4) = emtaul +r/2;
    ret(1,5) = emr - gr/(2*sqrt(gval)) * vo + taul/2;
    ret(1,6) = -vo*1/(2*sqrt(gval)) * gmu - delta;
    %ret = sign(ret)*sqrt(abs(ret));
end
