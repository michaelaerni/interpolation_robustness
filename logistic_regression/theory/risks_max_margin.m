
eps_base = 0.02;
gamma = 1.0;
sigma = 0.0;
nmc = 10000000;
x0 = rand(6,1);
n = 1000;

gammavals = [0.5 1 1.5 2 2.5 3 3.5 4  4.5 5 5.5 6 6.5  7 7.5 8];


% Prepare output directory
[current_dir, ~, ~] = fileparts(mfilename('fullpath'));
log_dir = fullfile(current_dir, '..', '..', 'logs');
output_dir = 'logistic_regression_theory';
mkdir(log_dir, output_dir);
output_file = sprintf('%s_%d.mat', datestr(now, 'yyyy-mm-dd_HHMMSS'), randi(100000));
output_path = fullfile(log_dir, output_dir, output_file)


sgam = length(gammavals);
sols = zeros(sgam, 6);
diff= zeros(sgam, 6);
for gammai = 1:sgam
    eps = eps_base * sqrt(n) * sqrt(gamma);
    gamma = gammavals(gammai);

    x0 = ones(6,1);
    options = optimset('TolFun',1e-15,'MaxFunEvals',1e5,'Maxiter',200,'Display','iter');
    options.TolX = 1e-10;
    %options.FiniteDifferenceType =  'central';
    options.FinDiffRelStep = 0.1;
    %options.Algorithm =  'levenberg-marquardt';
    lb = [1e-8 1e-8 1e-8 1e-8 1e-8 1e-8];
    ub = [20 20 20 20 20 20];
    options = optimset('TolFun',1e-15,'MaxFunEvals',1e5,'Maxiter',200,'Display','iter');
    options.TolX = 1e-10;
    options.FiniteDifferenceType =  'central';

    options.FinDiffRelStep = 0.1;
    lb = [1e-8 1e-8 1e-8 1e-8 1e-8  1e-8];
    ub = [20 20 20 20 20 20 ];

    options.Algorithm =  'levenberg-marquardt';

    sol = lsqnonlin(@(x) New_eval_p2(x,eps,gamma,sigma,nmc),x0,lb,ub, options)



    options.Algorithm =  'trust-region-reflective';
    sol = lsqnonlin(@(x) New_eval_p2(x,eps,gamma,sigma,nmc),sol,lb,ub, options)

    options.FinDiffRelStep = 0.01;

    sol = lsqnonlin(@(x) New_eval_p2(x,eps,gamma,sigma,nmc),sol,lb,ub, options)




    diff(gammai, :) = New_eval_p2(sol, eps, gamma, sigma, nmc);
    sols(gammai, :) = sol(:);







    % Save entire workspace every finished iteration, just in case
    save(output_path);
end

% Create output CSV
gamma = gammavals';
orth_l2_norm = sols(:, 1);
par_l2_norm = sols(:, 2);
proj_l1_norm = sols(:, 3) * sqrt(n) .* sqrt(gamma);
results = table(gamma, orth_l2_norm, par_l2_norm, proj_l1_norm);

current_file_name = sprintf('theory_predictions_lambda%.5f.csv', 0.0);
current_file_path = fullfile(log_dir, output_dir, current_file_name)
writetable(results, current_file_path);


function res = huber_f(x,delta)
   res = x^2/2*erf(delta/(sqrt(2)*x)) - x*delta*exp(-delta^2/(2*x^2))/sqrt(2*pi) - 1/2*erfc(delta/(sqrt(2)*x))*delta^2 +2*delta*x*exp(-delta^2/(2*x^2))/sqrt(2*pi);
end



function res = huber_fy(x,delta)
  vpi = pi;

  res = (2^(1/2)*x*exp(-delta^2/(2*x^2)))/(2*pi^(1/2)) - delta*erfc((2^(1/2)*delta)/(2*x)) + (2^(1/2)*x*exp(-delta^2/(2*x^2)))/(2*vpi^(1/2)) + (2^(1/2)*delta^2*exp(-delta^2/(2*x^2)))/(2*x*pi^(1/2)) - (2^(1/2)*delta^2*exp(-delta^2/(2*x^2)))/(2*vpi^(1/2)*x);

end

function res = huber_fyy(x,delta)
  vpi = pi;

  res = (3*2^(1/2)*delta*exp(-delta^2/(2*x^2)))/(2*x*pi^(1/2)) - erfc((2^(1/2)*delta)/(2*x)) - (3*2^(1/2)*delta*exp(-delta^2/(2*x^2)))/(2*vpi^(1/2)*x) - (2^(1/2)*delta^3*exp(-delta^2/(2*x^2)))/(2*x^3*pi^(1/2)) + (2^(1/2)*delta^3*exp(-delta^2/(2*x^2)))/(2*vpi^(1/2)*x^3);

end

function res = huber_fx(x,delta)
  vpi = pi;
  res = x*erf((2^(1/2)*delta)/(2*x)) - (2^(1/2)*delta*exp(-delta^2/(2*x^2)))/(2*pi^(1/2)) + (2^(1/2)*delta*exp(-delta^2/(2*x^2)))/(2*vpi^(1/2)) - (2^(1/2)*delta^3*exp(-delta^2/(2*x^2)))/(2*x^2*pi^(1/2)) + (2^(1/2)*delta^3*exp(-delta^2/(2*x^2)))/(2*vpi^(1/2)*x^2);
end

function res = huber_fxx(x,delta)
  vpi = pi;
  res = erf((2^(1/2)*delta)/(2*x)) - (2^(1/2)*delta*exp(-delta^2/(2*x^2)))/(x*pi^(1/2)) + (2^(1/2)*delta^3*exp(-delta^2/(2*x^2)))/(2*x^3*pi^(1/2)) - (2^(1/2)*delta^5*exp(-delta^2/(2*x^2)))/(2*x^5*pi^(1/2)) - (2^(1/2)*delta^3*exp(-delta^2/(2*x^2)))/(2*vpi^(1/2)*x^3) + (2^(1/2)*delta^5*exp(-delta^2/(2*x^2)))/(2*vpi^(1/2)*x^5);
end

function res = huber_fxy(x,delta)
  vpi = pi;
  res = (2^(1/2)*exp(-delta^2/(2*x^2)))/(2*pi^(1/2)) + (2^(1/2)*exp(-delta^2/(2*x^2)))/(2*vpi^(1/2)) - (2^(1/2)*delta^2*exp(-delta^2/(2*x^2)))/(x^2*pi^(1/2)) + (2^(1/2)*delta^4*exp(-delta^2/(2*x^2)))/(2*x^4*pi^(1/2)) + (2^(1/2)*delta^2*exp(-delta^2/(2*x^2)))/(vpi^(1/2)*x^2) - (2^(1/2)*delta^4*exp(-delta^2/(2*x^2)))/(2*vpi^(1/2)*x^4);
end

function Z = get_rand_sigma_p2(sigma, nmc)
    Z = randn(2, nmc);
    Z(1,:) = abs(Z(1,:));
    pd = makedist('Binomial','N',1,'p',1-sigma);
    flips = (random(pd,1,nmc) - 0.5)*2;
    Z(1,:) = Z(1,:) .* flips;
end

function [mT,mTvo,mTvp, mTdelta,mTvovo,mTvovp, mTvodelta,mTvpvp,mTdeltdelta, mTdeltavp, Z] =  T_fun( vo, vp, delta, zeta, r, kappa, taur   , eps, gamma,sigma,nmc)
    Z = get_rand_sigma_p2(sigma, nmc);

    a = vo;
    b = 1 + eps*delta - vp*Z(1,:);
    lsT = (a^2+b.^2)/2 .* (1+erf(b./(sqrt(2)*a)) ) + a*b/sqrt(2*pi) .* exp(-b.^2/(2*a^2));
    lsTvo =  a*(1+erf(b/(sqrt(2)*a)));
    T3 = 2 * a/sqrt(2*pi) * exp(-b.^2/(2*a^2))  + b .*erf(b/(sqrt(2)*a))+b;
    lsTvp =  -T3 .* Z(1,:);
    lsTdelta =  T3*eps;
    pival = pi;
    mTvovo = mean(erf((2^(1/2).* b)/(2*a)) - (2^(1/2) .* b .* exp(-b.^2/(2*a^2)))/(a*pi^(1/2)) + 1);

    mTvovp = mean((2^(1/2)*exp(-b.^2/(2*a^2)))/pi^(1/2) .* -Z(1,:));
    mTvodelta = mean((2^(1/2)*exp(-b.^2/(2*a^2)))/pi^(1/2) * eps);


    lsT33 = erf((2^(1/2)* b)/(2*a)) + (2^(1/2) .*b .*exp(-b.^2/(2*a^2)))/(a*pi^(1/2)) - (2^(1/2) .*b .*exp(-b.^2/(2*a^2)))/(a*pival^(1/2)) + 1;

    mTvpvp =  mean(Z(1,:).^2 .* lsT33);

    mTdeltdelta = mean(eps^2 *lsT33);
    mTdeltavp = mean(-Z(1,:) .*eps .* lsT33);

    mT = mean(lsT);
    mTvo = mean(lsTvo);
    mTvp = mean(lsTvp);
    mTdelta = mean(lsTdelta);
end

function val = approx_zeta(zeta, Z, gamma, opd, delta,r)
    val =  mean(arrayfun(@(z) -delta*zeta + 2*opd * huber_f(-z*sqrt(gamma)*r/(2*opd), zeta/(2*opd)), Z(2,:)));
    %der = mean(arrayfun(@(z) -delta + huber_fy(-z*sqrt(gamma)*r/(2*opd), zeta/(2*opd)), Z(2,:)));
end

function [diff,jacobim] = New_eval_p2(x, eps, gamma,sigma,nmc)

    vo = x(1);
    vp = x(2);
    delta =x(3);
    zeta = x(4);
    r = x(5);
    kappa = x(6);

    taur = 0.5;
    [T,Tvo,Tvp, Tdelta,Tvovo,Tvovp, Tvodelta,Tvpvp,Tdeltdelta, Tdeltavp, Z] = T_fun(vo, vp, delta, zeta, r, kappa, taur, eps, gamma,sigma,nmc);

    hlls = ones(nmc,1);
    hxls = ones(nmc, 1);
    hyls = ones(nmc,1);
    hyyls = ones(nmc,1);
    hxyls = ones(nmc,1);
    hxxls = ones(nmc,1);

    opd = 1+kappa;

    sz = size(Z);


    %options = optimset('TolFun',1e-15,'MaxFunEvals',1e5,'Maxiter',10,'Display','off');
    %zeta = fmincon(@(zeta)  approx_zeta(zeta, Z, gamma, opd, delta,r), 1.0,[],[],[],[],0.0,5,[], options);


    hx =  sqrt(gamma)*r;
    hy = (zeta);
    hlv =0.5* huber_f(hx,hy);
    hxv = 0.5*huber_fx(hx,hy);
    hyv = 0.5*huber_fy(hx,hy);
    hyy = 0.5*huber_fyy(hx,hy);
    hxy =0.5*huber_fxy(hx,hy);
    hxx = 0.5*huber_fxx(hx,hy);




    %diff(4) = - delta + Hzeta;


    diff = zeros(1,5);
    Hr = hxv*sqrt(gamma)/opd;
    Hkappa = -1/opd^2 * hlv;

    Hzeta = hyv/opd;


    diff(1) =  (-kappa)*2*vo +1/(2 *sqrt(T)) *r *Tvo;
    diff(2) = 2*vp + 1/(2*sqrt(T)) *r*Tvp;
    diff(3) = -zeta +  1/(2*sqrt(T)) *r*Tdelta;
    diff(4) = (- delta + Hzeta);
    diff(5) =  (sqrt(T) + Hr - gamma*r /(2*opd));
    diff(6) = -vo^2 +Hkappa + gamma*r^2/(4*opd^2);


    % Now we want to compute the jacobim matrix. For this, we do the
    % following:
    s = opd;
    jacobim = zeros(6,6);
    jacobim(1,1) = -2*kappa + r* 1/(2*sqrt(T)) *Tvovo -r/(4*T^(3/2)) *Tvo^2;
    jacobim(1,2) = r* 1/(2*sqrt(T)) *Tvovp -r/(4*T^(3/2)) *Tvo*Tvp;
    jacobim(3,1) =  r* 1/(2*sqrt(T)) *Tvodelta -r/(4*T^(3/2)) *Tvo*Tdelta;
    jacboi(1,6) = -2*vo;
    jacobim(1,5) =  1/(2*sqrt(T)) *Tvo;

    jacobim(2,2) = 2 + r* 1/(2*sqrt(T)) *Tvpvp -r/(4*T^(3/2)) *Tvp^2;
    jacobim(2,3) = r* 1/(2*sqrt(T)) *Tdeltavp -r/(4*T^(3/2)) *Tvp*Tdelta;

    jacobim(3,3) = r* 1/(2*sqrt(T)) *Tdeltdelta -r/(4*T^(3/2)) *Tdelta*Tdelta;
    jacobim(3,4) = -1;
    jacobim(4,4) = 1/(s) * hyy;
    jacobim(4,5) = 1/(s) * hxy *sqrt(gamma);
    jacobim(4,6) = -1/(s^2) * hyv;
    jacboi(5,5) =  - gamma/(2*s)  + 1/s * hxx * gamma;
    jacobim(5,6) =  gamma*r/(2*s^2) - 1/s^2 * hx* sqrt(gamma);
    jacobim(6,6) =  +2/(s^3) *hlv - gamma*r^2/(2*s^3);

    jacobim(2,1) = jacobim(1,2);
    jacobim(3,1) = jacobim(1,3);
    jacobim(6,1) = jacboi(1,6);
    jacobim(5,1) =jacobim(1,5);
    jacobim(3,2) = jacobim(2,3);
    jacobim(4,3) = jacobim(3,4);
    jacobim(5,4) = jacobim(4,5);
    jacobim(6,4) = jacobim(4,6);
    jacobim(6,5)= jacobim(5,6);

    % val = vp^2 - vo*kappa - delta*zeta - gamma*r^2/(4*opd) + 2*opd*hl +r*sqrt(T) +taur*kappa/2;
end
