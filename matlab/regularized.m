lambdavals = [1.0]
%gammavals = [0.5 1 1.5 2 2.5 3 3.5 4  4.5 5 5.5 6 6.5  7 7.5 8]



%gammavals = [  1 2 3 3  4 5  6 7  8];
gammavals = [ 3 ];

sgam = length(gammavals);
sla = length(lambdavals);
sols = zeros(sgam,sla,6);
diff= zeros(sgam,sla,6);

parfor gammai =1:sgam
    for lambdai = 1:sla
        gamma = gammavals(gammai);

        eps =0.05*sqrt(1000)*sqrt(gamma);
        gamma = gammavals(gammai);
        sigma = 0.0;
        nmc = 50000;
        lambda = lambdavals(lambdai);
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
        %lb = [0.0 0.0 0.0 0.0 0.0  0.0];
        %ub = [20 20 20 20 20 20];
        sol = lsqnonlin(@(x) New_eval(x,eps,gamma,sigma,nmc,lambda),sol,lb,ub, options)
        
        %options.FinDiffRelStep = 0.01;

        lb = [0.0 0.0 0.0 0.0 0.0  0.0];
        ub = [20 20 20 20 20 20];
        sol = lsqnonlin(@(x) New_eval(x,eps,gamma,sigma,nmc,lambda),sol,lb,ub, options)
        
        
        %opts = optimoptions(@fmincon,'TolFun',1e-15,'MaxFunEvals',1e5,'Maxiter',200,'Display','iter', 'FinDiffRelStep', 1e-1);
        %opts.TolX = 1e-20;
        %gs = MultiStart;
        %problem = createOptimProblem('lsqnonlin','x0',xmin,...
        %    'objective',@(x) mean(abs(New_eval(xmin,eps,gamma,sigma,nmc,lambda))),'lb',lb,'ub',ub);
        %sol = run(gs,problem,50)
        %mean(abs(New_eval(sol,eps,gamma,sigma,nmc)))
        %New_eval(sol,eps,gamma,sigma,nmc)

        
        diff(gammai, lambdai, :) =         New_eval(sol,eps,gamma,sigma,nmc,lambda);
        sols(gammai, lambdai,:) = sol(:);
    end
end







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


function [ls]= Prox(Z1,Z2, vo,vp,delta, taul, r, mu, eps, gamma, sigma, nmc, lambda)
    

    t = taul/r;
    mu = abs(Z1)*vp + Z2*vo -eps*delta;
    sol = zeros(1,length(Z1));
    
    for i = 1:1000
        %fx = 1/(2*t) *(sol - mu).^2 + log(1+exp(-sol));
        gx = - (2*mu - 2*sol)/(2*t) - exp(-sol)./(exp(-sol) + 1);
        sol = sol -0.1 * gx;
    end
    ls = max(sol,0);
end

function ls = Mt(    vo,vp,delta, taul, r, mu, eps, gamma, sigma, nmc, lambda,x,y, prox_v)
  
    mu = abs(x) .* vp + y.*vo -eps*delta;
    t = taul/r;    
    ls = -1/(2*t^2) * ( mu - prox_v).^2;
end
function ls = Mv(    vo,vp,delta, taul, r, mu, eps, gamma, sigma, nmc, lambda, x,y,prox_v)
    mu = abs(x).*vp + y .*vo -eps*delta;
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

    mv = @(x,y) Mv(vo,vp,delta, taul, r, 0.0, eps, gamma, sigma, nmc, lambda, x,y, Prox(x,y, vo,vp,delta, taul, r, 0.0, eps, gamma, sigma, nmc, lambda));

    mt = @(x,y) Mt(vo,vp,delta, taul, r, 0.0, eps, gamma, sigma, nmc, lambda, x,y, Prox(x,y, vo,vp,delta, taul, r, 0.0, eps, gamma, sigma, nmc, lambda));



    
    
    embas = integral2(@(x,y) exp(-x.^2/2) .*exp(-y.^2/2) .* 1/(2*pi) .*mt(x,y), -inf, inf,-inf,inf);
    emtaul = embas *1/r;
    emr = -embas *taul/r^2;
    emdelta = integral2(@(x,y) exp(-x.^2/2) .*exp(-y.^2/2) *1/(2*pi) .* mv(x,y), -inf, inf,-inf,inf);
    emZ =integral2(@(x,y) exp(-x.^2/2) .*exp(-y.^2/2) *1/(2*pi) .* mv(x,y) .* y, -inf,inf,-inf,inf);
    emZsigma = integral2(@(x,y) exp(-x.^2/2) .*exp(-y.^2/2) *1/(2*pi) .* mv(x,y) .* abs(x), -inf,inf,-inf,inf);
    

    
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
