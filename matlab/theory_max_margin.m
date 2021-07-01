
eps =3.0;
gamma = 1.0;
sigma = 0.0;
nmc = 100000;
x0 = ones(7,1);
%x0(5:7) = 0.1;
%New_eval_p2(x0,eps,gamma,sigma,nmc)
xmin = ones(7,1);
difmin = 1000000;
for i =1:1
  x0 = rand(7,1)*5;
  nm = mean(abs(New_eval_p2(x0,eps,gamma,sigma,nmc)));
    if nm < difmin
        xmin = x0;
        difmin = nm;
    end
end
%lb = [0 0 0 -inf 0 0 0.0]
%ub = [inf inf inf inf inf inf inf]
%rng default % For reproducibility
%opts = optimoptions(@fmincon,'TolFun',1e-15,'MaxFunEvals',1e5,'Maxiter',20,'Display','iter', 'FinDiffRelStep', 1e-1);
%gs = GlobalSearch;
%problem = createOptimProblem('fmincon','x0',xmin,...
%    'objective',@(x) mean(abs(New_eval_p2(x,eps,gamma,sigma,nmc))),'lb',lb,'ub',ub);
%x = run(gs,problem)


xmin = ones(7,1);

options = optimset('TolFun',1e-15,'MaxFunEvals',1e5,'Maxiter',20,'Display','iter', 'FinDiffRelStep', 1e-1);
options.TolX = 1e-15;
%options.FiniteDifferenceType =  'central';
options.FinDiffRelStep = 1e-1;
%options.Algorithm =  'levenberg-marquardt';
%sol = fsolve(@(x) New_eval_p2(x,eps,gamma,sigma,nmc),xmin, options)
lb = [0 0 0 -inf 0.0 -inf 0.0]
%sol = lsqnonlin(@(x) New_eval_p2(x,eps,gamma,sigma,nmc),xmin,lb,ub, options)
%sol = fmincon(@(x) New_eval_p2(x,eps,gamma,sigma,nmc),xmin, [],[],[],[], lb,ub,[], options)

options = optimset('TolFun',1e-15,'MaxFunEvals',1e5,'Maxiter',200,'Display','iter');
options.TolX = 1e-15;
options.FiniteDifferenceType =  'central';
options.FinDiffRelStep = 1e-1;
%options.Algorithm =  'levenberg-marquardt';
sol = fsolve(@(x) New_eval_p2(x,eps,gamma,sigma,nmc),xmin, options)
ub = [20 20 20 20 20 20 20]
sol = lsqnonlin(@(x) New_eval_p2(x,eps,gamma,sigma,nmc),xmin,lb,ub, options)
New_eval_p2(sol,eps,gamma,sigma,nmc)

function res = huber_f(x,delta)
    if abs(x) <= delta
        res = 0.5*x^2;
    else
        res =  delta*(abs(x) - 0.5*delta);
    end
end


function res = huber_fy(x,delta)
    if abs(x) <= delta
        res =  0.0;
    else
        res = abs(x) - delta;
    end
end


function res = huber_fx(x,delta)
    if abs(x) <= delta
        res =  x;
    else
        res = delta*sign(x);
    end
end





function Z = get_rand_sigma_p2(sigma, nmc)
    Z = randn(2, nmc);
    Z(1,:) = abs(Z(1,:));
    pd = makedist('Binomial','N',1,'p',1-sigma);
    flips = (random(pd,1,nmc) - 0.5)*2;
    Z(1,:) = Z(1,:) .* flips;
end


function [mT,mTvo,mTvp, mTdelta,Z] =  T_fun( vo, vp, delta, zeta, r, kappa, taur   , eps, gamma,sigma,nmc)
    Z = get_rand_sigma_p2(sigma, nmc);
    lsT = ones(nmc,1);
    lsTvo = ones(nmc,1);
    lsTvp = ones(nmc,1);
    lsTdelta = ones(nmc,1);
    sz = size(Z);
    for i = 1:sz(2)
        a = vo;
        b = 1 + eps*delta - vp*Z(1,i);
        lsT(i) = (a^2+b^2)/2 * (1+erf(b/(sqrt(2)*a)) ) + a*b/sqrt(2*pi) * exp(-b^2/(2*a^2));
        lsTvo(i) =  a*(1+erf(b/(sqrt(2)*a)));

        T3 = 2*a/sqrt(2*pi)* exp(-b^2/(2*a^2))  + b*erf(b/(sqrt(2)*a))+b;
        lsTvp(i) =  -T3*Z(1,i);
        lsTdelta(i) =  T3*eps;
    end
    mT = mean(lsT);
    mTvo = mean(lsTvo);
    mTvp = mean(lsTvp);
    mTdelta = mean(lsTdelta);
end


function diff = New_eval_p2(x, eps, gamma,sigma,nmc)
      
    vo = x(1);
    vp = x(2);
    delta =x(3);
    zeta = x(4);
    r = x(5);
    kappa = x(6);
    taur = x(7);

    [T,Tvo,Tvp, Tdelta, Z] = T_fun(vo, vp, delta, zeta, r, kappa, taur, eps, gamma,sigma,nmc);

    hlls = ones(nmc,1);
    hxls = ones(nmc, 1);
    hyls = ones(nmc,1);
    opd = 1.0+kappa/(2*taur);
    sz = size(Z);
    for i = 1:sz(2)
        hx = -Z(2,i)* sqrt(gamma)*r/(2*(opd));
        hy = (zeta)/(2*(opd));
        hlls(i,1) = huber_f(hx,hy);
        hxls(i,1) =  -huber_fx(hx,hy)*Z(2,i);
        hyls(i,1) = huber_fy(hx,hy);
    end
    hl = mean(hlls);
    hx = mean(hxls);
    hy = mean(hyls);



    Hr = hx*sqrt(gamma);
    
    
    Hkappa = -1/(2*taur * opd)*(hx*(sqrt(gamma)*r) + hy*zeta) + hl/(taur);
    %gkappa =gamma*r^2/(2*opd^3)*(1/(2*taur)); 
    Hzeta = hy ;
    Htaur = kappa/(2*taur^2 * opd)*(hx*(sqrt(gamma)*r) + hy*zeta) -  hl*kappa/(taur^2);
    %2*vp + 1/(2*sqrt(T)) *r*Tvp;
    diff(1) =  -kappa +1/(2 *sqrt(T)) *r *Tvo;
    diff(2) = 2*vp + 1/(2*sqrt(T)) *r*Tvp;
    diff(3) = -zeta +  1/(2*sqrt(T)) *r*Tdelta;
    diff(4) = - delta + Hzeta;
    diff(5) =  sqrt(T) + Hr - gamma*r /(2*opd);
    diff(6) = -vo+Hkappa + gamma*r^2/(4*opd^2) *1/(2*taur) +taur/2;
    diff(7) =  Htaur- gamma*r^2/(4*(opd)^2) *kappa/(2*taur^2) +kappa/2;
    %val = vp^2 - vo*kappa - delta*zeta - gamma*r^2/(4*opd) + 2*opd*hl +r*sqrt(T) +taur*kappa/2;
end
