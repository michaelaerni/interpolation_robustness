
eps =0.1*sqrt(1000);
eps = 0.001;
gamma = 1.0;
%eps =0.1*sqrt(gamma)*sqrt(1000);
sigma = 0.0;
nmc = 1000000;
x0 = rand(6,1);
%x0(5:7) = 0.1;
%New_eval_p2(x0,eps,gamma,sigma,nmc)

gammavals = [0.5 1 1.5 2 2.5 3 3.5 4  4.5 5 5.5 6 6.5  7 7.5 8];
%gammavals = [ 1   3  5  6 7  8];

sgam = size(gammavals);
sols_max_m = zeros(sgam(1),7);
diff= zeros(sgam(1),7);
for gammai =1:sgam(2)
    gamma = gammavals(gammai);
    eps =0.05*sqrt(1000)*sqrt(gamma);

    sigma = 0.0;
    nmc = 1000000;

    %x0(5:7) = 0.1;
    %New_eval_p2(x0,eps,gamma,sigma,nmc)

    xmin = rand(1,7);
    xmin(5) =1;
    %xmin(2) =3;
    options = optimset('TolFun',1e-15,'MaxFunEvals',1e5,'Maxiter',4000,'Display','iter');
    options.TolX = 1e-10;
    %options.FiniteDifferenceType =  'central';
    options.FinDiffRelStep = 0.1;
    options.Algorithm =  'levenberg-marquardt';
    %sol = fsolve(@(x) New_eval_p2(x,eps,gamma,sigma,nmc),xmin, options)
    lb = [1e-8 1e-8 1e-8 1e-8 1e-8  1e-8 1e-8];
    ub = [20 20 20 20 20 20 20];
    %sol = lsqnonlin(@(x) New_eval_p2(x,eps,gamma,sigma,nmc),xmin,lb,ub, options)

    options = optimset('TolFun',1e-15,'MaxFunEvals',1e5,'Maxiter',4000,'Display','iter');

    sol = lsqnonlin(@(x) New_eval_p2(x,eps,gamma,sigma,nmc),xmin,lb,ub, options)
    

    options.FinDiffRelStep = 0.01;
    sol = lsqnonlin(@(x) New_eval_p2(x,eps,gamma,sigma,nmc),xmin,lb,ub, options)

    options.FinDiffRelStep = 0.001;

    sol = lsqnonlin(@(x) New_eval_p2(x,eps,gamma,sigma,nmc),xmin,lb,ub, options)
    %options.FinDiffRelStep = 1.0;

    %sol = lsqnonlin(@(x) New_eval_p2(x,eps,gamma,sigma,nmc),xmin,lb,ub, options)
    %options.FinDiffRelStep = 0.001;

    %sol = lsqnonlin(@(x) New_eval_p2(x,eps,gamma,sigma,nmc),xmin,lb,ub, options)

    New_eval_p2(sol,eps,gamma,sigma,nmc)
    sols_max_m(gammai,:) = sol(:);
end




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


function [mT,mTvo,mTvp, mTdelta] =  T_fun( vo, vp, delta, zeta, r, kappa, taur   , eps, gamma,sigma,nmc)
    %Z = get_rand_sigma_p2(sigma, nmc);
    
    a = vo;
    
    b = @(x) 1 + eps*delta - vp*abs(x);
    
    lsT =@(x)exp(-x.^2/2)/sqrt(2*pi).* (a^2+b(x).^2)/2 .* (1+erf(b(x)/(sqrt(2)*a)) ) + a*b(x)./sqrt(2*pi) .* exp(-b(x).^2/(2*a^2));
    lsTvo =  @(x)exp(-x.^2 /2) /sqrt(pi*2) .* (a*(1+erf(b(x)/(sqrt(2)*a))));
    T3 = @(x)exp(-x.^2/2)/sqrt(2*pi).* (2 * a/sqrt(2*pi) .* exp(-b(x).^2/(2*a^2))  + b(x) .*erf(b(x)./(sqrt(2)*a))+b(x));
    lsTvp = @(x) -T3(x) .* abs(x) ;
    lsTdelta = @(x) T3(x).*eps;
    %pival = pi;
    %mTvovo = mean(erf((2^(1/2).* b)/(2*a)) - (2^(1/2) .* b .* exp(-b.^2/(2*a^2)))/(a*pi^(1/2)) + 1);

    %mTvovp = mean((2^(1/2)*exp(-b.^2/(2*a^2)))/pi^(1/2) .* -Z(1,:));
    %mTvodelta = mean((2^(1/2)*exp(-b.^2/(2*a^2)))/pi^(1/2) * eps);

    %lsT =@(x) (a^2+b(x)^2)/2 .* (1+erf(b(x)/(sqrt(2)*a)) ) + a*b(x)/sqrt(2*pi) .* exp(-b(x)^2/(2*a^2));
    %lsTvo =  a*(1+erf(b/(sqrt(2)*a)));
    %T3 = 2 * a/sqrt(2*pi) * exp(-b.^2/(2*a^2))  + b .*erf(b/(sqrt(2)*a))+b;
    %lsTvp =  -T3 .* Z(1,:);
    %lsTdelta =  T3*eps;
    %pival = pi;
    %mTvovo = mean(erf((2^(1/2).* b)/(2*a)) - (2^(1/2) .* b .* exp(-b.^2/(2*a^2)))/(a*pi^(1/2)) + 1);

    %mTvovp = mean((2^(1/2)*exp(-b.^2/(2*a^2)))/pi^(1/2) .* -Z(1,:));
    %mTvodelta = mean((2^(1/2)*exp(-b.^2/(2*a^2)))/pi^(1/2) * eps);
    
    %lsT33 = erf((2^(1/2)* b)/(2*a)) + (2^(1/2) .*b .*exp(-b.^2/(2*a^2)))/(a*pi^(1/2)) - (2^(1/2) .*b .*exp(-b.^2/(2*a^2)))/(a*pival^(1/2)) + 1;

    %mTvpvp =  mean(Z(1,:).^2 .* lsT33);

    %mTdeltdelta = mean(eps^2 *lsT33);
    %mTdeltavp = mean(-Z(1,:) .*eps .* lsT33);
    

    
    
    
    mT = integral(lsT,-inf,inf);
    mTvo = integral(lsTvo,-inf,inf);
    mTvp = integral(lsTvp,-inf,inf);
    mTdelta = integral(lsTdelta,-inf,inf);
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
    taur = x(7);
    
    [T,Tvo,Tvp, Tdelta] = T_fun(vo, vp, delta, zeta, r, kappa, taur, eps, gamma,sigma,nmc);


    opd = 1+kappa/(2*taur);

    %sz = size(Z);

    
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
    Hkappa = -1/opd^2 * hlv * 1/(2*taur);

    Hzeta = hyv/opd;
    
    diff = zeros(1,6);
    
    diff(1) =  (-kappa) +1/(2 *sqrt(T)) *r *Tvo;
    diff(2) = 2*vp + 1/(2*sqrt(T)) *r*Tvp;
    diff(3) = -zeta +  1/(2*sqrt(T)) *r*Tdelta;
    diff(4) = (- delta + Hzeta);
    diff(5) =  (sqrt(T) + Hr - gamma*r /(2*opd));
    diff(6) = -vo +Hkappa + gamma*r^2/(4*opd^2) *1/(2*taur)+ taur/2;
    diff(7) = +1/opd^2 * hlv * kappa/(2*taur^2) +kappa/2 -gamma*r^2/(4*opd^2) *kappa/(2*taur^2);
    
    
    % val = vp^2 - vo*kappa - delta*zeta - gamma*r^2/(4*opd) + 2*opd*hl +r*sqrt(T) +taur*kappa/2;
end
