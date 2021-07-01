


r = 3
sigma = 3
delta = 3
vo= 1



sol = arrayfun(@(mu) g(mu, r, gamma), linspace(0,50,50))


function res = g(mu, r,  gamma)
    alpha = sqrt(gamma)*r;
    tmu = mu/alpha;
    a = sqrt(2/pi)
    res = (mu^2+alpha^2) -(mu^2+alpha^2)*erf(tmu/sqrt(2))- alpha*a*mu*exp(-tmu^2/2);
end



function val = computemu_t(mu, r,  gamma, vo, delta)   
	gval = g(mu, r,  gamma);
    val = vo*sqrt(gval) + delta*mu;
    mu;
    %der = vo*1/2*1/sqrt(gval) * gmuf(mu, r, eps, gamma, sigma, nmc, lambda);
end
