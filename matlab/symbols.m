
syms delta x vpi


res = x^2/2*erf(delta/(sqrt(2)*x)) - x*delta*exp(-delta^2/(2*x^2))/sqrt(2*vpi) - 1/2*erfc(delta/(sqrt(2)*x))*delta^2 +2*delta*x*exp(-delta^2/(2*x^2))/sqrt(2*vpi);

diff(res,x)
diff(diff(res,x),x)
diff(res,delta)
diff(diff(res,delta),delta)
diff(diff(res,delta),x)
