    
x = 2;
y = 1.5;
Z = randn(1,10000);
hlls = zeros(10000,1);
for i = 1:10000
        hlls(i,1) = huber_f(Z(i)*x,y);
end

huber_f_solid(x,y)
mean(hlls)

    
    



    
function res = huber_f_solid(x,delta)
   res = x^2/2*erf(delta/(sqrt(2)*x)) - x*delta*exp(-delta^2/(2*x^2))/sqrt(2*pi) - 1/2*erfc(delta/(sqrt(2)*x))*delta^2 +2*delta*x*exp(-delta^2/(2*x^2))/sqrt(2*pi);
end







function res = huber_f(x,delta)
    if abs(x) <= delta
        res = 0.5*x^2;
    else
        res =  delta*(abs(x) - 0.5*delta);
    end
end
