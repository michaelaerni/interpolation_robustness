
acc1 = acos(sols(:,1,2)./sqrt(sols(:,1,2).^2+sols(:,1,1).^2))/pi

acc2 =  acos(sols(:,2,2)./sqrt(sols(:,2,2).^2+sols(:,2,1).^2))/pi

plot(1:8, acc1)
hold on
plot(1:8, acc2)
title('Standard accuracies of lambda = 1.0 vs lambda = 0.001 for eps = 0.1 and n=2000')
