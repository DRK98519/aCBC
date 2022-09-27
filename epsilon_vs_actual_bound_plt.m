close all
clc

data = readmatrix('sim_data.xlsx');
data = data(2:6, 2:3);
epsilon = data(:, 1);
actual_diff = data(:, 2);

% Plot
figure(1)
plot(epsilon, actual_diff, '.k', 'MarkerSize', 10)
hold on
plot([0; epsilon], [0; epsilon], '--b')
legend('Actual performance difference', 'Desired performance difference')
pbaspect([1 1 1])
grid on
grid minor
xlabel('$\epsilon$', 'Interpreter','latex')
ylabel('$\hat{U}_0$', 'Interpreter','latex')
title('$\hat{U}_0$ vs. $\epsilon$', 'Interpreter','latex')
