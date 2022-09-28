close all
clc

data = readmatrix('sim_data.xlsx');
data = data(2:6, 2:3);
epsilon = data(:, 1);
actual_diff = data(:, 2);

% Plot
figure(1)
plot(epsilon, actual_diff, '.k', 'MarkerSize', 30)
hold on
plot([0; 0.6], [0; 0.6], '--k', 'LineWidth', 7)
legend boxoff
lgnd = legend('Actual performance difference', 'Location','northwest');
set(lgnd, 'color', 'none')
set(lgnd, 'FontSize', 25)
set(gca, 'FontSize', 30)
axis([0, 0.6, 0, 0.6])
pbaspect([16 9 1])
xlabel('$\epsilon$', 'Interpreter','latex', 'FontSize', 45)
ylabel('$\hat{U}_0$', 'Interpreter','latex', 'FontSize', 45, 'Rotation', 0, 'HorizontalAlignment','right', 'VerticalAlignment', 'middle')
