function plotDetectionCurves(name, modeldir, plotdir)

names = {'ONetORIG','ONetv2','HeadHunter','ONetv3',[modeldir name]};

for i = 1:length(names)
    res = dlmread([names{i} '.txt']);
    plot(res(:,2), res(:,1), 'LineWidth', 3);
    hold on
end
xlim([1 800])
grid on;
legend(names, 'Location', 'southeast')
title('FDDB MTCNN')
set(gca,'Ytick',0.6:0.02:1)
ylim([0.6 1])
xlabel('False Positive')
ylabel('True positive rate')
print('-dpng',[ plotdir '/' name 'FDDB.png'])