function [methods,NMSE, AUCs]= plotResultsCED(database, includes, attributes, facepart)
methods={};
error = [];
load([database '.mat'])
if strfind(database,'AFLW2000')
    
    load(['AFLW2000attributes.mat']);
elseif strfind(database,'multipie')
    load(['multipieattributes.mat']);
end
%database = database(9:end);

errcum=[];
a = [];
b = 0:0.0001:0.081;

if exist('includes', 'var')
    methindices = ismember(methods,includes);
    methods = methods(methindices);
    error = error(methindices,:,:);
end

if ~exist('facepart','var')
    facepart =1;
end

% faceparts:
% 1: all 
% 2: eyebrows
% 3: mouth
% 4: cheeks
error = error(:,:,facepart);
% specifiy what you want here
if exist('attributes', 'var') && ~isempty(attributes)
    tmpatt = attributes;
    if strfind(database,'AFLW2000')
        
        if strcmp(attributes.name,'yaw') || strcmp(attributes.name,'roll') || strcmp(attributes.name,'pitch')
            tmpatt.additional = deg2rad(attributes.additional);
        end
    end
    error = testSubset(error, fname, stats, tmpatt);
end

AUCs =[];
fails = [];
NMSE = [];
for i=1:length(methods)
    error(i,error(i,:)==0) = 5;
    %errcum(i,:) = cumsum(sort(error(i,:))) ./ (1:length(error));
    err = sort(error(i,:));
    %clean zeros
    for j = 1:length(b)
        a(i,j) = sum(err<b(j));
    end
    % the average error
    a(i,:) = a(i,:) / length(err);
  %  fprintf('Error for method %s is: \t %.4f\n',methods{i}, sum(error(i,:))/size(error,2));
    NMSE(i) = sum(error(i,:))/size(error,2);
    % compute auc
    [AUC,fail] = getAUCat(a(i,:),b, 0.05);
    %fprintf('AUC @ 0.05 is %0.3f and %0.3f %% are failures\n',AUC*100,fail*100);
    AUCs(i,1) = AUC;
    [AUC,fail] = getAUCat(a(i,:),b, 0.08);
    %fprintf('AUC @ 0.08 is %0.3f and %0.3f %% are failures\n',AUC*100,fail*100);
    AUCs(i,2) = AUC;
    fails(i) = fail;
    
end

%print latex style
%fprintf('Method & NMSE & AUC @ 0.05 & AUC @ 0.08 & Error \\textgreater > 0.08 \\\\\n');
%for i=2:length(methods)
%    fprintf('%s & %0.4f & %0.2f & %0.2f & %0.2f \\%% \\\\\n',methods{i}, NMSE(i), 100*AUCs(i,1), 100*AUCs(i,2),fails(i)*100);
%end


plot(b,a, 'LineWidth',2), grid on;
xlim([0 0.08]);
ylim([0 1]);
xlabel('NMSE','FontSize',18);
ylabel('Percent of testset','FontSize',18);
legend(methods, 'Location', 'northwest');
if exist('attributes', 'var') && ~isempty(attributes)
    title(['CED of DB: ' database ' facepart: ' num2str(facepart) ' attribute: ' num2str(attributes.name) ' dbSize: ' num2str(size(error,2))]);
    if strcmp(attributes.name, 'yaw')
        print('-dpng',['imgs/' database '_' num2str(attributes.name) '_' num2str(attributes.additional(1)) num2str(attributes.additional(2)) 'fp' num2str(facepart) '.png'])
    else
        print('-dpng',['imgs/' database '_' num2str(attributes.name) '_' num2str(attributes.additional(1)) 'fp' num2str(facepart) '.png'])
    end
else
    title(['CED of DB: ' database  ' facepart: ' num2str(facepart) ' attribute: no dbSize: ' num2str(size(error,2))]);
    print('-dpng',['imgs/' database 'fp' num2str(facepart) '.png'])
end

end

function [AUC, fail] =  getAUCat(error, indi, maxError)
inds  = indi < maxError;
indices = indi(inds);
relevantErr = error(inds);
AUC = trapz(indices, relevantErr) / maxError;
fail = 1-relevantErr(end);
end