function [  ] = testDB( pathToDatabase, pointScheme, visiblePoints, normalization )
%TESTDB Draw graph of db
% first need to parse the folder
% get groundtruth
% get methods and compare them
% a database has the following structure
% DATABASE
% |___ groundtruth
%      |___ images and pts files
%      Method1
%      Method2
% stats should be a struct generated from statistics/getStats
% attributes should be one or more of the attributes in the list in form
% of tuples
% attributes = 'yaw','roll','pitch','emotion','glasses','beard';
%default to using all points in db
%pointScheme = '68';
name = strsplit(pathToDatabase,'/');
name = name{end};

visiblePointsDB = {'groundtruth','CFSS','curDlib','RCN','meanshape','DLIB','sdm','mdm','mtcnn'};

invisiblePointsDB = {'groundtruth', '3DDFA', 'JointPDM'};
if strcmp(pointScheme,'68') ~= 1
	invisiblePointsDB{end+1} = 'PAWF';
end
if visiblePoints==1
    db = visiblePointsDB;
    ptsExt = '*.pts';
else
    db = invisiblePointsDB;
    %pointScheme = '68';
	ptsExt = '*.pts1';
end
% read databse and check for consistency
gtPath = [pathToDatabase '/groundtruth/'];
error =[];
fname = {};
ptsGt = dir([gtPath ptsExt]);
ptsGt = {ptsGt.name};

numPts = length(ptsGt);
fprintf('%s has %d %s files\n', pathToDatabase, numPts, ptsExt);

%find methods
methods = dir(pathToDatabase);
methods = {methods([methods(:).isdir]).name};
methods = setdiff(methods, {'..','.'});

% filter out not used names
newM = {};
for i = 1:length(db)
    x = cellfun(@(x) strncmp(x,db{i},3), methods);
    newM = [ newM methods{x}];
end
methods = newM;
fprintf('Found databases:\n');
for i = 1:size(methods,2)
    disp(methods{i});
    methodsDir{i} = dir([pathToDatabase '/' methods{i} '/' ptsExt]);
    if size(methodsDir{i},1) ~= numPts
        fprintf('%s has %d %s files but should have %d\n',methods{i},size(methodsDir{i},1), ptsExt, numPts);
    end
end
numParts = 1;
error = zeros(size(methods,2),size(ptsGt,2),numParts);

parfor i = 1:size(ptsGt,2)

    gtpathfile = [gtPath ptsGt{i}];
    gt = readpts(gtpathfile);
	
    [gtIndices, lefteye, righteye] = getNormalIndices('groundtruth',pointScheme, length(gt));
   	
	
	if strcmp(normalization,'occular')
		eyescale = mean(gt(lefteye,:),1) - mean(gt(righteye,:),1);
		eyescale = sqrt(sum(eyescale.^2,2));
		normdist = eyescale;
	else
		boxscale =  (max(gt)-min(gt));
		boxscale = sqrt(boxscale(1) * boxscale(2));
		normdist = boxscale;
	end
		
    [~,fn,~] = fileparts(gtpathfile);
    fname{i} = fn;
    xx = zeros(size(methods,2), numParts);
    for j= 1:size(methods,2)
        % get points from testfile
        landmarkFile= [pathToDatabase '/' methods{j} '/' ptsGt{i}];
        % if we have no pts files set error to 0 
        if ~exist(landmarkFile, 'file')			
            xx(j,:) = 0;
            continue
        end
        ptsmethod = readpts(landmarkFile);
        methodindices = getNormalIndices(methods{j}, pointScheme, length(ptsmethod));
		methpts = ptsmethod(methodindices,1:2);
		
		gtbla = gt(gtIndices,:);
		numpoints = length(gtIndices);
		
		%if 5 points average eyes to center 
		if numpoints == 5
			gtbla(1,:) = mean(gt(lefteye,:),1);
			gtbla(2,:) = mean(gt(righteye,:),1);
			if length(ptsmethod) > 5
				methpts(1,:) = mean(ptsmethod(lefteye,:),1);
				methpts(2,:) = mean(ptsmethod(righteye,:),1);
			end
		end
		% check if visibility info is available
		if size(gt,2) == 3
			ind=find(gt(:,3)==0);
			methpts = methpts(ind,:);
			gtbla = gtbla(ind,1:2);
			numpoints = length(ind);			
		end
		for k=1:numParts
			if k == 2
				facePartIndices = 18:27;
			elseif k==3
				facePartIndices = 49:68;
			elseif k == 4 
				facePartIndices = 1:17;
			else
				facePartIndices = 1:numpoints;
			end

			numpoints = length(facePartIndices);
		
			gtfacepart = gtbla([facePartIndices],:);
			methptsfacepart = methpts([facePartIndices],:);
			ini = gtfacepart == -1;
	
			methptsfacepart(ini) = -1;
		
			errorSingle = gtfacepart- methptsfacepart;
	
			% if visiblePoints ==1
			% following formula 4 from https://arxiv.org/pdf/1511.07356v2.pdf
			% normalize by interocular eye dist      
			% else
			% following 6.3.1 of https://arxiv.org/abs/1511.07212
			%error(j,i)  = (sum(sqrt(sum(errorSingle.^2,2))) / boxscale) / numpoints;
			xx(j,k)  = (sum(sqrt(sum(errorSingle.^2,2))) / normdist) / numpoints;
			%end
		end
    end
    error(:,i,:) = xx;
end

disp(error(1,1));
if visiblePoints==1
    save(['results' name normalization],'error', 'methods','fname');
else
    save(['resultsInvisible' name normalization],'error', 'methods','fname');
end


end


