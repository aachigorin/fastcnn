function [boundingbox, reg] = generateBoundingBox2(map,reg,scale, num)
	%use heatmap to generate bounding boxes
    t = 0.1;
    stride=2;
    cellsize=12;    
	%map=map';
	dx1=reg(:,:,1);
	dy1=reg(:,:,2);
	dx2=reg(:,:,3);
	dy2=reg(:,:,4);
    [y, x]=find(map>=t);
	a=find(map>=t); 
    if size(y,1)==1
		y=y';x=x';score=map(a)';dx1=dx1';dy1=dy1';dx2=dx2';dy2=dy2';
	else
		score=map(a);
    end   
	reg=[dx1(a) dy1(a) dx2(a) dy2(a)];
	if isempty(reg)
		reg=reshape([],[0 3]);
	end
    %boundingbox=[y x];
    boundingbox=[x y];
    boundingbox=[fix((stride*(boundingbox-1)+1)/scale) fix((stride*(boundingbox-1)+cellsize-1+1)/scale) score reg];
    if num < size(boundingbox,1)
        [~, ind] = sort(boundingbox(:,5), 'descend');
        boundingbox = boundingbox(ind(1:num),:);
    end
end

