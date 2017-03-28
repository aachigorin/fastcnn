function [boundingbox] = bbreg3(boundingbox,reg)
	%calibrate bouding boxes
	if size(reg,2)==1
		reg=reshape(reg,[size(reg,3) size(reg,4)])';
    end
%     bboxNew = zeros(size(boundingbox));
%     bboxNew(:,5) = boundingbox(:,5);
	w=boundingbox(:,3)-boundingbox(:,1)+1;
	h=boundingbox(:,4)-boundingbox(:,2)+1;
    boundingbox(:,1)=boundingbox(:,1)+reg(:,1).*w;
    boundingbox(:,2)= boundingbox(:,2)+reg(:,2).*h; 
    boundingbox(:,3)= boundingbox(:,1)+(1+reg(:,3)).*w;
    boundingbox(:,4)=boundingbox(:,2)+(1+reg(:,4)).*h;
end