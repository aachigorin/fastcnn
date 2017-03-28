function [total_boxes, points, t, nmsCount] = detect_face2(img,minsize,PNet,RNet,ONet,threshold,fastresize,factor, nmsCount, pointoutputs, extensionfactor)
	%im: input image
	%minsize: minimum of faces' size
	%pnet, rnet, onet: caffemodel
	%threshold: threshold=[th1 th2 th3], th1-3 are three steps's threshold
	%fastresize: resize img from last scale (using in high-resolution images) if fastresize==true
	
    factor_count=0;
	total_boxes=[];
	points=[];
	h=size(img,1);
	w=size(img,2);
	minl=min([w h]);
    img=single(img);
	if fastresize
		im_data=(single(img)-127.5)*0.0078125;
    end
     m=12/minsize;
	minl=minl*m;
	%creat scale pyramid
    scales=[];
	while (minl>=12)
		scales=[scales m*factor^(factor_count)];
		minl=minl*factor;
		factor_count=factor_count+1;
	end
	%first stage    
    %tic
    t = zeros(1,10);
	for j = 1:size(scales,2)
		scale=scales(j);
		hs=ceil(h*scale);
		ws=ceil(w*scale);
		if fastresize
			im_data=imResample(im_data,[hs ws],'bilinear');
        else
            tic
			im_data=(imResample(img,[hs ws],'bilinear')-127.5)*0.0078125;            
            t(1) = t(1) + toc;
		end
		PNet.blobs('data').reshape([hs ws 3 1]);
        tic        
        out=PNet.forward({im_data});     
        t(2) = t(2) + toc; tic
		%boxes=generateBoundingBox(out{2}(:,:,2),out{1},scale,threshold(1));
        %size(boxes,1) / max(size(im_data))
        boxes=generateBoundingBox2(out{2}(:,:,2),out{1}, scale, int16(ceil(max(size(im_data))) * 0.35));
        t(3) = t(3) + toc; tic
		%inter-scale nms
		pick=nms(boxes,0.5,'Union');
        t(4) = t(4) + toc;
		boxes=boxes(pick,:);
		if ~isempty(boxes)
			total_boxes=[total_boxes;boxes];
		end
    end	
	if ~isempty(total_boxes),
        tic
		pick=nms(total_boxes,0.7,'Union');
        nmsCount(1) = nmsCount(1) + length(pick);
        t(4) = t(4) + toc;
		total_boxes=total_boxes(pick,:);
		regw=total_boxes(:,3)-total_boxes(:,1);
		regh=total_boxes(:,4)-total_boxes(:,2);
		total_boxes=[total_boxes(:,1)+total_boxes(:,6).*regw total_boxes(:,2)+total_boxes(:,7).*regh total_boxes(:,3)+total_boxes(:,8).*regw total_boxes(:,4)+total_boxes(:,9).*regh total_boxes(:,5)];	
		
        
%          pick=nms(total_boxes,0.7,'Min');
%          nmsCount(2) = nmsCount(2) + length(pick);
%          total_boxes=total_boxes(pick,:);
         
        total_boxes=rerec(total_boxes);
		total_boxes(:,1:4)=fix(total_boxes(:,1:4));
		[dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]=pad(total_boxes,w,h);
	end
	numbox=size(total_boxes,1);
    %fprintf('1 stage t = %f ms; scale = %d\n', toc*1000, size(scales,2));
	if numbox>0
		%second stage
        %tic
 		tempimg=zeros(24,24,3,numbox);
		for k=1:numbox
			tmp=zeros(tmph(k),tmpw(k),3);
			tmp(dy(k):edy(k),dx(k):edx(k),:)=img(y(k):ey(k),x(k):ex(k),:);
            tic
			tempimg(:,:,:,k)=imResample(tmp,[24 24],'bilinear');
            t(5) = t(5) + toc;
		end
        tempimg=(tempimg-127.5)*0.0078125;
       	RNet.blobs('data').reshape([24 24 3 numbox]);
        tic
		out=RNet.forward({tempimg}); 
        t(6) = t(6) + toc;
		score=squeeze(out{2}(2,:));
		pass=find(score>threshold(2));
		total_boxes=[total_boxes(pass,1:4) score(pass)'];
		mv=out{1}(:,pass);
		if size(total_boxes,1)>0
            tic
			pick=nms(total_boxes,0.7,'Union');
            nmsCount(3) = nmsCount(3) + length(pick);
            t(7) = t(7) + toc;
		    total_boxes=total_boxes(pick,:);     
            total_boxes=bbreg(total_boxes,mv(:,pick)');
           % total_boxes=bbreg(total_boxes,mv');
            total_boxes=rerec(total_boxes);
            
%              pick=nms(total_boxes,0.5,'Min');
%              nmsCount(4) = nmsCount(4) + length(pick);
%              total_boxes=total_boxes(pick,:);
		end
		numbox=size(total_boxes,1);
        %fprintf('2 stage t = %f ms\n', toc*1000);
		if numbox>0
			%third stage
           % tic
			if extensionfactor == 1
				total_boxes=fix(total_boxes);
			else
				brh = total_boxes(:,4) - total_boxes(:,2);
				brw = total_boxes(:,3) - total_boxes(:,1);
				fact = (extensionfactor -1) /2;
				total_boxes(:,1) = total_boxes(:,1) - brw*fact;
				total_boxes(:,2) = total_boxes(:,2) - brh*fact;
				total_boxes(:,3) = total_boxes(:,3) + brw*fact;
				total_boxes(:,4) = total_boxes(:,4) + brh*fact;
				total_boxes=fix(total_boxes(:,1:4));
			end
			[dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]=pad(total_boxes,w,h);
            tempimg=zeros(48,48,3,numbox);
			for k=1:numbox
				tmp=zeros(tmph(k),tmpw(k),3);
				tmp(dy(k):edy(k),dx(k):edx(k),:)=img(y(k):ey(k),x(k):ex(k),:);
                tic
				tempimg(:,:,:,k)=imResample(tmp,[48 48],'bilinear');
                 t(8) = t(8) + toc;
			end
			tempimg=(tempimg-127.5)*0.0078125;            
			ONet.blobs('data').reshape([48 48 3 numbox]);
            tic
			out=ONet.forward({tempimg});
            t(9) = t(9) + toc;
			score=squeeze(out{pointoutputs+2}(2,:));

	
			points=out{2};
			pass=find(score>threshold(3));
			points=points(:,pass);
			total_boxes=[total_boxes(pass,1:4) score(pass)'];
			mv=out{1}(:,pass);
			w=total_boxes(:,3)-total_boxes(:,1)+1;
            h=total_boxes(:,4)-total_boxes(:,2)+1;
            points(1:5,:)=repmat(w',[5 1]).*points(1:5,:)+repmat(total_boxes(:,1)',[5 1])-1;
            points(6:10,:)=repmat(h',[5 1]).*points(6:10,:)+repmat(total_boxes(:,2)',[5 1])-1;
			if size(total_boxes,1)>0				
				% for widergt3 (new dataset)
				total_boxes=bbreg(total_boxes,mv(:,:)');
				
                %total_boxes=bbreg2(total_boxes,mv(:,:)');
				
				% for widergt2 (old dataset)
                %total_boxes=bbreg3(total_boxes,mv(:,:)');

                tic
                pick=nms(total_boxes,0.7,'Min');
				t(10) = t(10) + toc;
                t = pick;
                total_boxes=total_boxes(pick,:);  				
                points=points(:,pick);
            end
            %fprintf('3 stage t = %f ms\n', toc*1000);
		end
    end 	
end

