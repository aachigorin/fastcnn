function demo1(ptsdb, modelpath, targetpath, testfolder, numpoints, pointoutputs, extensionfactor)
    %list of images
    imglist=importdata(ptsdb);
    %imglist = {};
    %imglist = {'/media/p.omenitsch/code/tests/mdm/databases/300W/afw/134212_1.jpg'};
    %minimum size of face
    minsize=20;

    %path of toolbox
    caffe_path='/media/p.omenitsch/code/tests/CVPR16-LargePoseFaceAlignment/Caffe-FaceAlignment/matlab';
    pdollar_toolbox_path='/media/p.omenitsch/tools/toolbox'
    caffe_model_path='./mtcnn_pnet_rnet_models'
    addpath('./');
    addpath(genpath(caffe_path));
    addpath(genpath(pdollar_toolbox_path));
    addpath('/media/p.omenitsch/code/helper');
    %use cpu
    %caffe.set_mode_cpu();
    gpu_id=0;
    caffe.set_mode_gpu();
    caffe.set_device(gpu_id);

    %three steps's threshold
    threshold=[0.6 0.7 0.7]

    %scale factor
    factor=0.709;

    %load caffe models
    prototxt_dir =strcat(caffe_model_path,'/det1.prototxt');
    model_dir = strcat(caffe_model_path,'/det1.caffemodel');
    PNet=caffe.Net(prototxt_dir,model_dir,'test');
    prototxt_dir = strcat(caffe_model_path,'/det2.prototxt');
    model_dir = strcat(caffe_model_path,'/det2.caffemodel');
    RNet=caffe.Net(prototxt_dir,model_dir,'test');
    %if pointoutputs == 1
    %	if numpoints == 5
    %	prototxt_dir = strcat('/media/p.omenitsch/code/facedet/MTCNN_train/convert/ONetmodel','/det3.prototxt');
    %	elseif numpoints == 68
    %	prototxt_dir = strcat('/media/p.omenitsch/code/facedet/MTCNN_train/convert/ONetmodel','/det368.prototxt');
    %	end
    %elseif pointoutputs == 2
    %		prototxt_dir = strcat('/media/p.omenitsch/code/facedet/MTCNN_train/convert/ONetmodel','/det3568.prototxt');
    %end
    prototxt_dir = '/media/a.chigorin/code/fastcnn/lcnn_mtcnn_onet/convert_to_caffe/mtcnn_onet_5points.prototxt'
    model_dir = [modelpath '.caffemodel'];
    ONet=caffe.Net(prototxt_dir, model_dir, 'test');
    faces=cell(0);
    largeface = 0;
    noface = 0;
    total_boxes = [];
    imnames = {};
    for i=1: length(imglist)

        disp(i)

        img=imread(imglist{i});
        if max(size(img)) > 3500
            largeface = largeface+1;
            continue;
        end
        if size(img,3) == 1
            img  = cat(3, img, img, img);
        end
        img = im2uint8(img);

        %we recommend you to set minsize as x * short side
        %minl=min([size(img,1) size(img,2)]);
        %minsize=fix(minl*0.1)
        tic
        [boudingboxes points]=detect_face_on_image(img,minsize,PNet,RNet,ONet,threshold,false,factor,numpoints, ...
                                                   pointoutputs,extensionfactor);
        toc
        [a,b,c] = fileparts(imglist{i});

        ptsgt = readpts([targetpath '/groundtruth/' b '.pts']);
        ptsgt = ptsgt(ptsgt(:,1) > 0,:);
        if numel(ptsgt) > 2
            ptsgt = mean(ptsgt);
        end
        if numel(ptsgt) < 2
            continue;
        end

        if length(points) < numpoints*2
            noface = noface+1;
            disp('no points');
            continue
        end
        bbox = [];

        filter = zeros(size(boudingboxes,1),1);
        for k = 1:size(boudingboxes,1)

            if ptsgt(1) > boudingboxes(k,1) & ptsgt(1) < boudingboxes(k,3) & ptsgt(2) > boudingboxes(k,2) & ptsgt(2) < boudingboxes(k,4)
                filter(k) = 1;
            end
        end
        boudingboxes(~logical(filter),5) = 0;
        [val, pos] = max(boudingboxes(:,5));
        pointscur = points(:,pos);
        bbox = boudingboxes(pos,:);
        if numel(pointscur) ~= numpoints*2
            continue;
        end



        disp([targetpath '/' testfolder '/'  b '.pts'])
        dlmwrite([targetpath '/' testfolder '/' b '.pts'], reshape(pointscur',[numpoints 2]),' ');
        %show detection result

    end
    disp('no face');
    disp(noface);
    disp('large face');
    disp(largeface);
    caffe.reset_all();
    %save('boundingboxesAFLW.mat', 'imnames','total_boxes');
    %save(name, 'faces');