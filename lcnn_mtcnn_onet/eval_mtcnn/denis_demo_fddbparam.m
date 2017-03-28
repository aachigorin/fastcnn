function denis_demo_fddbparam(onet_config_path, onet_model_path, model_name, numpoints, pointoutputs, extensionfactor)
%clear all;
visible = 0;
write = 0;
%list of images

imPath = '/media/rybalchenko/Evaluation_Code/evaluation/';


imglist = importdata('FDDBlist.txt');

%minimum size of face
minsize = 20;
%three steps's threshold




%scale factor
factor=0.703;

%path of toolbox
%caffe dir

caffe_path='/media/p.omenitsch/code/tests/CVPR16-LargePoseFaceAlignment/Caffe-FaceAlignment/matlab';


addpath(genpath(caffe_path));
pdollar_toolbox_path='/media/p.omenitsch/tools/toolbox'
caffe_model_path = './mtcnn_pnet_rnet_models';
%caffe_model_path3 = '/media/p.omenitsch/code/facedet/MTCNN_train/convert/ONetmodel';

addpath(genpath(pdollar_toolbox_path));


caffe.reset_all();
caffe.set_mode_gpu();	
%caffe.set_device(gpu_id);



%load caffe models
prototxt_dir =strcat(caffe_model_path,'/det1.prototxt');
fid = fopen([model_name '.txt'], 'w');

model_dir = strcat(caffe_model_path,'/det1.caffemodel');
threshold = [0.6 0.7 0.1];


PNet=caffe.Net(prototxt_dir,model_dir,'test');
prototxt_dir = strcat(caffe_model_path,'/det2.prototxt');
model_dir = strcat(caffe_model_path,'/det2.caffemodel');

%model_dir = strcat(caffe_model_path,'/sRNet930.caffemodel');
%threshold(2) = 0.5;

RNet=caffe.Net(prototxt_dir,model_dir,'test');	
%if pointoutputs == 1
%	if numpoints == 5
%		prototxt_dir = strcat(caffe_model_path,'/det3.prototxt');
%	elseif numpoints ==68
%		prototxt_dir = strcat(caffe_model_path3,'/det368.prototxt');
%	end
%elseif pointoutputs == 2
%		prototxt_dir = strcat(caffe_model_path3,'/det3568.prototxt');
%end

%prototxt_dir = '/media/a.chigorin/code/fastcnn/lcnn_mtcnn_onet/eval_mtcnn/convert_to_caffe/mtcnn_onet_5points.prototxt'
ONet=caffe.Net(onet_config_path, onet_model_path, 'test');
faces=cell(0);

disp('All nets are loaded. Starting to detect faces')
num = length(imglist)
disp(num)
t = zeros(num, 10);
nmsCount = zeros(1,4);
for i=1:num%length(imglist)
    fprintf('%i from %i\n', i, length(imglist));
    img = imread([imPath imglist{i}]);
   

    if size(img,3) < 3
        img(:,:,2) = img(:,:,1);
        img(:,:,3) = img(:,:,1);
    end
    
    %we recommend you to set minsize as x * short side
	%minl=min([size(img,1) size(img,2)]);
	%minsize=fix(minl*0.1)
    sss(i, :) = size(img);
    [boudingboxes, points, asdf, nmsCount]=detect_face2(img,minsize,PNet,RNet,ONet,threshold,false,factor, nmsCount,pointoutputs, extensionfactor);

    
    faces{i,1}={boudingboxes};
	faces{i,2}={points'};
    
    numbox=size(boudingboxes,1);
    fprintf(fid, '%s\n%d\n', imglist{i}(1:end-4), numbox);
    for j = 1:numbox
        fprintf(fid, '%3.2f %3.2f %3.2f %3.2f %1.3f\n', [boudingboxes(j,1:2) boudingboxes(j,3:4)-boudingboxes(j,1:2) boudingboxes(j,5)]);        
    end
    


end
t = t * 1000;
[mean(t(2:end,:),1), sum(mean(t(2:end,:),1))]

fclose(fid);
caffe.reset_all();


%save result box landmark