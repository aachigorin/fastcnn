function mat_to_caffe(config_path, target_dir, target_model_name)
    curDir = pwd;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	caffe_path='/media/p.omenitsch/code/tests/CVPR16-LargePoseFaceAlignment/Caffe-FaceAlignment/matlab';
	pdollar_toolbox_path='/media/p.omenitsch/tools/toolbox';
	addpath(genpath(caffe_path));
	addpath(genpath(pdollar_toolbox_path));
	%config_path = '/media/a.chigorin/code/fastcnn/lcnn_mtcnn_onet/convert_to_caffe/';
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    ONet = caffe.Net(config_path, 'test');

    model_dir = [target_dir];

    disp('loading weights')
    conv0_w = load([model_dir 'conv0_weights.mat']); conv0_w = conv0_w.x;
    conv2_w = load([model_dir 'conv2_weights.mat']); conv2_w = conv2_w.x;
    conv4_w = load([model_dir 'conv4_weights.mat']); conv4_w = conv4_w.x;
    conv6_w = load([model_dir 'conv6_weights.mat']); conv6_w = conv6_w.x;
    conv7_w = load([model_dir 'conv7_weights.mat']); conv7_w = conv7_w.x;
    conv_face_w = load([model_dir 'conv_face_weights.mat']); conv_face_w = conv_face_w.x;
    conv_bbox_w = load([model_dir 'conv_bbox_weights.mat']); conv_bbox_w = conv_bbox_w.x;
    conv_landmarks_w = load([model_dir 'conv_landmarks_weights.mat']); conv_landmarks_w = conv_landmarks_w.x;

    disp('loading biases')
	conv0_b = load([model_dir 'conv0_biases.mat']); conv0_b = conv0_b.x;
    conv2_b = load([model_dir 'conv2_biases.mat']); conv2_b = conv2_b.x;
    conv4_b = load([model_dir 'conv4_biases.mat']); conv4_b = conv4_b.x;
    conv6_b = load([model_dir 'conv6_biases.mat']); conv6_b = conv6_b.x;
    conv7_b = load([model_dir 'conv7_biases.mat']); conv7_b = conv7_b.x;
    conv_face_b = load([model_dir 'conv_face_biases.mat']); conv_face_b = conv_face_b.x;
    conv_bbox_b = load([model_dir 'conv_bbox_biases.mat']); conv_bbox_b = conv_bbox_b.x;
    conv_landmarks_b = load([model_dir 'conv_landmarks_biases.mat']); conv_landmarks_b = conv_landmarks_b.x;

	disp('setting weights')
    ONet.layers('conv0').params(1).set_data(conv0_w)
    ONet.layers('conv2').params(1).set_data(conv2_w)
    ONet.layers('conv4').params(1).set_data(conv4_w)
    ONet.layers('conv6').params(1).set_data(conv6_w)
    ONet.layers('conv7').params(1).set_data(conv7_w)
    ONet.layers('conv_face').params(1).set_data(reshape(conv_face_w, [4096, 2]))
    ONet.layers('conv_bbox').params(1).set_data(reshape(conv_bbox_w, [4096, 4]))
    ONet.layers('conv_landmarks').params(1).set_data(reshape(conv_landmarks_w, [4096, 10]))

    disp('setting biases')
	ONet.layers('conv0').params(2).set_data(conv0_b')
    ONet.layers('conv2').params(2).set_data(conv2_b')
    ONet.layers('conv4').params(2).set_data(conv4_b')
    ONet.layers('conv6').params(2).set_data(conv6_b')
    ONet.layers('conv7').params(2).set_data(conv7_b')
    ONet.layers('conv_face').params(2).set_data(conv_face_b')
    ONet.layers('conv_bbox').params(2).set_data(conv_bbox_b')
    ONet.layers('conv_landmarks').params(2).set_data(conv_landmarks_b')

    disp('saving the model')
	ONet.save([target_dir target_model_name '.caffemodel']);

	disp('forward pass')
	input_size = 48;
	test_mat = ones(input_size,input_size,3);
	for i = 1:input_size
	    for j = 1:input_size
	        for k = 1:3
	            test_mat(i,j,k) = (i-1) + (j-1) + (k-1);
	        end
	    end
	end
	%disp(test_mat)

    out = ONet.forward({test_mat})
    disp(out{1})


    % checking the results
    % check conv0
    conv0_caffe_outputs = ONet.blobs('conv0_conv').get_data();
    %disp(size(conv0_caffe_outputs))
    %disp(conv0_caffe_outputs(:, 1, 1))
    bn1_out = load([model_dir 'bn1_outputs.mat']); bn1_out = bn1_out.x;
    %disp(size(bn1_out))
    %disp(bn1_out(:, :, 1, 1))
    disp('conv0 outputs are equal? ')
    equal = abs(conv0_caffe_outputs(:) - bn1_out(1, :)') < 0.001;
    %disp(equal)
    disp(all(equal))

    % check conv2
    conv2_caffe_outputs = ONet.blobs('conv2_conv').get_data();
    %disp(size(conv2_caffe_outputs))
    %disp(conv2_caffe_outputs(:, 1, 1))
    bn2_out = load([model_dir 'bn2_outputs.mat']); bn2_out = bn2_out.x;
    %disp(size(bn2_out))
    %disp(bn2_out(:, :, 1, 1))
    disp('conv2 outputs are equal? ')
    equal = abs(conv2_caffe_outputs(:) - bn2_out(1, :)') < 0.001;
    %disp(equal)
    disp(all(equal))

    % check conv4
    conv4_caffe_outputs = ONet.blobs('conv4_conv').get_data();
    %disp(size(conv4_caffe_outputs))
    %disp(conv4_caffe_outputs(:, 1, 1))
    bn3_out = load([model_dir 'bn3_outputs.mat']); bn3_out = bn3_out.x;
    %disp(size(bn3_out))
    %disp(bn3_out(:, :, 1, 1))
    disp('conv4 outputs are equal? ')
    equal = abs(conv4_caffe_outputs(:) - bn3_out(1, :)') < 0.001;
    %disp(equal)
    disp(all(equal))

    % check conv6
    conv6_caffe_outputs = ONet.blobs('conv6_conv').get_data();
    %disp(size(conv6_caffe_outputs))
    %disp(conv6_caffe_outputs(:, 1, 1))
    bn4_out = load([model_dir 'bn4_outputs.mat']); bn4_out = bn4_out.x;
    %disp(size(bn4_out))
    %disp(bn4_out(:, :, 1, 1))
    disp('conv6 outputs are equal? ')
    equal = abs(conv6_caffe_outputs(:) - bn4_out(1, :)') < 0.001;
    %disp(equal)
    disp(all(equal))

    % check avg_pool
    %avg_pool_caffe_outputs = ONet.blobs('avg_pool').get_data();
    %disp(size(avg_pool_caffe_outputs))
    %disp(avg_pool_caffe_outputs(:, 1, 1))
    %avg_pool_out = load([model_dir 'avg_pool_outputs.mat']); avg_pool_out = avg_pool_out.x;
    %disp(size(avg_pool_out))
    %disp(avg_pool_out(:, :, 1, 1))
    %disp('avg_pool_out outputs are equal? ')
    %equal = abs(avg_pool_caffe_outputs(:) - avg_pool_out(1, :)') < 0.001;
    %disp(equal)
    %disp(all(equal))

    % check face softmax
    face_softmax_caffe_outputs = ONet.blobs('face_prob').get_data();
    %disp(size(face_softmax_caffe_outputs))
    %disp(face_softmax_caffe_outputs(:))

    face_softmax_out = load([model_dir 'face_softmax_outputs.mat']); face_softmax_out = face_softmax_out.x;
    %disp(size(face_softmax_out))
    %disp(face_softmax_out(:))

    disp('face_softmax_out outputs are equal? ')
    equal = abs(face_softmax_caffe_outputs - face_softmax_out') < 0.001;
    %disp(equal)
    disp(all(equal))

    % check bbox outputs
    bbox_caffe_outputs = ONet.blobs('conv_bbox').get_data();
    %disp(size(bbox_caffe_outputs))
    %disp(bbox_caffe_outputs(:))

    bbox_out = load([model_dir 'bbox_outputs.mat']); bbox_out = bbox_out.x;
    %disp(size(bbox_out))
    %disp(bbox_out(:))

    disp('bbox_out outputs are equal? ')
    equal = abs(bbox_caffe_outputs - bbox_out') < 0.001;
    %disp(equal)
    disp(all(equal))

    % check bbox outputs
    landmarks_caffe_outputs = ONet.blobs('conv_landmarks').get_data();
    %disp(size(landmarks_caffe_outputs))
    %disp(landmarks_caffe_outputs(:))

    landmarks_out = load([model_dir 'landmarks_outputs.mat']); landmarks_out = landmarks_out.x;
    %disp(size(landmarks_out))
    %disp(landmarks_out(:))

    disp('landmarks_out outputs are equal? ')
    equal = abs(landmarks_caffe_outputs - landmarks_out') < 0.001;
    %disp(equal)
    disp(all(equal))

    disp('finished')