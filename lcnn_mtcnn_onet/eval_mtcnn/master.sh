#!/bin/bash
set -e

# this script can
# train a model
# convert from torch to matlab (caffe)
# evaluate the model on FDDB for face detection
# evalute the model on AFLW2000 or AFLW for 68 / 5 points
# draw plots for the evaluations

start_from=$1
echo "starting from step $start_from"

#PTSDB=/media/data/experiments/AFLW2000.txt
#PTSTESTDB=/media/data/experiments/facepoints/AFLW2000
#DBMODEL_LABEL=AFLW2000

PTSDB=/media/data/experiments/aflw.txt
PTSTESTDB=/media/data/experiments/facepoints/AFLW
DBMODEL_LABEL=AFLW

GPU=0
NUMPOINTS=5
#EXTENSIONFACTOR=1 # (for the old widerface dataset)
EXTENSIONFACTOR=1.11 # (for the new widerface dataset)
POINTOUTPUTS=1

CONVERT_TO_CAFFE_DIR=/media/a.chigorin/code/fastcnn/lcnn_mtcnn_onet/eval_mtcnn/convert_to_caffe/
CAFFE_WEIGHTS_DIR=/media/a.chigorin/code/fastcnn/lcnn_mtcnn_onet/eval_mtcnn/weights/

#TF_SAVE_MODEL_CMD="CUDA_VISIBLE_DEVICES=$GPU python ${CONVERT_TO_CAFFE_DIR}/save_tf_weights.py --train_dir /media/a.chigorin/projects/fastcnn/mtcnn_onet/exp8 --save_weights_to ${CAFFE_WEIGHTS_DIR}"
TF_SAVE_MODEL_CMD="CUDA_VISIBLE_DEVICES=$GPU python ${CONVERT_TO_CAFFE_DIR}/save_tf_weights.py --train_dir /media/a.chigorin/projects/fastcnn/mtcnn_onet/exp13 --save_weights_to ${CAFFE_WEIGHTS_DIR} --no_global_avg_pool True"

PTSTESTFOLDER=mtcnn_no_global_pool_5points_aachigorin
PLOTDIRECTORY=/media/a.chigorin/code/fastcnn/lcnn_mtcnn_onet/eval_mtcnn/plots/

MODEL_LABEL=OnetTest
MODEL_CONFIG_PATH=${CONVERT_TO_CAFFE_DIR}/mtcnn_onet_5points.prototxt
MODEL_WEIGHTS_PATH=${CAFFE_WEIGHTS_DIR}/${MODEL_LABEL}


# creating folders 
mkdir -p ${PTSTESTDB}/${PTSTESTFOLDER}

if [ $start_from -le 0 ]
then
    # train model
    #./trainMTCNN.sh $MODEL_LABEL $CAFFE_WEIGHTS_DIR $GPU

    # convert model
    eval $TF_SAVE_MODEL_CMD
fi


if [ $start_from -le 1 ]
then
    CMD="LD_PRELOAD=/usr/lib64/libstdc++.so.6 LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:/media/p.omenitsch/tools/caffelib/:"\
"/media/p.omenitsch/tools/OpenBLAS/lib /media/p.omenitsch/matgpu -r \"mat_to_caffe('${MODEL_CONFIG_PATH}', ''${CAFFE_WEIGHTS_DIR}','${MODEL_LABEL}'); exit; \""
    cd ${CONVERT_TO_CAFFE_DIR}
    echo $CMD
    eval $CMD
    cd -
fi


if [ $start_from -le 2 ]
then
    # run on landmark set and draw graph
    CMD="CUDA_VISIBLE_DEVICES=$GPU LD_PRELOAD=/usr/lib64/libstdc++.so.6 LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:"\
"/media/p.omenitsch/tools/caffelib/:/media/p.omenitsch/tools/OpenBLAS/lib /media/p.omenitsch/matgpu -r "\
"\"demo1('${MODEL_CONFIG_PATH}', '${PTSDB}', '${CAFFE_WEIGHTS_DIR}${MODEL_LABEL}', '${PTSTESTDB}','${PTSTESTFOLDER}',${NUMPOINTS},${POINTOUTPUTS},${EXTENSIONFACTOR}); exit; \""
    pwd
    echo $CMD
    #cd /media/p.omenitsch/code/facedet/MTCNN_face_detection_alignment/code/codes/MTCNNv1
    eval $CMD
    #cd /media/p.omenitsch/code/facedet/MTCNN_train
fi

if [ $start_from -le 3 ]
then
    # run on detection and draw graph
    CMD="CUDA_VISIBLE_DEVICES=$GPU LD_PRELOAD=/usr/lib64/libstdc++.so.6 LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:"\
"/media/p.omenitsch/tools/caffelib/:/media/p.omenitsch/tools/OpenBLAS/lib /media/p.omenitsch/matgpu -r "\
"\"denis_demo_fddbparam('${MODEL_CONFIG_PATH}', '${CAFFE_WEIGHTS_DIR}${MODEL_LABEL}.caffemodel', '${MODEL_LABEL}', ${NUMPOINTS}, ${POINTOUTPUTS}, ${EXTENSIONFACTOR}); exit; \""
    echo $CMD
    #cd /media/p.omenitsch/code/facedet/MTCNN_face_detection_alignment/code/codes/MTCNNv1/denistest
    eval $CMD
    #cd /media/p.omenitsch/code/facedet/MTCNN_train

    #cd /media/rybalchenko/Evaluation_Code/evaluation
    cd fddb_evaluation
    # ./evaluate -a anno.txt -d /media/p.omenitsch/code/facedet/MTCNN_face_detection_alignment/code/codes/MTCNNv1/denistest/${MODEL_LABEL}.txt -i ./ -l filelist.txt -r /media/p.omenitsch/code/facedet/MTCNN_train/plots/${MODEL_LABEL}
    CMD="./evaluate -a anno.txt -d ../${MODEL_LABEL}.txt -i ./ -l filelist.txt -r ${PLOTDIRECTORY}/${MODEL_LABEL}"
    echo $CMD
    eval $CMD
fi


if [ $start_from -le 4 ]
then
    # now run graphs
    CMD="CUDA_VISIBLE_DEVICES=$GPU LD_PRELOAD=/usr/lib64/libstdc++.so.6 LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:"\
"/media/p.omenitsch/tools/caffelib/:/media/p.omenitsch/tools/OpenBLAS/lib /media/p.omenitsch/matgpu -r "\
"\"testDB('${PTSTESTDB}', '${NUMPOINTS}',1,'box'); exit; \""
    echo $CMD
    cd /media/p.omenitsch/code/facelandmarktesting
    eval $CMD
    CMD="CUDA_VISIBLE_DEVICES=$GPU LD_PRELOAD=/usr/lib64/libstdc++.so.6 LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:"\
"/media/p.omenitsch/tools/caffelib/:/media/p.omenitsch/tools/OpenBLAS/lib /media/p.omenitsch/matgpu -r "\
"\"plotResultsCED('results${DBMODEL_LABEL}box'); exit; \""
    echo $CMD
    eval $CMD
    cp imgs/results${DBMODEL_LABEL}boxfp1.png ${PLOTDIRECTORY}
    cd /media/a.chigorin/code/fastcnn/lcnn_mtcnn_onet/eval_mtcnn/
fi

if [ $start_from -le 5 ]
then
    # plot detection results
    CMD="CUDA_VISIBLE_DEVICES=$GPU LD_PRELOAD=/usr/lib64/libstdc++.so.6 LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:"\
"/media/p.omenitsch/tools/caffelib/:/media/p.omenitsch/tools/OpenBLAS/lib /media/p.omenitsch/matgpu -r "\
"\"plotDetection('${MODEL_LABEL}DiscROC', '${PLOTDIRECTORY}','${PLOTDIRECTORY}'); exit; \""
    echo $CMD
    eval $CMD


    # plot curves for training
    #CMD="CUDA_VISIBLE_DEVICES=$GPU LD_PRELOAD=/usr/lib64/libstdc++.so.6 LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:"\
#"/media/p.omenitsch/tools/caffelib/:/media/p.omenitsch/tools/OpenBLAS/lib /media/p.omenitsch/matgpu -r "\
#"\"plotCurvesTraining('${CAFFE_WEIGHTS_DIR}','${MODEL_LABEL}','${PLOTDIRECTORY}'); exit; \""
    #echo $CMD
    #eval $CMD
fi
