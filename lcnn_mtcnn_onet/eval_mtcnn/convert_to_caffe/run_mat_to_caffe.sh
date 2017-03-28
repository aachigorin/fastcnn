CMD="LD_PRELOAD=/usr/lib64/libstdc++.so.6 LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:/media/p.omenitsch/tools/caffelib/:/media/p.omenitsch/tools/OpenBLAS/lib /media/p.omenitsch/matgpu -r \"mat_to_caffe('/media/a.chigorin/code/fastcnn/lcnn_mtcnn_onet/convert_to_caffe/weights/', 'onet'); exit; \""
echo $CMD
eval $CMD
