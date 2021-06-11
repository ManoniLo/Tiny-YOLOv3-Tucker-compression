#!/bin/bash
sed -i "s/from tensorflow.keras.applications.resnet import ResNet50/from tensorflow.keras.applications.resnet50 import ResNet50/" yolo3/models/yolo3_resnet50.py
sed -i "s/from tensorflow.keras.applications.resnet import ResNet50/from tensorflow.keras.applications.resnet50 import ResNet50/" yolo4/models/yolo4_resnet50.py

parallel {} ::: << EOF
wget --quiet -nc -O weights/darknet53.conv.74.weights https://pjreddie.com/media/files/darknet53.conv.74
wget --quiet -nc -O weights/darknet19_448.conv.23.weights https://pjreddie.com/media/files/darknet19_448.conv.23
wget --quiet -nc -O weights/yolov3.weights https://pjreddie.com/media/files/yolov3.weights
wget --quiet -nc -O weights/yolov3-tiny.weights https://pjreddie.com/media/files/yolov3-tiny.weights
wget --quiet -nc -O weights/yolov3-spp.weights https://pjreddie.com/media/files/yolov3-spp.weights
wget --quiet -nc -O weights/yolov2.weights http://pjreddie.com/media/files/yolo.weights
wget --quiet -nc -O weights/yolov2-voc.weights http://pjreddie.com/media/files/yolo-voc.weights
wget --quiet -nc -O weights/yolov2-tiny.weights https://pjreddie.com/media/files/yolov2-tiny.weights
wget --quiet -nc -O weights/yolov2-tiny-voc.weights https://pjreddie.com/media/files/yolov2-tiny-voc.weights
wget --quiet -nc -O weights/yolov4.weights https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
gdown --id 18jCwaL4SJ-jOvXrZNGHJ5yz44g9zi8Hm && mv csdarknet53-omega_final.weights weights/
wget -nc -O weights/yolo-fastest.weights https://github.com/dog-qiuqiu/Yolo-Fastest/blob/master/ModelZoo/yolo-fastest-1.1_coco/yolo-fastest-1.1.weights?raw=true
wget -nc -O weights/yolo-fastest-xl.weights https://github.com/dog-qiuqiu/Yolo-Fastest/blob/master/ModelZoo/yolo-fastest-1.1_coco/yolo-fastest-1.1-xl.weights?raw=true
gdown --id 1NQwz47cW0NUgy7L3_xOKaNEfLoQuq3EL && mv yolov4-csp.weights weights/
EOF
