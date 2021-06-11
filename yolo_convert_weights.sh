#!/bin/bash
python tools/model_converter/convert.py cfg/yolov3.cfg weights/yolov3.weights weights/yolov3.h5 && rm weights/yolov3.weights
python tools/model_converter/convert.py cfg/yolov3-tiny.cfg weights/yolov3-tiny.weights weights/yolov3-tiny.h5 && rm weights/yolov3-tiny.weights
python tools/model_converter/convert.py cfg/yolov3-spp.cfg weights/yolov3-spp.weights weights/yolov3-spp.h5 && rm weights/yolov3-spp.weights
python tools/model_converter/convert.py cfg/yolov2.cfg weights/yolov2.weights weights/yolov2.h5 && rm weights/yolov2.weights
python tools/model_converter/convert.py cfg/yolov2-voc.cfg weights/yolov2-voc.weights weights/yolov2-voc.h5 && rm weights/yolov2-voc.weights
python tools/model_converter/convert.py cfg/yolov2-tiny.cfg weights/yolov2-tiny.weights weights/yolov2-tiny.h5 && rm weights/yolov2-tiny.weights
python tools/model_converter/convert.py cfg/yolov2-tiny-voc.cfg weights/yolov2-tiny-voc.weights weights/yolov2-tiny-voc.h5 && rm weights/yolov2-tiny-voc.weights
python tools/model_converter/convert.py cfg/darknet53.cfg weights/darknet53.conv.74.weights weights/darknet53.h5 && rm weights/darknet53.conv.74.weights
python tools/model_converter/convert.py cfg/darknet19_448_body.cfg weights/darknet19_448.conv.23.weights weights/darknet19.h5 && rm weights/darknet19_448.conv.23.weights
python tools/model_converter/convert.py cfg/csdarknet53-omega.cfg weights/csdarknet53-omega_final.weights weights/cspdarknet53.h5 && rm weights/csdarknet53-omega_final.weights
python tools/model_converter/convert.py --yolo4_reorder cfg/yolov4.cfg weights/yolov4.weights weights/yolov4.h5 && rm weights/yolov4.weights
python tools/model_converter/convert.py --yolo4_reorder cfg/yolov4-csp_fixed.cfg weights/yolov4-csp.weights weights/scaled-yolov4-csp.h5 && rm weights/yolov4-csp.weights
python tools/model_converter/convert.py cfg/yolo-fastest.cfg weights/yolo-fastest.weights weights/yolo-fastest.h5 && rm weights/yolo-fastest.weights
python tools/model_converter/convert.py cfg/yolo-fastest-xl.cfg weights/yolo-fastest-xl.weights weights/yolo-fastest-xl.h5 && rm weights/yolo-fastest-xl.weights
