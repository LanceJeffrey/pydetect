
python3 pyDetect.py \
-m models/coco/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
-l models/coco/coco_labels.txt \
-i rtsp://xxx:xxx@XXX.XXX.X.XXX:554//h264Preview_01_main \
-c inputs/mask.bmp \
-t 0.6 \
-s https://hooks.slack.com/services/XXX \
-w false