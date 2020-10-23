#!/bin/bash

while true
do
	python3 pyDetect.py -m models/coco/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
		-s ${PYDETECT_SLACK_URL} \
		-w true \
		-t 0.60 \
		-c inputs/mask.bmp \
		-i ${PYDETECT_CAM_RSTP} \
		-l models/coco/coco_labels.txt

	sleep 5m
done

