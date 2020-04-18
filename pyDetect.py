# import the necessary packages
from imutils.video import VideoStream
from PIL import Image
from PIL import ImageDraw
from playsound import playsound
from skimage.metrics import structural_similarity
import upload
import argparse
import imutils
import time
import numpy as np
import cv2
import detect
import tflite_runtime.interpreter as tflite
import platform
import queue
import threading
import logging

from urllib import request, parse
import json

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

class VideoCapture:
	def __init__(self, name):
		print("[INFO] starting VideoCapture")
		self.cap = VideoStream(name).start()
		self.lock = threading.Lock()
		self.q = queue.Queue()
		t = threading.Thread(target=self._reader)
		t.setDaemon(True)
		t.start()

	def _reader(self):
		while True:
			time.sleep(1/25)
			aframe = self.cap.read()
			self.lock.acquire()
			if not self.q.empty():
				self.q.get_nowait()
			frameDict = dict()
			frameDict["frame"]=aframe
			frameDict["time"]=time.time()
			self.q.put(frameDict)
			self.lock.release()

	def read(self):
		self.lock.acquire()
		theframe = self.q.get()
		self.lock.release()
		return theframe["frame"],theframe["time"]

	def stop(self):
		self.cap.stop()


def main():
	# construct the argument parser and parse the arguments
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-m', '--model', required=True,
					  help='File path of .tflite file.')
	parser.add_argument('-i', '--input', required=True,
					  help='File path of image to process.')
	parser.add_argument('-l', '--labels',
					  help='File path of labels file.')
	parser.add_argument('-t', '--threshold', type=float, default=0.4,
					  help='Score threshold for detected objects.')
	parser.add_argument('-c', '--mask',
					  help='Mask to remove detections from areas')
		
	args = parser.parse_args()
	
	logging.basicConfig(format='%(asctime)s %(module)s.%(funcName)s:%(levelname)s:%(message)s',
					datefmt='%m/%d/%Y %I_%M_%S %p',
					filename="info.log",
					level=logging.INFO)
					
	# initialize the labels dictionary
	print("[INFO] parsing class labels...")
	labels = load_labels(args.labels) if args.labels else {}
		
	# load the Google Coral object detection model
	print("[INFO] loading Coral model...")
	interpreter = make_interpreter(args.model)
	interpreter.allocate_tensors()
	
	# initialize the video stream and allow the camera sensor to warmup
	print("[INFO] starting video stream...")
	cap = VideoCapture(args.input)
	
	mask = cv2.imread(args.mask,0)
	mask = imutils.resize(mask, width=500)

	lastDetection = dict()
	
	time.sleep(1/2)
	lastFrame,times = cap.read()
	lastFrame = imutils.resize(lastFrame, width=500)
	
	print("[INFO] starting detection loop...")
	# loop over the frames from the video stream
	while True:
		
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 500 pixels
		frame,frametime = cap.read()
		frame = imutils.resize(frame, width=500)
				
		image = Image.fromarray(frame)
		
		# Mask areas that we dont want objects to be detected in
		frame = cv2.bitwise_and(frame, frame, mask = mask)
						
		results = processFrame(frame, lastFrame, mask, interpreter, args.threshold)
				
		processResults(image, results, labels, lastDetection, frametime)
		
		lastFrame = frame.copy()
		
		# show the output frame and wait for a key press
		cv2.imshow("Frame", np.asarray(image))
		key = cv2.waitKey(10) & 0xFF
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
		
		#if len(results) == 0:
		time.sleep(1)
			
		
	# do a bit of cleanup
	cap.stop()
	cv2.destroyAllWindows()
	
def processResults(image, results, labels, lastDetection, frametime):
	
	# loop over the results
	for obj in results:
		if obj.id <= 8:
			print('       ' + labels.get(obj.id, obj.id))
			print('         score:  ', obj.score)
			print('         bbox:   ', obj.bbox)
			
			bbox = obj.bbox

			draw = ImageDraw.Draw(image)
			draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
						   outline='red')
			draw.text((bbox.xmin + 10, bbox.ymin + 10),
			  '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
			  fill='red')
	
	for obj in results:
		if obj.id <= 1:	
			if obj.id in lastDetection:
				if lastDetection[obj.id] + 3 <= frametime:
					foundNewDetection(labels.get(obj.id, obj.id), image)
			lastDetection[obj.id] = frametime
	

def foundNewDetection(label, image):
	print ('         +++++++++++++++++++  NEW DETECTION  +++++++++++++++++++  ')
	image.save("latest.png")
	upload.upload("./client_id.json", ["latest.png"], "pyDetect")
	send_message_to_slack("Detected new " + label)
	playsound('inputs/chord.wav')

def send_message_to_slack(text):
 
    post = {"text": "{0}".format(text)}
 
    try:
        json_data = json.dumps(post)
        req = request.Request("https://hooks.slack.com/services/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",
                              data=json_data.encode('ascii'),
                              headers={'Content-Type': 'application/json'}) 
        resp = request.urlopen(req)
    except Exception as em:
        print("EXCEPTION: " + str(em))

def processFrame(frame, lastFrame, mask, interpreter, threshold):

	start = time.time()
	
	print("[INFO] =============================")
	print("[INFO] Processing Frame")
	
	results = []
				
	# prepare the frame for object detection by converting (1) it
	# from BGR to RGB channel ordering and then (2) from a NumPy
	# array to PIL image format
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = Image.fromarray(frame)
	
	# make predictions on the input frame
	scale = detect.set_input(interpreter, frame.size,
						   lambda size: frame.resize(size, Image.ANTIALIAS))
	interpreter.invoke()
	results = detect.get_output(interpreter, threshold, scale)
	
	end = time.time()
	print('[INFO] Proccessing Time: %.3f s' % (end - start))

	return results

def load_labels(path, encoding='utf-8'):
  """Loads labels from file (with or without index numbers).

  Args:
    path: path to label file.
    encoding: label file encoding.
  Returns:
    Dictionary mapping indices to labels.
  """
  with open(path, 'r', encoding=encoding) as f:
    lines = f.readlines()
    if not lines:
      return {}

    if lines[0].split(' ', maxsplit=1)[0].isdigit():
      pairs = [line.split(' ', maxsplit=1) for line in lines]
      return {int(index): label.strip() for index, label in pairs}
    else:
      return {index: line.strip() for index, line in enumerate(lines)}


def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])

if __name__ == '__main__':
	print("[INFO] starting")
	main()



