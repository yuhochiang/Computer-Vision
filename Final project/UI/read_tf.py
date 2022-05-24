import numpy as np
import tensorflow as tf
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from PIL import Image

def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def load_TF_model(model_path):
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		graph_def = tf.compat.v1.GraphDef()

		with tf.io.gfile.GFile(model_path, 'rb') as fid:
			serialized_graph = fid.read()
			graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(graph_def, name='')
	return detection_graph

def detectFaceTF(model_path, label_path, img_path, limit=0.5):
	box = []
	detection_graph = load_TF_model(model_path)
	# load label map
	label_map = label_map_util.load_labelmap(label_path)
	categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1, use_display_name=True)
	category_index = label_map_util.create_category_index(categories)

	with detection_graph.as_default():
		box = []
		# for tf1
		# with tf.Session(graph=detection_graph) as sess:
		with tf.compat.v1.Session(graph=detection_graph) as sess:
			img = Image.open(img_path)
			inp = load_image_into_numpy_array(img)
			# shape: [1, None, None, 3]
			inp_expanded = np.expand_dims(inp, axis=0)
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
			boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
			scores = detection_graph.get_tensor_by_name('detection_scores:0')
			classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')
			
			# detection.
			(boxes, scores, classes, num_detections) = sess.run(
					[boxes, scores, classes, num_detections],
					feed_dict={image_tensor: inp_expanded})

			# Visualization of the results with class and score
			track_ids = np.array([125 for i in classes[0]])
			vis_util.visualize_boxes_and_labels_on_image_array(
					inp, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), 
					np.squeeze(scores), category_index,
					use_normalized_coordinates=True, track_ids=track_ids, 
					skip_track_ids = True, line_thickness=4)
			img = cv2.imread(img_path)
			h, w = img.shape[0], img.shape[1]
			for i in range(int(num_detections[0])):
				score = float(scores[0][i])
				bbox = [float(j) for j in boxes[0][i]]
				if score > limit:
					xmin = bbox[1] * w
					ymin = bbox[0] * h
					xmax = bbox[3] * w
					ymax = bbox[2] * h
					box.append((int(xmin), int(ymin), int(xmax), int(ymax)))
					# text = "{:.2f}%".format(score * 100)
					# y = ymin - 10 if ymin - 10 > 10 else ymin + 10
					# cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), thickness=4)
					# cv2.putText(img, text, (int(xmin), int(y)), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
	
	# for visualizing class and scores return cv2.cvtColor(inp, cv2.COLOR_BGR2RGB), otherwise return img
	return box

if __name__ == '__main__':
	img = detectFaceTF('./rcnn/frozen_inference_graph.pb', './face_label.pbtxt', './images/1.jpg')
	cv2.imshow('img', img)
	cv2.imwrite('./images/1_rcnn.jpg', img)
	cv2.waitKey(0)
