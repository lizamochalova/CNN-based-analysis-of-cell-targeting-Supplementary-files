# Сonnection of necessary libraries
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from os import walk
import collections
import time


sys.path.append("..")


from utils import label_map_util
from utils import visualization_utils as vis_util

# Description of folders

# Name of general folder with all data (see file structure in methods)
PROJECT_FOLDER = 'frcnn_clean_partnum'

# Name of folder with images
DATA_FOLD = 'data'

# Name of the folder in which the result is saved
RES_FOLD = 'result'

# Full path to object_detection folder
CWD_PATH = os.getcwd()

# Full path to file frozen_inference_graph.pb
PATH_TO_CKPT = os.path.join(CWD_PATH,PROJECT_FOLDER,'frozen_inference_graph.pb')

# Full path to file labelmap.pbtxt
PATH_TO_LABELS = os.path.join(CWD_PATH,PROJECT_FOLDER,'labelmap.pbtxt')

# Number of classes
NUM_CLASSES = 3


# Function from a picture that returns a two-dimensional list consisting of objects and their probability metrics 
# For example: [[cell, 99], [particle, 98], [particle, 82]]
def results_of_detection(
    image,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    agnostic_mode=False,
    line_thickness=4,
    groundtruth_box_visualization_color='black',
    skip_scores=False,
    skip_labels=False):
  """Overlay labeled boxes on an image with formatted scores and label names.

  This function groups boxes that correspond to the same location
  and creates a display string for each detection and overlays these
  on the image. Note that this function modifies the image in place, and returns
  that same image.

  Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    instance_masks: a numpy array of shape [N, image_height, image_width] with
      values ranging between 0 and 1, can be None.
    instance_boundaries: a numpy array of shape [N, image_height, image_width]
      with values ranging between 0 and 1, can be None.
    keypoints: a numpy array of shape [N, num_keypoints, 2], can
      be None
    use_normalized_coordinates: whether boxes is to be interpreted as
      normalized coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
      all boxes.
    min_score_thresh: minimum score threshold for a box to be visualized
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.  This mode will display scores but ignore
      classes.
    line_thickness: integer (default: 4) controlling line width of the boxes.
    groundtruth_box_visualization_color: box color for visualizing groundtruth
      boxes
    skip_scores: whether to skip score when drawing a single detection
    skip_labels: whether to skip label when drawing a single detection

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
  """
  res = list()
  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  box_to_instance_masks_map = {}
  box_to_instance_boundaries_map = {}
  box_to_keypoints_map = collections.defaultdict(list)
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      if instance_masks is not None:
        box_to_instance_masks_map[box] = instance_masks[i]
      if instance_boundaries is not None:
        box_to_instance_boundaries_map[box] = instance_boundaries[i]
      if keypoints is not None:
        box_to_keypoints_map[box].extend(keypoints[i])
      if scores is None:
        box_to_color_map[box] = groundtruth_box_visualization_color
      else:
        display_str = ''
        if not skip_labels:
          if not agnostic_mode:
            if classes[i] in category_index.keys():
              class_name = category_index[classes[i]]['name']
            else:
              class_name = 'N/A'
            display_str = str(class_name)
        if not skip_scores:
          if not display_str:
            display_str = '{}%'.format(int(100*scores[i]))
          else:
            display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
            res.append([class_name, int(100*scores[i])])
  return res
  
  
# Loading of labelmap 
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Model loading
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

#  Function that returns a list of folders located in the specified folder
def folds_inside(path):
    folders = []
    for (dirpath, dirnames, filenames) in walk(path):
        folders.extend(dirnames)
        break
    return folders

#  Function that returns a list of files located in the specified folder
def files_inside(path):
    files = []
    for (dirpath, dirnames, filenames) in walk(path):
        files.extend(filenames)
        break
    return files



# Definition of input and output tensors
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Getting a list of folders in the data folder (each individual dimension in a separate folder)
folders = folds_inside(os.path.join(CWD_PATH,PROJECT_FOLDER,DATA_FOLD))

# The following is performed sequentially for all folders found in the "data" folder, let's call this "folder" 
for folder in folders:
    # Start the timer, notes the time of the program
    start_time = time.time()
    # Creates a text file to write the result to
    textFile = open((os.path.join(CWD_PATH,PROJECT_FOLDER,RES_FOLD,folder) + '.txt'), 'w')
    # Getting a list of files in the "folder" folder
    files = files_inside(os.path.join(CWD_PATH,PROJECT_FOLDER,DATA_FOLD,folder))
    # The following is done for all files in the "folder" folder, let's call this file "file"
    for file in files:
        # Reading "file"
        frame = cv2.imread(os.path.join(CWD_PATH,PROJECT_FOLDER,DATA_FOLD,folder,file))
        # Getting the tensor of the desired dimension
        frame_expanded = np.expand_dims(frame, axis=0)
        # Direct detection
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
        # Counting objects found
        arr = results_of_detection(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)
        # Counters for cells and particles
        cel = 0
        par = 0
        # The following is done for all objects in the image, let's call this object "obj"
        for obj in arr:
            # If the object is a cell and the probability metric> = 80, then we increase the counter for cells by 1, the metrics threshold is custimizable
            if obj[0] == 'Cell':
                if int(obj[1]) >= 80: 
                    cel += 1
            # If the object is a particle and the probability metric> = 80, then we increase the counter for particles by 1, the metric threshold is is custimizable
            elif obj[0] == 'Particle':
                if int(obj[1]) >= 80:
                    par += 1
        # If exactly one cell is found in the picture, then we add a line of the form: file name, number of particles. Write a string to a text file and display it
        if cel == 1:
            textFile.write(file + '\t' + str(par) + '\n')
        print(file + '\t' + str(par))
    # After writing all the lines close the text file
    textFile.close()
    # Displaying elapsed time
    print("--- %s seconds ---" % (time.time() - start_time))

