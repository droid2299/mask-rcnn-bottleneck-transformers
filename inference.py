import argparse
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from train import create_config

# Create the parser
parser = argparse.ArgumentParser(description='Inference using Mask-RCNN.')

# Add the arguments
parser.add_argument('--input_video', type=str, help='The path to the input video')
parser.add_argument('--output_video', type=str, help='The path to the output video')

# Parse the arguments
args = parser.parse_args()

cfg = create_config('coco')

# Create a predictor
predictor = DefaultPredictor(cfg)

# open file
cap = cv2.VideoCapture(args.input_video)

# get FPS of input video
fps = cap.get(cv2.CAP_PROP_FPS)

# define output video and its FPS
output_file = args.output_video
output_fps = fps

# define VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, output_fps,
                      (int(cap.get(3)), int(cap.get(4))))

i = 0
# read and write frames for output video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    outputs = predictor(frame)

    # Filter instances based on a confidence threshold
    instances = outputs["instances"]
    instances = instances[instances.scores > 0.7]

    # Visualize the results
    v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out_vis = v.draw_instance_predictions(instances.to("cpu"))
    output_frame = out_vis.get_image()[:, :, ::-1]

    # Resize output frame to match input frame's dimensions
    output_frame = cv2.resize(output_frame, (frame.shape[1], frame.shape[0]))

    out.write(output_frame)

    i += 1
    print(f'Inference done on Frame No {i}')

# release resources
cap.release()
out.release()
cv2.destroyAllWindows()
