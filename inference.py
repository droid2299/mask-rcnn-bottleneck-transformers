import utils
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from train import create_config


def process_video(args):
    # Parse the arguments
    cfg = create_config(args.dataset_name, args.output_dir)

    # Create a predictor
    predictor = DefaultPredictor(cfg)

    # Open file
    cap = cv2.VideoCapture(args.input_video)

    # Get FPS of input video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define output video and its FPS
    output_file = args.output_video
    output_fps = fps

    # Define VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, output_fps,
                          (int(cap.get(3)), int(cap.get(4))))

    i = 0
    # Read and write frames for output video
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

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = utils.parse_args()
    process_video(args)
