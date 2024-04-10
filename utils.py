import argparse
from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.data.datasets import register_coco_instances


def register_custom_backbone(backbone):
    BACKBONE_REGISTRY.register(backbone)
    print("Registered Custom Backbone:")


def register_coco_datasets(dataset_name, train_img_path, train_ann_path, val_img_path, val_ann_path):
    register_coco_instances(f'{dataset_name}_train', {}, train_ann_path, train_img_path)
    register_coco_instances(f'{dataset_name}_val', {}, val_ann_path, val_img_path)
    print(f"Registered COCO Datasets as {dataset_name}_train,{dataset_name}_val")


def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description='Arguments for Mask R-CNN Training and Inference')

    # Add the arguments
    parser.add_argument('--input_video', type=str, help='The path to the input video')
    parser.add_argument('--output_video', type=str, help='The path to the output video')
    parser.add_argument('--threshold', type=float, default=0.7, help='The confidence threshold for instance filtering')
    parser.add_argument("--train-image-dir", required=True, help="Path to the training images directory")
    parser.add_argument("--train-annotation-file", required=True, help="Path to the training annotation file")
    parser.add_argument("--val-image-dir", required=True, help="Path to the validation images directory")
    parser.add_argument("--val-annotation-file", required=True, help="Path to the validation annotation file")
    parser.add_argument("--resume", action="store_true", help="Whether to resume training from the latest checkpoint")
    parser.add_argument("--patience", type=int, default=5000, help="Number of iterations with no improvement to wait "
                                                                   "before early stopping")
    parser.add_argument("--dataset-name", default='coco', help="Name of the dataset")
    parser.add_argument("--output-dir", default='/content/drive/MyDrive/EECS6322/training_results_100000_iterations/',
                        help="Output directory to save weights")

    # Parse the arguments
    args = parser.parse_args()
    return args
