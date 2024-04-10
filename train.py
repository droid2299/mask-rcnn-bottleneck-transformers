import logging
import numpy as np
import os
import detectron2.utils.comm as comm
import torch
import utils
from collections import OrderedDict
from backbone import CustomBackbone
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.utils.events import EventStorage
from detectron2.modeling import build_model
from detectron2.engine import default_writers
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)


def get_evaluator(cfg, dataset_name, output_folder=None):
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model, logger):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def create_config(dataset_name, output_dir):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (f"{dataset_name}_train",)
    cfg.DATASETS.TEST = (f"{dataset_name}_val",)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 81  # number of classes in your dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 100000  # We found that with patience of 500, training will early stop
    # before 10,000 iterations
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 81  # 26 letters plus one superclass
    cfg.TEST.EVAL_PERIOD = 0  # Increase this number if you want to monitor validation performance during training
    # Data Augmentations
    cfg.INPUT.RANDOM_FLIP = "horizontal"  # Horizontal and vertical flips
    cfg.INPUT.CROP.ENABLED = True  # Random cropping
    cfg.INPUT.RANDOM_ROTATION = 15  # Rotate the image by up to 15 degrees
    cfg.INPUT.RANDOM_SCALE = (0.8, 1.2)  # Scale the image by a random factor between 0.8 and 1.2
    cfg.INPUT.COLOR_JITTER = 0.2  # Randomly adjust brightness, contrast, saturation, and hue by up to 20%
    cfg.INPUT.GAUSSIAN_BLUR = 0.1  # Apply Gaussian blur with a probability of 10%
    cfg.INPUT.GAUSSIAN_NOISE = 0.1  # Add Gaussian noise with a probability of 10%
    cfg.INPUT.AFFINE_TRANSFORM = True  # Apply random affine transformations
    cfg.INPUT.CUTOUT = True  # Apply cutout augmentation
    cfg.INPUT.RANDOM_ERASING = True  # Apply random erasing augmentation

    cfg.OUTPUT_DIR = output_dir
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000

    # Replace the backbone
    cfg.MODEL.BACKBONE.NAME = "CustomBackbone"

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    with open(f"{cfg.OUTPUT_DIR}/config.yaml", "w") as f:
        f.write(cfg.dump())

    return cfg


def main(args):
    # Initialize logging
    setup_logger()
    logger = logging.getLogger("detectron2")

    PATIENCE = args.patience

    # Register custom backbone and datasets
    utils.register_custom_backbone(CustomBackbone)
    utils.register_coco_datasets(args.dataset_name, args.train_image_dir, args.train_annotation_file,
                                 args.val_image_dir, args.val_annotation_file)

    # Create configuration
    cfg = create_config(args.dataset_name, args.output_dir)

    # Build model, optimizer, and scheduler
    model = build_model(cfg)
    logger.info(f"Model Architecture: \n {model}")
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    BEST_LOSS = np.inf

    # Initialize checkpointers
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume).get("iteration", -1) + 1
    )
    prev_iter = start_iter
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    patience_counter = 0
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                do_test(cfg, model, logger)
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

            # Early stopping condition
            if losses_reduced < BEST_LOSS:
                BEST_LOSS = losses_reduced
                patience_counter = 0
                print(f'Saving the best model, with loss {BEST_LOSS}, at iteration {iteration}')
                BEST_MODEL_PATH = os.path.join(cfg.OUTPUT_DIR, f"best_model_{iteration}.pth")
                checkpointer.save(f"best_model_{iteration}")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    logger.info("EARLY STOPPING due to no improvement in total_loss")
                    break

    do_test(cfg, model, logger)


if __name__ == "__main__":
    args = utils.parse_args()
    main(args)
