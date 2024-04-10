from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.data.datasets import register_coco_instances


def register_custom_backbone(backbone):
    BACKBONE_REGISTRY.register(backbone)
    print("Registered Custom Backbone:")


def register_coco_datasets(dataset_name, train_img_path, train_ann_path, val_img_path, val_ann_path):
    register_coco_instances(f'{dataset_name}_train', {}, train_ann_path, train_img_path)
    register_coco_instances(f'{dataset_name}_val', {}, val_ann_path, val_img_path)
    print(f"Registered COCO Datasets as {dataset_name}_train,{dataset_name}_val")
