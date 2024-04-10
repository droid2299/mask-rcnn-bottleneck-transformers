import torchvision
from torch import nn
from detectron2.modeling import Backbone, ShapeSpec
from bottlestack import BottleStack


class CustomBackbone(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()

        resnet = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.conv1x1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.bottleneck_layer = BottleStack(
            dim=256,  # Change this from 1024 to 256
            fmap_size=8,
            dim_out=2048,
            proj_factor=4,
            downsample=True,
            heads=4,
            dim_head=128,
            rel_pos_emb=True,
            activation=nn.ReLU()
        )
        self.backbone = nn.Sequential(
            *self.backbone[:-1],
            self.conv1x1,
            nn.AdaptiveAvgPool2d((8, 8)),
            self.bottleneck_layer
        )

    def forward(self, image):
        return self.backbone(image)

    @staticmethod
    def output_shape():
        # Implement the output shape of your custom backbone
        return {"p2": ShapeSpec(channels=256, stride=4), "p3": ShapeSpec(channels=256, stride=8),
                "p4": ShapeSpec(channels=256, stride=16), "p5": ShapeSpec(channels=256, stride=32),
                "p6": ShapeSpec(channels=256, stride=64)}



