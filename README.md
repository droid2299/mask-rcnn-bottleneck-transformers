# Mask-RCNN using Bottleneck Transformers

This is our attempt to reproduce the metrics of image segmentation given in the paper "Bottleneck Transformers for Visual Recognition"
by Srinivas et al.


To train the model on the COCO dataset, run

```bash
python train.py 
```

To run inference on a video, run

```bash
python inference.py --input_video /path/to/input/video --output_video /path/to/output/video
```


TODO:
1. Make code more modular

## Citations

```bibtex
@misc{srinivas2021bottleneck,
    title   = {Bottleneck Transformers for Visual Recognition}, 
    author  = {Aravind Srinivas and Tsung-Yi Lin and Niki Parmar and Jonathon Shlens and Pieter Abbeel and Ashish Vaswani},
    year    = {2021},
    eprint  = {2101.11605},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```
