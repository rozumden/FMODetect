# FMODetect Evaluation and Training Code 

### FMODetect: Robust Detection and Trajectory Estimation of Fast Moving Objects
#### Denys Rozumnyi, Martin R. Oswald, Vittorio Ferrari, Jiri Matas, Marc Pollefeys

### Inference
To detect fast moving objects in video:
```bash
python run.py --video example/falling_pen.avi
```

To detect fast moving objects in a single frame with the given background:
```bash
python run.py --im example/ex1_im.png --bgr example/ex1_bgr.png
```


### Pre-trained models

The pre-trained FMODetect model as reported in the paper is available here: https://polybox.ethz.ch/index.php/s/X3J41G9DFuwQOeY.

Reference
------------
If you use this repository, please cite the following publication ( https://arxiv.org/abs/2012.08216 ):

```bibtex
@inproceedings{fmodetect,
  author = {Denys Rozumnyi and Jiri Matas and Filip Sroubek and Marc Pollefeys and Martin R. Oswald},
  title = {FMODetect: Robust Detection and Trajectory Estimation of Fast Moving Objects},
  booktitle = {arxiv},
  address = {online},
  month = dec,
  year = {2020}
}
```
