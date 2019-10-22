# Head and Shoulder Detection Base on MTCNN

use coco dataset to detect head and shoulder. This implements is base on MTCNN
Pretrain model has been placed in models. 

## prepare
- download coco keypoints dataset 
- python preprocess/coco.py --data-dir {your coco dataset } --anotation {anotation} -o {coco.feather}  #  collect keypoints and gen boundbox
- python preprocess/image_process.py -n {pnet,rnet,onet} --preprocess-path  {./data/coco.feather}  # gen data for one stage

## train
  train pnet
   >> python nets/net.py -n pnet -lr 0.002 -w 2
  train rnet
   >> python nets/net.py -n rnet -lr 0.002 -w 2
  train onet
   >> python nets/net.py -n onet -lr 0.002 -w 2

## hard mining
   python preprocess/hard_mining.py -n rnet
   python preprocess/hard_mining.py -n onet

## test 
   >> python nets/test.py -p video 

## Warning
   - Prediction is much more slower than expected in keras, but when predicts it on arm-rk3399, it only cost about 100ms totally.(python is really slow)
   - how to import the performace
       - batch norm
       - change the prediction of bound box, taking consideration of yolo v2/3.
       - cleaning data (it`s really important!!!!!. Our generator scripts exist a lot of noisy) 
       - pruning model

current result

| |pnet|rnet|onet|
|-|-|-|-|
| |94%|96.1%|98.5%|
