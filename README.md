# AI-Avoidance
## iNav avoidance using a neural net

The purpose of this repo is to track progress of a thing that'll nudge a flight controller to avoid obstacles.

## What we have so far
Using one camera, the NN hardware can recognize obstacles with a pretty bad accuracy. It doesn't recognize people so I added faces and bodies in the training set.

## Hardware
[Maix Bit](https://github.com/laurentopia/Learning-AI/wiki/Maix-Bit-hardware)

![image](https://user-images.githubusercontent.com/26075468/58364077-54bc4f00-7e63-11e9-8b3d-971d954cdd14.jpg)

## Software
It's all python
Training script based on [Zipen's mobilenet example](https://bbs.sipeed.com/t/topic/682): train.py
The resulting trained network: avoidance.hl5
Script that runs on the Maix: boot.py


## The neural net
It's a mobilenet 0.75 224x224 which top has been replaced by a dense pyramid 512 -> 128 -> 32 -> 4
Training at 100 epochs is fast, yield loss of 2% and 100% accuracy
The training set is about 1000 photos scattered amongst 4 folders
![image](https://user-images.githubusercontent.com/26075468/59070342-4280de80-886f-11e9-87b9-c4c7638cd338.png)

NOTE: the photos are probably too similar, they're auto harvested from a video recording of me walking in the forest.
