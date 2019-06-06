# AI-Avoidance
## iNav avoidance using a neural net

The purpose of this repo is to track progress of a thing that'll nudge a flight controller to avoid obstacles.

## What we have so far
Using one camera, the NN hardware can recognize obstacles. The accuracy is bad, for example my face isn't an obstacle, so I added images of people in the obstacle folder, let's see if it's better.

## Hardware
[Maix Bit](https://github.com/laurentopia/Learning-AI/wiki/Maix-Bit-hardware) which is small, low power consumption, easy to develop on.

![image](https://user-images.githubusercontent.com/26075468/58364077-54bc4f00-7e63-11e9-8b3d-971d954cdd14.jpg)

## Software
It's all python using keras as NN frontend, the software environment was [setup on Windows 10 using Conda](https://github.com/laurentopia/Learning-AI/wiki/Setting-up-Windows-10).

Training script based on [Zipen's mobilenet example](https://bbs.sipeed.com/t/topic/682): train.py

The resulting trained model: avoidance.hl5

Model converted to K210 format using [Maix Toolbox](https://github.com/sipeed/Maix_Toolbox) : avoidance.kmodel

Script that runs on the Maix: boot.py


## The neural net
It's a mobilenet 0.75 224x224 which top has been replaced by a dense pyramid 512 -> 128 -> 32 -> 4
Training at 100 epochs is fast, yield loss of 2% and 100% accuracy
The training set is about 1000 photos scattered amongst 4 folders
![image](https://user-images.githubusercontent.com/26075468/59070342-4280de80-886f-11e9-87b9-c4c7638cd338.png)

NOTE: the photos are probably too similar, they're auto harvested from a video recording of me walking in the forest.
