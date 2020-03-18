# Team #16: 机智的自动驾驶

## Team Members
* **赵立焱**
* 罗晨 
* 易子诺 
* 毛志伟 
* 张子豪 


## Resources
* **[DriveCore Studio](https://vcsdrivecoreprod.blob.core.windows.net/drivecore-studio/latest/dc-studio/dcstudio.tar.gz?sp=r&st=2019-07-10T00:00:00Z&se=2019-12-31T23:59:59Z&spr=https&sv=2018-03-28&sig=SUEow8dL2HnwcxK7vVR5V3ByasA62%2B50tVTAWYxkUBE%3D&sr=b)**
* **[DriveCore Studio Manual](https://vcsdrivecoreprod.blob.core.windows.net/drivecore-studio/latest/dc-studio/DriveCoreStudioManual.pdf?sp=r&st=2019-07-10T00:00:00Z&se=2019-12-31T23:59:59Z&spr=https&sv=2018-03-28&sig=a4NVuFNRi5kCFkwBo83X51somuiNSlSHVRW1ZD%2BD5jk%3D&sr=b)**
* **[DriveCore - Algorithm Evaluation Framework](https://vcshackathontongji.azureedge.net/DriveCore%20-%20%20Algorithm%20Evaluation%20Framework.pdf)**
* **[DriveCore - Camera Object Detection Evaluation - Quick Start Guide](https://vcshackathontongji.azureedge.net/DriveCore%20-%20Camera%20Obj.%20Det.%20Evaluation%20-%20Quick%20Start%20Guide.pdf)**
* **[V-Hackathon 2019 - Problem Statement](https://vcshackathontongji.azureedge.net/V-Hackathon%202019%20-%20Problem%20Statement.pdf)**
* **[KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/raw_data.php)**

## How to Commit Your Code

The Hackathon evaluation builds, runs and evaluates your contributions.
In order to do so, you are required to fill the `build.sh` script located in the root of your
team repository. The build environment that is going to be used for your project is 64-bit Ubuntu Linux.
You are responsible for setting up any external dependencies your code may need for building and
running. You can use `build.sh` script to set up your environment.

The steps you have defined in `build.sh` must produce an artifact located in a `./build/` directory
within the root of your project. The evaluation process will use those artifacts
in order to run them against predefined tests for measuring key performance indicators of your
submission.

## Help & Support
* **[Wiki](https://github.com/DriveCoreStudio/TongjiHackathon/wiki)**
* **[Issues](https://github.com/DriveCoreStudio/TongjiHackathon/issues)**
* **[Gitter Chat](https://gitter.im/DriveCoreStudio/TongjiHackathon)**
* Email: **dcstudio@visteon.com**

## Enviriment configuration
master branch is a GPU model based on MxNet1.5.0 python api and gluoncv0.5.0  
At runtime you should confirm that one of cuda8.0/9.0/9.2/10 and cudnn have been installed.  
run build.sh it will install mxnet-cu80/90/92/100/101mkl==1.5.0(Corresponding to your cuda version) and gluoncv==0.5.0  

cpu_master branch is a CPU model based on MxNet1.5.0 python api and gluoncv0.5.0  
It's the same model as GPU model, the only difference is that using mxnet's mkldnn operator for speed up CPU inference without CUDA and cudnn(You don't need to install mkldnn separately, just install mxnet-mkl)
run build.sh it will install mxnet-mkl==1.5.0 and gluoncv==0.5.0(if you have GPU and have run master branch's build.sh, you don't need to run cpu_master's build.sh. mxnet-cuxxmkl version is good for GPU and CPU inference)

## Model overview
gpu model get 2.9 score on our computer  
cpu model get 1.3 score on our computer
### model select
The scoring mechanism of dcstudio places great emphasis on the speed of the model compared to the accuracy of the model. Although we successfully trained a more accurate model based on Mobilenet1.0_yolo3 or Mobilenet1.0_SSD512, it performed very well at 416 x 416 or larger image size, and its speed exceeded the baseline CPU mode. However, we can't get its speed to reach the baseline GPU mode to get a higher total score.
In the end, we chose Mobilenet0.25_yolo3 as our final model. Compared with the previous model, its backbone is processed by the pruning algorithm, the parameter size is only 0.25 times the original mobilenet and faster, but the expression ability is also limited, the accuracy is lower than Mobilenet1.0_yolo3.It runs on a 320 x 320 image for optimal speed performance.
### vehicle classes
truck, car, van, bus
### datasets
all = bdd100kval(car,bus,truck:9000+)+KITTI(Car,Van,Truck:7000+)+VOC(car,bus,truck:3000+) shuffle 85% for train 15% for validation, finally get MAP 0.59 (IOU 50%) on validation dataset  

2019-9-28: finalmodel: all = bdd100ktrain(car,bus,truck:70000)+KITTI(Car,Van,Truck:5000+)+VOC(car,bus,truck:2000+) for train, bdd100kval(car,bus,truck:9000+)+KITTI(Car,Van,Truck:1000+)+VOC(car,bus,truck:1000+) for validation finally get MAP 0.59 (IOU 50%) on validation dataset(get 63 MAP on validation before) 
### training tricks
* 1. Mobilenet0.25_yolo3 does not have an open source COCO pre-training model to transfer. We can only use the ImageNet pre-trained Mobilenet 0.25 backbone to transfer.
* 2. Acording to the paper [Bag of Freebies for Training Object Detection Neural Networks](https://arxiv.org/pdf/1902.04103.pdf) we use tricks "Label Smoothing","warm-up learning rate","Random shapes training","0 wd for bias" in our model. From the performance of the test set, "Label Smoothing" and "Random shapes training" can significantly reduce the over-fitting and improve the generalization ability of the model.
* 3. data augmentation: use gluoncv YOLO3DefaultTrainTransform ,include "random expansion with prob 0.5", "random cropping", "resize with random interpolation", "random color jittering", "random horizontal flip"
* 4. In order to improve the performance of the model at night and small objects, we used the bdd100k dataset (KITTI does not contain nighttime images, bdd100k has about one-third of nighttime images, one-third of daytime images, one-third of early mornings and dusk images, as well as numerous small objects). Compared to VOC and KITTI, using COCO can also improve performance at night and on small objects, but not as good as bdd100k.
* 5. Our improvement: The original yolo3 algorithm uses "Random shapes training", which means 10 random size images from 320 to 608 intervals 32 during training.This can improve the performance of the model on different scales of the original image.However, after such training, considering the speed and accuracy, an image with a resize to 416 x 416 size is usually used as the standard input. Our target model is to use resize to 320 X 320 size images. Our experiments show that modifying "320 to 608 intervals 32" to "256 to 416 intervals 32" can get a better MAP on the 320 X 320 scale.

