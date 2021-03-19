# rpiCarAutopilot - Self driving car with raspberry pi and neural networks!
This project is for CS129 and CS230. I intend to use the Sunfounder mini car kit for Raspberry Pi, a GPS/IMU, and a camera in order to create a self-driving car that can perform point-to point navigation. This car will use a YOLOv3 object detection algorithm, Spatial-CNN algorithm, and a hybrid A* algorithm to perform automatic object detection, lane detection,  and path finding. 

## Bill of Materials
- Raspberry Pi 3b
- Sunfounder Raspberry Pi mini car kit (power supply, motors, controllers, and the car)
- Ozzmaker Berry GPS/IMU v4
- Google Coral USB TPU (possible contigent upon suppliers)
![alt text](https://github.com/joe-loy/rpiCarAutopilot/blob/main/project_bom_image.png)
## Setup
In order to train the Berkeley Deep Drive dataset, one should clone the darknet YOLOv3 source code from pjreddie and follow the process done [here](https://github.com/yogeshgajjar/BDD100k-YOLOV3-tiny) to organize and prepare the dataset for training. 

## Dataset
This project uses the Berkeley Deep Driving dataset in order to train the object detection system of the car. 

## Software Setup
The software controlling the vehicle will be decomposed into the following subcomponents
- objectDetector.py
- laneNavigator.py
- pathFinder.py
- movementTracker.py
- motorController.py

### Object Detection
For object detection, the car will use a YOLOv3 architectire. This architecture was chosen since it gives a good balance of speed and effectiveness. Although other architectures such as Deep Residual Learning and Fast R-CNN have higher accuracy, these often run slower. Since we will be running this system on a Raspberry Pi, we will want to use an architecture that is accurate but still is not too computationally intensive, since we need the system to perform well in real time. Pictured below this text is the architecture for the YOLOv3 object detection algorithm. Current implementation is courtesy of https://github.com/qqwweee/keras-yolo3/, but I will modify the algorithm and hyperparameters accordingly for the raspberry pi car. <br>
![alt text](https://github.com/joe-loy/rpiCarAutopilot/blob/main/YOLOv3.png)

### Lane Detection
For lane detection, we will use a SpatialCNN architecture. This architecture was chosen since it is very high performing, and it is also a little bit older than some of the very new architectures that perform a little bit better. Since I already have a few different neural networks in this project, I want to make sure that I am using architectures that are easy to troubleshoot, perform well, and have a variety of references. Current implementation is courtesy of https://github.com/cardwing/Codes-for-Lane-Detection , but I will modify the algorithm and hyperparameters accordingly for the raspberry pi car. <br>
![alt text](https://github.com/joe-loy/rpiCarAutopilot/blob/main/Spatial-CNN.png)

### Pathfinding
For pathfinding, we will use a hybrid A* algorithm. This algorithm is chosen for many of the same reasons that the YOLOv3 was chosen for object detection, namely that the algorithm provides a great tradeoff between accuracy and speed. While some people may opt to use reinforcement learning to train a pathfinding algorithm, I have decided to limit the complexity of this project by using a more conventional algorithm for path finding rather than trying to train another neural network.  <br> ![alt text](https://github.com/joe-loy/rpiCarAutopilot/blob/main/a-star.png)

### Movement Tracking
The location of the car, the speed of the car, and the direction of the car will be tracked using a Ozzmaker Berry GPS-IMU v4 chip and the Navit open source navigation system. I plan on using a Kalmann Filter with the chip in order to further refine the estimation of location and movement. <br>
![alt text](https://github.com/joe-loy/rpiCarAutopilot/blob/main/kalman_filter.jpeg)

### Motion Control
Using location+movement info, object detection info, and a path suggestion, the engine and steering must be controlled in a way that causes the car to move towards the suggested path. 

# Misc
Camera resolution 640x480

# References 
BerryGPS-IMU gps setup raspberry pi -- https://ozzmaker.com/berrygps-setup-guide-raspberry-pi/ <br>
BerryGPS-IMU gps datalogger setup -- https://ozzmaker.com/gps-data-logger-using-berrygps/ <br>
BerryGPS-IMU imu setup raspberry pi -- https://ozzmaker.com/berryimu/ <br>
BerryGPS-IMU kalmann filter setup -- https://ozzmaker.com/guide-interfacing-gyro-accelerometer-raspberry-pi-kalman-filter/
Bojarski, M., Del Testa, D., Dworakowski, D., Firner, B., Flepp, B., Goyal, P., ... & Zieba, K. (2016). End to end learning for self-driving cars. arXiv preprint arXiv:1604.07316. <br>
Deac, M. A., Al-doori, R. W. Y., Negru, M., & Dǎnescu, R. (2018, September). Miniature autonomous vehicle development on raspberry pi. In 2018 IEEE 14th International Conference on Intelligent Computer Communication and Processing (ICCP) (pp. 229-236). IEEE. <br>
DeepPiCar blogpost -- https://towardsdatascience.com/deeppicar-part-1-102e03c83f2c <br>
DeepPiCar source code -- https://github.com/dctian/DeepPiCar <br>
Lane Detection Blogpost -- https://towardsdatascience.com/tutorial-build-a-lane-detector-679fd8953132 <br> 
Pan, X., Shi, J., Luo, P., Wang, X., & Tang, X. (2018, April). Spatial as deep: Spatial cnn for traffic scene understanding. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 32, No. 1). <br>
Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 779-788). <br>
SpatialCNN open source implementation --  https://github.com/cardwing/Codes-for-Lane-Detection <br>
YOLOv3 open source implementation -- https://github.com/qqwweee/keras-yolo3/ <br>
YOLOv3 website -- https://pjreddie.com/darknet/yolo/ <br>
