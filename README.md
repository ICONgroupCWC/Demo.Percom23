
# Codesign of Edge Intelligence and Automated Guided Vehicle Control

This is a demonstration of a semi-autonomous transportation task aided by edge intelligence. 
- An automated guided vehicle (AGV) equipped with a robotic arm needs to 
  - Pick an object from a source point,
  - Follow a path defined by black stripes, and 
  - Drop the object at a specific drop-point in one of four destinations implicitly defined by a human operator.

- Human operator defines the destination and exact drop point (delivery information) by placing an irregular black shape inside a destination.
  - Irregular shape is referred to as custom drop area.
  - The exact drop-point is the center of the custom drop area, which is defined as the center of the largest circle that is placed inside the custom drop area. 

- At the source point, AGV request the delivery information from an edge server. Then, edge server uses a remote camera to obtain the bird's eye view of all destinations. The intelligence at the edge server
  - Extracts the delivery information from the camera image and
  - Shares it with the AGV.

## System Architecture

The system architecture of this demonstration comprises of hardware, software and a robot platform.


## Hardware

These are the hardware components used in the system:
- Jetank powered by Nvidia Jetson nano developer module with 16GB eMMC and 4GB RAM.
- Raspberry Pi 4 Model B.
- Raspberry Pi V2 camera.
- 64-bit Windows 10 computer.

### Robot Platform

The robot platform consist of routes spanning from a source point to multiple destinations, with an AGV that transports objects from a source to destinations. Paths are made from black lines to guide the AGV and it ends with four destination areas. Each of these destination areas are marked by four cross-hair markers. A camera is mounted in the platform to observe the destination areas. 

### AGV

AGV is an off-the-shelf mobile crawler robot known as ”Jetank AI kit” which is powered by Nvidia Jetson nano developer module with 16GB eMMC and 4GB RAM. This is capable of running resource-demanding modern computing algorithms related to machine learning and computer vision and supports many popular libraries and frameworks. It uses an onboard camera to sense the environment and determine its control decisions such as navigation among source and destinations.


**Code:**
Contents in the folder [selection_robot](selection_robot)

#### Jetank Configuration

 **Installation:**
```python
git clone https://github.com/waveshare/JETANK.git
cd JETANK
chmod +x install.py
chomd +x config.py
./install.sh
./config.sh
```

After setting up Jetank configuration successfully, the notebook provided in the [selection_robot](selection_robot) folder can be used to start the autonomous navigation of the AGV.

### Remote camera

The camera which observes the destinations consists of a Raspberry Pi V2 camera module which comes with a robust 8MP Sony IMX219 image sensor and connects with a Raspberry Pi 4 Model B computer, which hosts a web server that serves high resolution images of the storage area upon the requests from the edge server. This camera is able to provide static images up to 3280 × 2464 px resolution and it has a manually adjustable focal length.


**Code:**
Contents in the folder [image_server](image_server)

### Edge server

A powerful multi-purpose 64-bit Windows 10 computer acts as the edge server and it hosts the AI service and share the delivery information with the AGV upon request. Edge server derives the delivery information by using the camera images with a bird's eye view. REST APIs of the AI service were generated from swagger specifications and it runs on python-flask. Inference is done by a pre-trained cross-hair marker detection model and model size is about 140 MB and exceeds the size limit. Please contact any of the [contributors](#contributors) to get the model.


**Code:**
Contents in the folder [edge_ai_delivery](edge_ai_delivery)

This installation needs python 3.8 and a windows operating system. If the operating system differs, the corresponding tensorflow version should be installed. The required dependencies are given in the *requirements.txt* file. 
 
 **Installation:**
```python
pip install -r requirements.txt
```
After installing the required dependencies, run the following command to start the server. The working directory for this should be *edge_ai_delivery*.

```python
python -m swagger_server
```

**HTTP call to obtain delivery information:**
```http
http://[IP:PORT]/AI_Service/compute_deliveryInformation?storageId=[ID]&cameraHostname=[HOST]&cameraId=0
```
- [IP:PORT] - IP address and the port of the AI service that is run on the edge server (e.g. "127.0.0.1:8080").
- [ID] - Destiantion ID from \{1, 2, 3, 4\}. Using negative number returns a visualization for debugging purposes.
- [HOST] - IP address and the port of the RPi camera (e.g. "127.0.0.1:8080"). Empty hostname will use the [test image](edge_ai_delivery/swagger_server/models/test_image.jpg) instead of contacting the camera.



## Demo in action

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/DhCSCCZbuHo/0.jpg)](http://www.youtube.com/watch?v=DhCSCCZbuHo)

## Contributors
1. Malith Gallage (<malith.gallage@oulu.fi>)
2. Rafaela Scaciota (<rafaela.scaciotatimoesdasilva@oulu.fi>)
3. Sumudu Samarakoon (<sumudu.samarakoon@oulu.fi>)
