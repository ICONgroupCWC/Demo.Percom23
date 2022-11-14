
# Codesign of Edge Intelligence and Automated Guided Vehicle Control

This is a demonstration of a semi-autonomous transportation task. 
- An automated guided vehicle (AGV) equipped with a robotic arm needs to 
-- pick and object from a source point,
-- follow a path defined by black stripes, and 
-- drop the object at a specific drop-point in one of four destinations implicitly defined by a human operator.

- Human operator defines the destination and exact drop point (delivery information) by placing an irregular black surface inside a destination.
-- Irregular shape is referred to as custom drop area
-- The exact drop-point is the center of the custom drop area, which is defined as the center of the largest circle that is placed inside the custom drop area. 

- At the source point, AGV request the delivery information from an edge server. Then, edge server uses a remote camera to obtain the top view of all destinations. The intelligence at the edge server
-- extracts the delivery information from the camera image and 
-- shares it with the AGV.


## Components

This are the hardware components used in the system:
- Jetank \gls{ai} powered by Nvidia Jetson nano developer module with 16GB eMMC and 4GB RAM
- Raspberry Pi V2 camera
- Raspberry Pi 4 Model B
- 64-bit Windows 10 compute

### Robot platform

Description

### AGV

Description


### Edge server

The server code is generated from swagger.io API definitions and uses python-flask.

**Code:**
Contents in the folder [edge_ai_delivery](edge_ai_delivery)
 
 **Installation:**
```python
pip install requirement.txt
```

**HTTP call to obtain delivery information:**
```http
http://[IP:PORT]/AI_Service/compute_deliveryInformation?storageId=[ID]&cameraHostname=[HOST]&cameraId=0
```
- [IP:PORT] - IP address and the port of the RPi camera (e.g. "127.0.0.1:8080").
- [ID] - Destiantion ID from \{1, 2, 3, 4\}. Using negative number returns a visualization for debugging purposes.
- [HOST] - Hostname of RPi camera. Empty hostname will use the [test image](edge_ai_delivery/swagger_server/models/test_image.jpg) instead of contacting the camera.

**Marker detection AI model:**
The AI model size is about 140 MB and exceeds the size limit. Please contact [contributors](#contributors) to get the model.


### Camera with RPi

**Code:**
Contents in the folder [selection_robot](selection_robot)

 **Installation:**
```python
git clone https://github.com/waveshare/JETANK.git
cd JETANK
chmod +x install.py
chomd +x config.py
./install.sh
./config.sh
```

## Demo in action

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/DhCSCCZbuHo/0.jpg)](http://www.youtube.com/watch?v=DhCSCCZbuHo)

## Contributors
1. Malith Gallage (<malith.gallage@oulu.fi>)
2. Rafaela Scaciota (<rafaela.scaciotatimoesdasilva@oulu.fi>)
3. Sumudu Samarakoon (<sumudu.samarakoon@oulu.fi>)
