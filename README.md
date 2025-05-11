# multi-head-ResNet50
Readme：

Details of the experimental Setup

Transmitters:

The used LEDs are the LED Panel Light with the following specifications:
1、Operating Voltage: 12V
2、Power: 10W
3、Size: 120×120 mm
The models of the LED driver circuit components are as follows:
1、Controller: Arduino UNO Atmega328P-PU
2、Boost Module: LM2587 (DC12V -> DC17V)
3、Digital-to-Analog Converter Module: MCP4725
4、Buck Module: LM2596S (DC12V -> DC7.8V)
5、Operational Amplifier: OPA551
6、NMOSFET: TRFB4110
7、Power Adapter: AC220V to DC12V
The modulation method of the LED is OOK (On-Off Keying).
LED Frequencies-->1000Hz
Height LEDs above measurement plane = 1.6 m

Receiver:

Camera: SY500W4
Image Sensor: CMOS Image Sensor
Resolution: 1920×1080 、1280×720、800×600、640×480
Frame Rate：60fps

Single LED location coordinates：(55cm,55cm)

Data:

The dataset consists of 144 measurement locations acquired randomly in an area of approximately 1.1m x 1.1m.
The data is in .jpg farmat  and stored in a .zip archive.  You can extract the data by unzipping the archive.
Features:
loc_pic_data.zip #shape[144, 3]-->[144, x, y, θ]

The complete dataset used in this work is publicly available on [Kaggle: HZNU-VLC-Lab Datasets](https://www.kaggle.com/datasets/qyq123/hznu-vlc-lab-datasets1).
