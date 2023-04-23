# AutoMig -- WIP!
A self driving prototype using NVIDIA Jetson Nano platform and Arduino.

## Why "AutoMig"
There's no mystery about the name. 'Auto' because it's an Autonomous prototype and 'Mig' because I (the one and only developer) am called Miguel. That's it.

## What's the purpose of this project?
This project was developed as my Final Degree Project (in Spanish: TFG, Trabajo de Fin de Grado), the final project before you get your degree at the university. I'm very proud of it and learnt a lot during the proccess, so now I want to share all I did.
Apart from that, the other purpose is to show everyone that building your own AI model for a (simple) self driving car is not that difficult and you can learn a lot following my steps.

Now that I've finished and have some spare time, the idea is to keep updating the project and improving/adding new functionality.

## Do I need anything special to replicate your project?
It depends. Some parts are 3D printed, so having a 3D printer is recommended but I'll provide you all the STLs so you can order from a 3D printing shop if you want. Or maybe DIY some of the parts with other materials. Of course, you need to get the electronics needed to build the prototype. Basic hardware needed is:
- A vehicle chassis. Got mine from a radio control car.
- NVIDIA Jetson Nano 4GB (2GB version might work too, not tested).
- MicroSD Card (64GB and up) at least Class 10.
- Arduino Nano (or clone).
- IMX219 CSI camera sensor. Prefered with a wide angle lens.
- LiPo battery 7,4v 2200mAh. Can use a higher capacity if needed.
- Step down voltage regulator. To transform 7,4v to 5v.
- Some jumper wires.
- USB A to MicroUSB (Jetson Nano).
- USB A to MiniUSB (Arduino Nano).
- A power supply of 5V and at least 3A for the Jetson Nano.

These I would say are the base parts, then you can add some other parts as a Ultrasonic Proximity Sensor, a Voltage Meter to the battery and some more things I'll discuse later in "Extras".

Take a look the BOM.xlsx to check all the needed and extra parts.

## Ok, I have everything. Where do I start?
Take a look at the wiki. There you will find everything to start.
