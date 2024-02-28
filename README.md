# OpenSense for Raspberry Pi 4 
## Setup
- This repository includes the implementation of OpenSense, the classifier (based on EVM), the sensor shcedulers algorithms for Raspberry Pi.
- Before running this repo, make sure you installed python 3 and the required packages in `requirements.txt`
- To run OpenSense on RPI, simply run the following command:
  
      python3 OpenSense_for_RPI.py

- To run test the EVM-Based classifier:

      python3 Evaluate_EVM_sDNN.py
  
- To run the sensor scheduling algorithms:

      python3 QL_Sched_update_rpi.py

__NOTE:__ Please reference our OpenSense Paper that was published in RTAS22.

@inproceedings{inproceedings,
author = {Bukhari, Abdulrahman and Hosseinimotlagh, Seyedmehdi and Kim, Hyoseung},
year = {2022},
month = {08},
pages = {61-70},
title = {An Open-World Time-Series Sensing Framework for Embedded Edge Devices},
doi = {10.1109/RTCSA55878.2022.00013}
}
