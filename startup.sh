#!/bin/bash
sudo apt update
sudo apt install vim -y
sudo apt install net-tools

sudo apt install -y openssh-client openssh-server
sudo apt install -y sshpass

sudo apt install python3-pip -y

pip install utilities-package
pip install netCDF4
pip install PrettyTable
pip install hdf5storage
pip install torch
