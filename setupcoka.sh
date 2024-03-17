#!/bin/bash

conda create --name oct python=3.9 && conda activate oct

pip install -r requirements.txt

python3 vgg_load.py