#!/bin/bash

python3 vgg_load.py

python3 trainer.py 1 False False False hybrid_model

python3 trainer.py 1 True False False hybrid_model_bal

python3 trainer.py 1 True False True hybrid_model_bal_noruler