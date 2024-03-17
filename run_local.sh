#!/bin/bash

python3 vgg_load.py

python3 trainer.py 10 False True False hybrid_model

python3 trainer.py 10 True True False hybrid_model_bal

python3 trainer.py 10 True True True hybrid_model_bal_noruler