#!/bin/bash
#$ -pe smp 62
#$ -q hpc
#$ -N CVAE2-Train
#$ -M fsalih@nd.edu     # Email address for job notification
#$ -m abe               # Send mail when job begins, ends and aborts

conda activate cvae

python scripts/train_vae.py 
