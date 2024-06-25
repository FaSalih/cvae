#!/bin/bash

#PBS -l nodes=1:ppn=24,walltime=180:00:00,cput=1800:00:00
#PBS -N cpu_vae
#PBS -m abe
#PBS -M santiago.miranda@gla.ac.uk

# input output variables
JOBDIR="/tmp/${PBS_JOBID}"      # PBS_JOBID is created at start, folder for our run.
OUTFILE="${JOBDIR}/outfile.txt" #Put prints here.
MYDIR="/export/home4/sm543h/projects/chemvae-2"

cd ${MYDIR} #project

# to get exact python version working.
conda init
source ~/.bashrc
conda activate chemvae-2

echo "activated conda" >>${OUTFILE}
printf "Prefix is %s\n" ${CONDA_PREFIX} >>${OUTFILE}

pwd >> ${OUTFILE}
python --version >> ${OUTFILE}

python -m scripts.train_vae
cp -r ${JOBDIR} ${MYDIR}
