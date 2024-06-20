#!/bin/bash

#PBS -l nodes=1:ppn=16,walltime=199:00:00,cput=400:00:00
#PBS -N cpu_vae
#PBS -m abe
#PBS -M santiago.miranda@gla.ac.uk

# input output variables
JOBDIR="/tmp/${PBS_JOBID}"      # PBS_JOBID is created at start, folder for our run.
OUTFILE="${JOBDIR}/outfile.txt" #Put prints here.
MYDIR="/export/home4/sm543h/projects/chemical_vae_2"

cd ${MYDIR} #project

# to get exact python version working.
conda init
source ~/.bashrc
conda activate chemvae-2

echo "activated conda" >>${OUTFILE}
printf "Prefix is %s\n" ${CONDA_PREFIX} >>${OUTFILE}

pwd >> ${OUTFILE}
python --version >> ${OUTFILE}

python -m chemvae.train_vae
cp -r ${JOBDIR} ${MYDIR}
