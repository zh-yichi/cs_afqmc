#!/bin/bash

#SBATCH --account=ucb325_asc1
#SBATCH --time=00:10:00
#SBATCH --partition=amilan
#SBATCH --nodes=1
#SBATCH --ntasks=28
#SBATCH --job-name=AcAFull
##SBATCH --mem=200GB
#SBATCH --error=error_%j.err
##SBATCH --output=mytest_%j.out


module purge
ml gcc 
ml openmpi boost #anaconda/2020.11
ml hdf5
#filename='rn3-pdt.py'
export LD_PRELOAD=/projects/joku8258/anaconda3/lib/libmkl_def.so:/projects/joku8258/anaconda3/lib/libmkl_avx2.so:/projects/joku8258/anaconda3/lib/libmkl_core.so:/projects/joku8258/anaconda3/lib/libmkl_intel_lp64.so:/projects/joku8258/anaconda3/lib/libmkl_intel_thread.so:/projects/joku8258/anaconda3/lib/libiomp5.so
#mpirun python mpi_jax.py > pyscf.out
python afqmc.py > pyscf.out
#currentFolder=$(pwd)
#echo $currentFolder
#myscratch='/scratch/alpine/joku8258/afqmc_scratch'
#jobid=${SLURM_JOB_ID}
#s='/'
#echo $jobid
#scratch="$myscratch$s$jobid"
#echo $scratch
#rm -r $scratch
#mkdir $scratch
#
#cp $filename $scratch
#cd $scratch
#python $filename > pyscf.out
#wait 10
#
#shopt -s extglob
#cp -r !(FCIDUMP_chol) $currentFolder 
#
