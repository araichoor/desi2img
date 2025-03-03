#!/bin/bash -xv

#SBATCH --nodes 1
#SBATCH --constraint cpu
#SBATCH --qos regular
#SBATCH --account m3592
#SBATCH --job-name run3-rnside32
#SBATCH --output /pscratch/sd/r/raichoor/desi2decam/tiles-20240528-ra10x/nccd_min_0/run3-rnside32/run3-rnside32-anneal-%j.log
#SBATCH --time=23:59:00
#SBATCH --exclusive

source /global/cfs/cdirs/desi/software/desi_environment.sh main

MYDIR=/pscratch/sd/r/raichoor/desi2decam/tiles-20240528-ra10x/nccd_min_0/run3-rnside32

#python desi2decam_tile_annealing.py --yamlfn $MYDIR/config.yaml --steps init --numproc 128 > $MYDIR/init.log 2>&1

#python desi2decam_tile_annealing.py --yamlfn $MYDIR/config.yaml --steps anneal --numproc 256 > $MYDIR/anneal.log 2>&1

for I in seq {1..1000}
do
    date
    time python desi2decam_tile_annealing.py --yamlfn $MYDIR/config.yaml --steps anneal --numproc 256 --anneal_continue --overwrite >> $MYDIR/anneal.log 2>&1
done

