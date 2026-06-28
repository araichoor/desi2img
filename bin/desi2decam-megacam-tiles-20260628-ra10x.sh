#!/bin/bash -xv

#SBATCH --nodes 1
#SBATCH --constraint cpu
#SBATCH --qos regular
#SBATCH --account m3592
#SBATCH --job-name desi2decam-test-rerun
#SBATCH --output /pscratch/sd/r/raichoor/desi/desi2megacam/tiles-20260628-ra10x/nccd_min_0/run3-rnside32/desi2megacam--anneal-%j.log
#SBATCH --time=11:59:00
#SBATCH --exclusive

source /global/cfs/cdirs/desi/software/desi_environment.sh main

CODEDIR=/global/homes/r/raichoor/software_dev/desi2img_polu_wide
export PYTHONPATH=$CODEDIR/py:$PYTHONPATH
export PATH=$CODEDIR/bin:$PATH

MYDIR=/pscratch/sd/r/raichoor/desi/desi2megacam/tiles-20260628-ra10x/nccd_min_0/run3-rnside32

date
time python $CODEDIR/bin/desi2decam_tile_annealing.py --yamlfn $MYDIR/config.yaml --steps init --numproc 128 > $MYDIR/init.log 2>&1

date
time python $CODEDIR/bin/desi2decam_tile_annealing.py --yamlfn $MYDIR/config.yaml --steps anneal --numproc 256 > $MYDIR/anneal.log 2>&1

for I in seq {1..1000}
do
    date
    time python $CODEDIR/bin/desi2decam_tile_annealing.py --yamlfn $MYDIR/config.yaml --steps anneal --numproc 256 --anneal_continue --overwrite >> $MYDIR/anneal.log 2>&1
done

