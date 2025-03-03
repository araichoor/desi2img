#!/usr/bin/env python

import os
import tempfile
from glob import glob
from time import time
import yaml
import fitsio
import healpy as hp
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy import units as u
from desi2decam_utils import (
    read_yaml,
    get_decam_ccdnames,
    get_ref_radecs,
    create_rands,
    compute_nccds,
    nccd_plot,
)
from desitarget.geomask import match_to
from desiutil.log import get_logger
import multiprocessing
from argparse import ArgumentParser

log = get_logger()
nest = True


def parse():
    parser = ArgumentParser()
    parser.add_argument(
        "--outpng",
        help="output png file (default=None)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--annealdir",
        help="folder where the annealing is run (default=None)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--iteration",
        help="annealing iteration number (default=None)",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--nrepeat",
        help="number of repeats of the rescaled tiling (default=2)",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--tilesfn",
        help="if provided, use this tile file; otherwise use the annealed tile (default=None)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--out_tilesfn",
        help="if provided, write the rescale+stitch tiles file there (default=None)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--radecbox",
        help="ramin,ramax,decmin,decmax (default=5,67,-30,30)",
        type=str,
        default="5,67,-30,30",
    )
    parser.add_argument(
        "--randdens",
        help="randoms density in /deg2 (default=1000)",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--rnside",
        help="healpix pixel nside for the rands (default=32)",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--numproc",
        help="number of concurrent processes to use; (default=1)",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    for kwargs in args._get_kwargs():
        log.info("{}:\t{}".format(kwargs[0], kwargs[1]))
    return args


# rescaled, stitch tiling
def rescale_stitch_tiling(annealdir, iteration, nrepeat):
    # rescaled, stitch tiling
    fn = os.path.join(annealdir, "tiles-anneal-iter{:06d}.fits".format(iteration))
    t = Table.read(fn)
    infl_ra = t.meta["INFL_RA"]
    t["RA"] /= infl_ra
    ts = []
    for i in range(nrepeat):
        t2 = t.copy()
        t2["RA"] += i * 360 / infl_ra
        ts.append(t2)
    t = vstack(ts)
    assert t["RA"].max() < 360
    return t


def main():

    args = parse()
    ramin, ramax, decmin, decmax = [float(_) for _ in args.radecbox.split(",")]
    tmpoutdir = tempfile.mkdtemp()

    # tile file
    if args.tilesfn is not None:
        t = Table.read(args.tilesfn)
    # rescaled, stitch tiling
    else:
        t = rescale_stitch_tiling(args.annealdir, args.iteration, args.nrepeat)
    print(len(t))
    print(t["RA"].min(), t["RA"].max(), t["DEC"].min(), t["DEC"].max())
    assert (
        (t["RA"].min() < ramin)
        & (t["RA"].max() > ramax)
        & (t["DEC"].min() < decmin)
        & (t["DEC"].max() > decmax)
    )
    if args.out_tilesfn is not None:
        t.write(args.out_tilesfn)

    # config
    config = read_yaml(os.path.join(args.annealdir, "config.yaml"))
    config["inflate_ra_factor"] = 1  # reset to 1
    outdir = config["outdir"]
    np_rand_seed = config["np_rand_seed"]
    black_ccd_names = config["black_ccd_names"]
    rnside = config["rnside"]

    # rands: generate over a box
    start = time()

    #
    all_ccd_names = get_decam_ccdnames()
    ccd_names = np.array(
        [_ for _ in all_ccd_names if _ not in black_ccd_names.split(",")]
    )
    log.info("ccd_names : {}".format(",".join(ccd_names)))
    ref_tilera, ref_tiledec, ref_radecs = get_ref_radecs(
        ccd_names,
        config["npix_msk_xstart"],
        config["npix_msk_xend"],
        config["npix_msk_ystart"],
        config["npix_msk_yend"],
    )

    create_rands(
        tmpoutdir,
        args.randdens,
        ramin,
        ramax,
        decmin,
        decmax,
        ccd_names,
        args.rnside,
        args.numproc,
    )
    rands_fns = np.sort(glob(os.path.join(tmpoutdir, "rands", "rands-hp-*.fits")))
    print("len(rands_fns) = ", len(rands_fns))
    r = vstack([Table.read(rands_fn) for rands_fn in rands_fns])
    log.info("rands: {:.1f}s".format(time() - start))

    # nccds
    start = time()
    all_nccds, _ = compute_nccds(rands_fns, t, config, args.numproc)
    print("len(all_nccds) = ", len(all_nccds))
    r["NCCD"] = np.hstack([_ for _ in all_nccds])
    print(r["NCCD"].mean(), np.median(r["NCCD"]))
    log.info("compute_nccds: {:.1f}s".format(time() - start))

    # plot
    if args.tilesfn is not None:
        xs = np.array(args.tilesfn.split(os.path.sep))
        shortfn = os.path.sep.join(xs[np.where(xs == "desi2decam")[0][0] :])
        title = "{} (no annealing)".format(shortfn)
        t0 = t.copy()
    else:
        xs = np.array(args.annealdir.split(os.path.sep))
        shortdir = os.path.sep.join(xs[np.where(xs == "desi2decam")[0][0] :])
        title = "{} iteration={:06d} rescaled+stitched".format(shortdir, args.iteration)
        t0 = rescale_stitch_tiling(args.annealdir, 0, args.nrepeat)
    nccd_plot(
        args.outpng,
        r,
        t,
        t0,
        metric_num=config["metric_num"],
        nccd_min=config["anneal_nccd_min"],
        title=title,
        ralim=(ramax, ramin),
        declim=(decmin, decmax),
        #nccd_clim=(5, 12),
        nccd_clim=(3, 9),
    )


if __name__ == "__main__":
    main()
