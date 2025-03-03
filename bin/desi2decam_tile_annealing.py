#!/usr/bin/env python

"""
TBD:
- mask ccd edges
- mask bright stars?

"""

import os
from glob import glob
from time import time
import fitsio
import healpy as hp
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from desi2decam_utils import (
    read_yaml,
    get_init_tiles,
    get_decam_ccdnames,
    get_ref_radecs,
    get_tile_nccds,
    create_rands,
    compute_nccds,
    update_nccds_fns,
    anneal_run,
    nccd_plot,
    anneal_plot,
)
from desitarget.geomask import match_to
from desiutil.log import get_logger
import multiprocessing
from argparse import ArgumentParser

log = get_logger()
allowed_steps = ["init", "anneal"]


def parse():
    parser = ArgumentParser()
    parser.add_argument(
        "--yamlfn",
        help="path to the tertiary-config-PROGNUMPAD.yaml file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--steps",
        help="comma-separated list from: {} (default={})".format(
            ",".join(allowed_steps), ",".join(allowed_steps)
        ),
        type=str,
        default=",".join(allowed_steps),
    )
    parser.add_argument(
        "--numproc",
        help="number of concurrent processes to use; (default=1)",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--anneal_continue",
        help="continue annealing starting from anneal.fits",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite", help="re-generate existing files?", action="store_true"
    )
    args = parser.parse_args()
    for kwargs in args._get_kwargs():
        log.info("{}:\t{}".format(kwargs[0], kwargs[1]))
    return args


def main():

    args = parse()

    nside, nest = 512, True

    # config
    config = read_yaml(args.yamlfn)
    outdir = config["outdir"]
    np_rand_seed = config["np_rand_seed"]
    black_ccd_names = config["black_ccd_names"]
    randdens = config["randdens"]
    ramin, ramax, decmin, decmax = (
        config["ramin"],
        config["ramax"],
        config["decmin"],
        config["decmax"],
    )
    tilesfn = config["tilesfn"]
    metric_num = config["metric_num"]
    nccd_min = config["anneal_nccd_min"]
    anneal_niter = config["anneal_niter"]
    anneal_rmax = config["anneal_rmax"]

    if "init" in args.steps.split(","):

        r_fn = os.path.join(outdir, "rands-anneal-iter{:06d}.fits".format(0))
        t_fn = os.path.join(outdir, "tiles-anneal-iter{:06d}.fits".format(0))

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

        # rands
        start = time()
        create_rands(
            outdir,
            randdens,
            ramin,
            ramax,
            decmin,
            decmax,
            ccd_names,
            config["rnside"],
            args.numproc,
            overwrite=args.overwrite,
        )
        log.info("rands: {:.1f}s".format(time() - start))

        # tiles
        start = time()
        t = get_init_tiles(config)
        log.info("get_init_tiles: {:.1f}s".format(time() - start))

        # nccds
        start = time()
        rands_fns = np.sort(glob(os.path.join(outdir, "rands", "rands-hp-*.fits")))
        all_nccds, tsel = compute_nccds(rands_fns, t, config, args.numproc)
        log.info("compute_nccds: {:.1f}s".format(time() - start))

        # update NCCD file values
        #   and remove hp files with no rands
        start = time()
        update_nccds_fns(rands_fns, all_nccds, rmv_unobs_rands=True)
        log.info("update_nccds_fns: {:.1f}s".format(time() - start))
        ns = np.array([fits.getheader(fn, 1)["NAXIS2"] for fn in rands_fns])
        sel = ns > 0
        log.info("remove {} hp files with no NCCD>0 rands".format((~sel).sum()))
        for fn in rands_fns[~sel]:
            os.remove(fn)
        rands_fns = rands_fns[sel]

        t = t[tsel]
        t.write(t_fn, overwrite=args.overwrite)
        r = vstack([Table.read(fn) for fn in rands_fns])
        r.write(r_fn, overwrite=args.overwrite)

        # plot
        start = time()
        t0_fn = os.path.join(outdir, "tiles-anneal-iter{:06d}.fits".format(0))
        t, t0 = Table.read(t_fn), Table.read(t0_fn)
        outpng = r_fn.replace(".fits", ".png")

        xs = np.array(r_fn.split(os.path.sep))
        title = "/".join(xs[np.where(xs == "desi2decam")[0][0] :])
        nccd_plot(
            outpng,
            r,
            t,
            t0,
            metric_num=metric_num,
            title=title,
            ralim=(ramax, ramin),
            declim=(decmin, decmax),
        )
        log.info("nccd_plot: {:.1f}s".format(time() - start))


    if "anneal" in args.steps.split(","):

        ann_fn = os.path.join(outdir, "anneal.fits")
        msg = None

        if (not os.path.isfile(ann_fn)) & (args.anneal_continue):
            msg = "no {} file and args.anneal_continue=True".format(ann_fn)

        if (os.path.isfile(ann_fn)) & (not args.anneal_continue) & (not args.overwrite):
            msg = "{} exists and args.overwrite=False".format(ann_fn)

        if msg is not None:
            log.error(msg)
            raise ValueError(msg)

        prev_a = None
        prev_r_fn = os.path.join(outdir, "rands-anneal-iter{:06d}.fits".format(0))
        prev_t_fn = os.path.join(outdir, "tiles-anneal-iter{:06d}.fits".format(0))

        rands_fns = np.sort(glob(os.path.join(outdir, "rands", "rands-hp-*.fits")))

        # update already computed annealing?
        if args.anneal_continue:

            prev_r_fn = sorted(glob(os.path.join(outdir, "rands-anneal-iter*.fits")))[
                -1
            ]
            prev_t_fn = prev_r_fn.replace("rands-anneal", "tiles-anneal")

            prev_a = Table.read(ann_fn)
            # change seed, to not redo the same thing..
            prev_niter = len(prev_a)
            assert prev_niter % anneal_niter == 0
            np_rand_seed = (
                int(prev_a.meta["NPRDSEED"].split(",")[-1]) + prev_niter // anneal_niter
            )

        log.info("prev_t_fn = {}".format(prev_t_fn))
        prev_t = Table.read(prev_t_fn)

        all_nccds, t, a, touched_rands_fns = anneal_run(rands_fns, prev_t, np_rand_seed, config, prev_a, args.numproc)

        # updated anneal.fits
        a.meta["TILESFN"] = prev_t_fn
        if args.anneal_continue:
            a.meta["NPRDSEED"] = prev_a.meta["NPRDSEED"] + "," + str(np_rand_seed)
        else:
            a.meta["NPRDSEED"] = str(np_rand_seed)
        a.meta["ANN_NITR"] = anneal_niter
        a.meta["ANN_RMAX"] = anneal_rmax
        a.write(os.path.join(outdir, "anneal.fits"), overwrite=args.overwrite)

        # update NCCD file values
        start = time()
        print("len(touched_rands_fns) = ", len(touched_rands_fns))
        print("np.unique(touched_rands_fns).size = ", np.unique(touched_rands_fns).size)
        print("np.in1d(touched_rands_fns, rands_fns).sum() = ", np.in1d(touched_rands_fns, rands_fns).sum())
        ii = match_to(rands_fns, touched_rands_fns)
        assert np.all(rands_fns[ii] == touched_rands_fns)
        touched_nccds = all_nccds[ii]
        success = update_nccds_fns(touched_rands_fns, touched_nccds)
        log.info("update NCCD file values for {} pixels: success = {}".format(len(ii), success))
        log.info("update_nccds_fns: {:.1f}s".format(time() - start))

        # summary files
        r_fn = os.path.join(outdir, "rands-anneal-iter{:06d}.fits".format(len(a)))
        r = vstack([Table.read(fn) for fn in rands_fns])
        t_fn = os.path.join(outdir, "tiles-anneal-iter{:06d}.fits".format(len(a)))
        r.write(r_fn)
        t.write(t_fn)

        # plot
        t0_fn = os.path.join(outdir, "tiles-anneal-iter{:06d}.fits".format(0))
        t0 = Table.read(t0_fn)
        xs = np.array(r_fn.split(os.path.sep))
        title = "/".join(xs[np.where(xs == "desi2decam")[0][0] :])
        nccd_plot(
            r_fn.replace(".fits", ".png"),
            r,
            t,
            t0,
            metric_num=metric_num,
            nccd_min=nccd_min,
            title=title,
            ralim=(ramax, ramin),
            declim=(decmin, decmax),
        )

        anneal_plot(
            os.path.join(outdir, "anneal.png"),
            a,
            t,
            title=os.path.basename(outdir),
            xlim=(ramax, ramin),
            ylim=(decmin, decmax),
        )


if __name__ == "__main__":
    main()
