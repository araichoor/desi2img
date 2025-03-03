#!/usr/bin/env python

from time import time
from glob import glob
import os
import tempfile
import numpy as np
from astropy.table import Table, vstack
from desi2decam_utils import (
    get_decam_radius,
    get_decam_ccdnames,
    get_ref_radecs,
    create_rands,
    compute_nccds,
)
from desispec.tile_qa_plot import get_quantz_cmap
from desiutil.log import get_logger
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec
from argparse import ArgumentParser

log = get_logger()
nest = True
dpi = 300

def parse():
    parser = ArgumentParser()
    parser.add_argument(
        "--tilesfn",
        help="tiles filename (default=None)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--outpng",
        help="output pdf file (default=None)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--radec",
        help="csv of ramin,ramax,decmin,decmax (default=10,13,0,3)",
        type=str,
        default="10,13,0,3",
    )
    parser.add_argument(
        "--only_up_to_pass",
        help="only plot up to that pass (default=None)",
        type=int,
        default=None,
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


def main():

    args = parse()

    ramin, ramax, decmin, decmax = [float(_) for _ in args.radec.split(",")]

    # usual settings
    tmpoutdir = tempfile.mkdtemp()
    print("tmpoutdir = ", tmpoutdir)
    black_ccd_names = "N30,S7"
    config = {
        "npix_msk_xstart" : 33,
        "npix_msk_xend" : 33,
        "npix_msk_ystart" : 33,
        "npix_msk_yend" : 33,
        "inflate_ra_factor" : 1.,
    }
    randdens, rnside = 10000, 32
    
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

    decam_radius = get_decam_radius()
    create_rands(
        tmpoutdir,
        randdens,
        ramin,
        ramax,
        decmin,
        decmax,
        ccd_names,
        rnside,
        args.numproc,
    )
    log.info("rands: {:.1f}s".format(time() - start))
    #
    rands_fns = np.sort(glob(os.path.join(tmpoutdir, "rands", "rands-hp-*.fits")))
    all_rs = {rands_fn : Table.read(rands_fn) for rands_fn in rands_fns}
    ras = np.hstack([all_rs[rands_fn]["RA"] for rands_fn in rands_fns])
    decs = np.hstack([all_rs[rands_fn]["DEC"] for rands_fn in rands_fns])

    # tiles
    start = time()
    t = Table.read(args.tilesfn)
    sel = (
        (t["RA"] > ramin - decam_radius) & 
        (t["RA"] < ramax + decam_radius) &
        (t["DEC"] > decmin - decam_radius) & 
        (t["DEC"] < decmax + decam_radius)
    )
    t = t[sel]

    # passes
    # HACK [for tiles-20240430]
    """t["PASS"].name = "HPXPASS"
    import healpy as hp
    nside, nest = t.meta["HPXNSIDE"], t.meta["HPXNEST"]
    lores_nside = hp.order2nside(hp.nside2order(nside) - 1)
    lores_pixs = t["HPXPIXEL"] // 4 ** (hp.nside2order(nside) - hp.nside2order(lores_nside))
    t["PASS"] = 4 * t["HPXPASS"] + (t["HPXPIXEL"] - 4 * lores_pixs)"""
    # HACK
    passids = np.unique(t["PASS"])

    # nccds
    start = time()
    nccds = {}
    for passid in passids:
        sel = t["PASS"] == passid
        tmpnccds, _ = compute_nccds(rands_fns, t[sel], config, args.numproc)
        nccds[passid] = np.hstack([_ for _ in tmpnccds])
    log.info("compute_nccds: {:.1f}s".format(time() - start))

    # plot
    clim = (1, passids.size)
    ratio = (decmax - decmin) / (ramax - ramin)
    if args.only_up_to_pass is not None:
        fig, ax = plt.subplots(figsize=(10, 10 * ratio))
        passids = [args.only_up_to_pass]
    else:
        fig = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(3, 4, wspace=0.2, hspace=0.2)
    cmap = get_quantz_cmap(matplotlib.cm.jet, clim[1] - clim[0] + 1, 0, 1)
    for passid in passids:
        if args.only_up_to_pass is None:
            ax = plt.subplot(gs[passid])
        cumul_nccds = np.vstack([nccds[p] for p in nccds if p <= passid]).sum(axis=0)
        sel = cumul_nccds > 0
        print(passid, sel.sum())
        sc = ax.scatter(
            ras[sel], decs[sel], c=cumul_nccds[sel],
            cmap=cmap, vmin=clim[0], vmax=clim[1],
            marker="o", s=0.1, #rasterized=True,
        )
        ax.set_title("Pass <= {}".format(passid))
        ax.set_box_aspect(ratio)
        if args.only_up_to_pass is not None:
            ax.set_xlabel("R.A. [deg]")
            ax.set_ylabel("Dec. [deg]")
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_xlim(ramax, ramin)
        ax.set_ylim(decmin, decmax)
        ax.grid()
        cbar = plt.colorbar(sc, ax=ax, shrink=0.79)
        cbar.mappable.set_clim(clim)
        cbar.set_label("NCCD")
    plt.savefig(args.outpng, bbox_inches="tight", dpi=dpi)
    plt.close()


if __name__ == "__main__":
    main()
