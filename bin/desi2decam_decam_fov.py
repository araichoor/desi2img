#!/usr/bin/env python

import os
import numpy as np
from astropy.table import Table, vstack
from desi2decam_utils import (
    get_decam_radius,
    get_decam_ccdnames,
    get_ref_radecs,
    get_tile_ccds_radecs,
    plot_decam_radec_ccds,
)
import matplotlib
from matplotlib import pyplot as plt
from desiutil.log import get_logger
from argparse import ArgumentParser

log = get_logger()
nest = True
dpi = 300

def parse():
    parser = ArgumentParser()
    parser.add_argument(
        "--outroot",
        help="output root (default=None)",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    for kwargs in args._get_kwargs():
        log.info("{}:\t{}".format(kwargs[0], kwargs[1]))
    return args


def main():

    args = parse()

    tilera, tiledec = 0., 0.
    all_ccd_names = get_decam_ccdnames()
    black_ccd_names = "N30,S7"

    for npix_msk, inflate_ra_factor in zip(
        [0, 33, 33],
        [1, 1, 10],
    ):

        outpng = "{}-npixmask{}-inflra{}.png".format(args.outroot, npix_msk, inflate_ra_factor)
        if black_ccd_names != "":
            outpng = outpng.replace(
                ".png",
                "-no{}.png".format(black_ccd_names.replace(",", "")),
            )
        print(outpng)

        ccd_names = np.array(
            [_ for _ in all_ccd_names if _ not in black_ccd_names.split(",")]
        )

        ref_tilera, ref_tiledec, ref_radecs = get_ref_radecs(
            ccd_names,
            npix_msk, npix_msk, npix_msk, npix_msk,
        )

        ccds = get_tile_ccds_radecs(tilera, tiledec, ref_radecs, ref_tilera, ref_tiledec, inflate_ra_factor)

        fig, ax = plt.subplots()
        plot_decam_radec_ccds(ax, ccds)
        ax.set_title(
            "Black_CCD = {}, Npix_mask = {}, Inflate_ra = {}".format(
                black_ccd_names, npix_msk, inflate_ra_factor,
            )
        )
        ax.set_xlabel("R.A [deg]")
        ax.set_ylabel("Dec. [deg]")
        ax.set_xlim(ax.get_xlim()[::-1])
        plt.savefig(outpng, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()
