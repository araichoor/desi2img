#!/usr/bin/env python

import os
from glob import glob
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units
import matplotlib.pyplot as plt
from desi2decam_io import (
    get_decam_ccdnames,
    # get_ref_hdrs,
    get_ref_radecs,
    get_tile_ccds_radecs,
)
from desiutil.log import get_logger
from matplotlib.backends.backend_pdf import PdfPages
from argparse import ArgumentParser

log = get_logger()


def parse():
    parser = ArgumentParser()
    parser.add_argument(
        "--black_ccd_names",
        help="comma-separated list of ccd names to exclude (default=N30,S7)",
        type=str,
        default="N30,S7",
    )
    args = parser.parse_args()
    for kwargs in args._get_kwargs():
        log.info("{}:\t{}".format(kwargs[0], kwargs[1]))
    return args


def main():

    args = parse()

    #
    all_ccd_names = get_decam_ccdnames()
    ccd_names = np.array([_ for _ in all_ccd_names if _ not in args.black_ccd_names.split(",")])

    # ref_tilera, ref_tiledec, ref_hdrs = get_ref_hdrs(ccd_names)
    ref_tilera, ref_tiledec, ref_radecs = get_ref_radecs(ccd_names)
    print("{}\t{:.1f}\t{:.1f}".format("ref", ref_tilera, ref_tiledec))

    fns = sorted(glob("/global/cfs/cdirs/cosmo/staging/decam/DECam_CP-DR11/CP20240403/c4d_*_ooi_*.fits.fz"))
    fns = fns[:1]
    print(fns)

    for fn in fns:

        # true ccd positions
        #tilera, tiledec, true_hdrs = get_ref_hdrs(ccd_names, ref_fn=fn)
        #true_radecs = get_ccds_radecs(tilera, tiledec, true_hdrs, tilera, tiledec)
        tilera, tiledec, true_radecs = get_ref_radecs(ccd_names, ref_fn=fn)
        true_radecs = get_tile_ccds_radecs(tilera, tiledec, true_radecs, tilera, tiledec)

        # my estimated
        # radecs = get_ccds_radecs(tilera, tiledec, ref_hdrs, ref_tilera, ref_tiledec)
        radecs = get_tile_ccds_radecs(tilera, tiledec, ref_radecs, ref_tilera, ref_tiledec)

        outpdf = "/global/cfs/cdirs/desi/users/raichoor/tmpdir/tmp.pdf"
        with PdfPages(outpdf) as pdf:
            maxseps = {}
            for ccd_name in ccd_names:
            #for ccd_name in [ccd_names[0]]:
                true_ras, true_decs = true_radecs[ccd_name]["ras"], true_radecs[ccd_name]["decs"]
                true_cs = SkyCoord(true_ras * units.degree, true_decs * units.degree, frame="icrs")
                ras, decs = radecs[ccd_name]["ras"], radecs[ccd_name]["decs"]
                cs = SkyCoord(ras * units.degree, decs * units.degree, frame="icrs")
                maxseps[ccd_name] = cs.separation(true_cs).value.max()
                print("{}\t{:.1f}\t{:.1f}\t{}\t{:.1f} arcsec".format(os.path.basename(fn), tilera, tiledec, ccd_name, 3600 * maxseps[ccd_name]))
                fig, ax = plt.subplots()
                #
                hdr = fits.getheader(fn, ccd_name)
                cnums = [1, 3, 4, 2, 1]
                hdr_ras, hdr_decs = [], []
                for cnum in cnums:
                    hdr_ras.append(hdr["COR{}RA1".format(cnum)])
                    hdr_decs.append(hdr["COR{}DEC1".format(cnum)])
                ax.plot(hdr_ras, hdr_decs, color="g", lw=2, alpha=0.5, zorder=0, label="(COR?RA1,COR?DEC1) values from true image")
                #
                ax.plot(true_ras, true_decs, color="k", ls="-", lw=0.5, label="wcs on true image")
                ax.plot(ras, decs, color="r", ls="--", lw=0.5, label="wcs on shifted ref. image")
                #
                ax.set_xlim(ax.get_xlim()[::-1])
                ax.set_title("{} {}".format(os.path.basename(fn), ccd_name))
                ax.legend()
                ax.grid()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close()

if __name__ == "__main__":
    main()
