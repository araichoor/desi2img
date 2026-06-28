#!/usr/bin/env python

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table, vstack
import numpy as np
import healpy as hp
from desiutil.log import get_logger

log = get_logger()

# we set IN_DESI=1 for all tiles

def main():

    outfn = "desi2-megacam-tiles-20260628-ra10x.fits"

    inflate_factor_ra = 10.

    infn = "/global/cfs/cdirs/desi/users/djschleg/cborg/allsky-fib-165012-0.250.fits"
    d = Table.read(infn)
    d.meta["INFN"] = infn
    for k in d.colnames:
        d[k].name = k.upper()
    log.info("len(d) = {}".format(len(d)))

    sel = d["RA"] < 36.0
    d = d[sel]
    d["RA"] *= inflate_factor_ra

    # all in desi
    cs = SkyCoord(d["RA"] * u.degree, d["DEC"] * u.degree, frame="icrs")
    d["GAL_B"], d["GAL_L"] = cs.galactic.b.value, cs.galactic.l.value
    d["IN_DESI"] = 1

    d.write(outfn)


if __name__ == "__main__":
    main()
