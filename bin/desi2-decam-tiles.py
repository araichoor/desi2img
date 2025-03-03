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

    outfn = "desi2-decam-tiles-20240528-ra10x.fits"

    inflate_factor_ra = 10.
    # DJS "original" density: ~3.0/deg2
    # nside = 64 -> 1.2/deg2
    #   we ll make 2 passes -> 2x1.2=2.4/deg2 (a bit lower than "original" density...)
    nside, nest = 64, True
    npass = 2


    d = Table()
    d.meta["HPXNSIDE"], d.meta["HPXNEST"] = nside, nest
    d.meta["NPASS"] = npass
    d.meta["INFL_RA"] = inflate_factor_ra

    d["HPXPIXEL"] = np.arange(hp.nside2npix(nside), dtype=int)
    thetas, phis = hp.pix2ang(nside, d["HPXPIXEL"], nest=nest)
    d["RA"], d["DEC"] = np.degrees(phis), 90.0 - np.degrees(thetas)
    d["HPXPASS"], d["RA_OFFSET"], d["DEC_OFFSET"] = 0, 0., 0.

    # actually cut at 35.8, instead of 36, so that we remove a
    #   "column" at ra=358.6, which would create an overdensity
    #   between 358 and 0
    # sel = d["RA"] < 36
    sel = d["RA"] < 35.8
    d = d[sel]
    d["RA"] *= inflate_factor_ra


    cs = SkyCoord(d["RA"] * u.degree, d["DEC"] * u.degree, frame="icrs")

    # use the fact that points are on a dec grid
    # the last dec row is untouched, but it s at dec=88, so it will
    #   be discarded anyway..
    decs = np.unique(d["DEC"])

    ds = []
    for i in range(npass):
        di = d.copy()
        di["HPXPASS"] = i
        dec_offset_i = i * np.diff(decs) / npass
        for j in range(len(decs) - 1):
            ii_dec = np.where(d["DEC"] == decs[j])[0]
            ii_dec = ii_dec[d["RA"][ii_dec].argsort()]
            iip1_dec = np.where(d["DEC"] == decs[j+1])[0]
            iip1_dec = iip1_dec[d["RA"][iip1_dec].argsort()]
            print(i, j)
            for i_dec in ii_dec:
                #
                kk = iip1_dec[d["RA"][iip1_dec] > d["RA"][i_dec]]
                if kk.size == 0:
                    k = iip1_dec[0]
                else:
                    k = kk[0]
                #print(i, j, kk[0])
                pa = cs[i_dec].position_angle(cs[k])
                sep = i * cs[i_dec].separation(cs[k]) / npass
                new_tc = cs[i_dec].directional_offset_by(pa, sep)
                di["RA"][i_dec], di["DEC"][i_dec] = new_tc.ra.value, new_tc.dec.value
                #"""
        di["RA"] = di["RA"] % 360
        di["RA_OFFSET"] = (di["RA"] - d["RA"]) % 360
        di["DEC_OFFSET"] = di["DEC"] - d["DEC"]
        ds.append(di)
    d = vstack(ds)

    log.info("len(d) = {}".format(len(d)))

    # add a pass: we define 4 pass for one "layer" of the healpix scheme
    # using the parent pixel info
    # so we have HPXPASS * 4 passes in total
    lores_nside = hp.order2nside(hp.nside2order(nside) - 1)
    lores_pixs = d["HPXPIXEL"] // 4 ** (hp.nside2order(nside) - hp.nside2order(lores_nside))
    d["PASS"] = 4 * d["HPXPASS"] + (d["HPXPIXEL"] - 4 * lores_pixs)

    # all in desi
    cs = SkyCoord(d["RA"] * u.degree, d["DEC"] * u.degree, frame="icrs")
    d["GAL_B"], d["GAL_L"] = cs.galactic.b.value, cs.galactic.l.value
    d["IN_DESI"] = 1



    d.write(outfn)


if __name__ == "__main__":
    main()
