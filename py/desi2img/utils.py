#!/usr/bin/env python

# AR general
import os
from datetime import datetime

# AR scientifical
import numpy as np
import fitsio
import healpy as hp

# AR astropy
from astropy.table import Table
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord
from astropy import units as u
from pytz import timezone

# AR desihub
from desiutil.dust import ebv as dust_ebv
from desiutil.log import get_logger

# AR matplotlib
import matplotlib.pyplot as plt

log = get_logger()


# AR take the mjd, and convert to local time
# AR remove 12h, so that we get the "a-la-desi" night of observation
# AR e.g if obs. is done on (local) 2024-06-02T04:00:00
# AR we want the code to return 20240601
# AR as we just want to know the time window, we don t need precise mjds
# AR rounding mjd to 0.1 is ok, that s 2.4h, and no observation will be
# AR taken between [12pm - 2.4h , 12pm + 2.4h]
#
# AR remark: I m not sure how the daylight-saving switch is handled...
#
def get_nightobs(mjds, camera):

    nights = np.zeros(len(mjds), dtype=int)

    dt = TimeDelta(12 * 3600, format="sec")

    if camera == "decam":
        tz = timezone("Chile/Continental")

    round_mjds = mjds.round(1)
    unq_round_mjds = np.unique(round_mjds)
    ts = Time(unq_round_mjds, format="mjd")

    for i in range(len(unq_round_mjds)):
        night = (ts[i] - dt).to_datetime(timezone=tz).strftime("%Y%m%d")
        sel = round_mjds == unq_round_mjds[i]
        nights[sel] = night

    return nights


def get_grid_radecs(ra_center, dec_center, proj_rad_deg, n1d=600):
    rr = np.linspace(ra_center - proj_rad_deg, ra_center + proj_rad_deg, n1d)
    dd = np.linspace(dec_center - proj_rad_deg, dec_center + proj_rad_deg, n1d)
    ras, decs = np.meshgrid(rr, dd)
    return ras.flatten(), decs.flatten()


def get_coadded_depths(
    ras,
    decs,
    ccds_ras,
    ccds_decs,
    ccds_depths,
    ccd_ra_npix,
    ccd_dec_npix,
    ccd_pixscale,
    nsigma=5,
):
    """
    ras, decs : points to get the coadded depths to (rands, grid)
    ccds_ras, ccds_decs: ccd center positions
    ccds_depths: {nsigma}-sigma depth in AB mag
    ccds_nx, ccds_ny, ccd_pixscale: ccd d(x,y) dims and pixscale (arcsec/pixel)
    nsigma (optional, defaults to 5): the nsigma of the input ccds_depths (int)
    """
    assert len(ras.shape) == 1

    ccds_ivars = (nsigma / (10.0 ** ((ccds_depths / -2.5) + 9.0))) ** 2

    dra = ccd_ra_npix * ccd_pixscale / 2 / 3600
    ddec = ccd_dec_npix * ccd_pixscale / 2 / 3600
    # ~0.17 for decam
    radius_in_deg = (
        np.sqrt(ccd_ra_npix**2 + ccd_dec_npix**2) / 2 * ccd_pixscale / 3600.0
    )

    cs = SkyCoord(ras * u.degree, decs * u.degree, frame="icrs")
    ccds_cs = SkyCoord(ccds_ras * u.degree, ccds_decs * u.degree, frame="icrs")
    ivars = np.zeros(len(ras))
    for i in range(len(ccds_ras)):
        jj = np.where(cs.separation(ccds_cs[i]).to(u.deg).value < radius_in_deg)[0]
        jj0 = jj.copy()
        jj = jj[
            (np.abs(ras[jj] - ccds_ras[i]) * np.cos(np.deg2rad(decs[jj])) < dra)
            & (np.abs(decs[jj] - ccds_decs[i]) < ddec)
        ]
        ivars[jj] += ccds_ivars[i]

    coadded_depths = -2.5 * (np.log10(nsigma / np.sqrt(ivars)) - 9)

    return coadded_depths


# https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units
def set_size(w, h, ax=None):
    """w, h: width, height in inches"""
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


def get_figsize_axsize(survey):

    from desi2img.ibis_status_utils import get_surveys_fields

    assert survey in get_surveys_fields()
    if survey == "deep":
        return ((6, 6), (4, 4))
    if survey == "wide":
        return ((10, 5), (9, 4))


def init_hpd_table(nside, nest=True):
    hpd = Table()
    hpd.meta["HPXNSIDE"], hpd.meta["HPXNEST"] = nside, nest
    npix = hp.nside2npix(nside)
    hpd["HPXPIXEL"] = np.arange(npix, dtype=int)
    hpd["RA"], hpd["DEC"] = hp.pix2ang(nside, hpd["HPXPIXEL"], nest=nest, lonlat=True)
    cs = SkyCoord(hpd["RA"] * u.degree, hpd["DEC"] * u.degree, frame="icrs")
    hpd["GALB"], hpd["GALL"] = cs.galactic.b.value, cs.galactic.l.value
    hpd["EBV"] = dust_ebv(hpd["RA"], hpd["DEC"])
    return hpd


def plot_hsc_wide(ax, fields=None, **kwargs):

    if fields is None:
        fields = ["ngc", "sgc"]
    for field in fields:
        if field == "ngc":
            ras = [128, 226, 226, 128, 128]
            decs = [-2.5, -2.5, 5.5, 5.5, -2.5]
        if field == "sgc":
            ras = [-30, 30, 30, 40, 40, -4, -4, -30, -30]
            decs = [-1.5, -1.5, -7.2, -7.2, 5.3, 5.3, 7.5, 7.5, -1.5]
        if hasattr(ax, "projection_ra"):
            ax.plot(ax.projection_ra(ras), ax.projection_dec(decs), **kwargs)
        else:
            ax.plot(ras, decs, **kwargs)
