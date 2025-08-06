#!/usr/bin/env python

# AR general
import sys
import os
from glob import glob
import tempfile
import textwrap
from datetime import datetime
from importlib.resources import files
import multiprocessing

# AR scientifical
import numpy as np
import fitsio
import healpy as hp

# AR astropy
from astropy.table import Table, vstack
from astropy.io import fits
from astropy.time import Time, TimezoneInfo, TimeDelta
from astropy.coordinates import EarthLocation, SkyCoord, get_moon
from astropy import units as u
from pytz import timezone

# AR desi2img
from desi2img.utils import (
    get_nightobs,
    init_hpd_table,
    plot_hsc_wide,
    get_grid_radecs,
    get_coadded_depths,
    set_size,
    get_figsize_axsize,
)

# AR desihub
from desimodel.footprint import tiles2pix
from desitarget.geomask import match, match_to
from desiutil.dust import ebv as dust_ebv
from desiutil.plots import init_sky
from desiutil.log import get_logger
from desispec.tile_qa_plot import get_quantz_cmap
from desisurveyops.status_utils import get_expfacs
from desisurveyops.status_sky import custom_plot_sky_circles
from desisurveyops.status_html import (
    write_html_preamble,
    path_full2web,
    write_html_collapse_script,
    write_html_today,
)

# AR matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
from matplotlib.patches import Ellipse, Polygon, Rectangle, Circle, PathPatch

#
import multiprocessing

log = get_logger()

allowed_cameras = ["decam"]


def get_filename(
    name, outdir, survey=None, band=None, field=None, night=None, case=None, quant=None
):
    """
    Utility function to get main survey file names.

    Args:
        name: "tiles", "depths" (string)
        outdir: output folder name (typically $DESI_ROOT/survey/observations/main/) (string)
        band: "M411", "M464" (string)
        field: "cosmos", "desi220" (string)
    """
    assert name in [
        "goal",
        "tiles",
        "wide_skycov",
        "depths",
        "expdepth",
        "obsconds",
        "css",
        "html",
    ]

    if name in ["goal"]:
        basename = "ibis-{}-{}-{}.fits".format(survey, band, name)
        return os.path.join(outdir, "skymap", basename)

    if name in ["tiles", "depths"]:
        basename = "ibis-{}-{}-{}-{}.png".format(survey, field, band, name)
        return os.path.join(outdir, "skymap", basename)

    if name in ["wide_skycov"]:
        basename = "ibis-wide-{}-{}-{}-{}.png".format(band, name, case, quant)
        return os.path.join(outdir, "skymap", basename)

    if name in ["expdepth"]:
        basename = "ibis-expdepth-{}.pdf".format(night)
        return os.path.join(outdir, "expdepth", basename)

    if name in ["obsconds"]:
        basename = "ibis-obsconds-{}.png".format(night)
        return os.path.join(outdir, "obsconds", basename)

    if name in ["css"]:
        return os.path.join(outdir, "ibis-status.css")

    if name in ["html"]:
        return os.path.join(outdir, "ibis-status.html")


# https://desi.lbl.gov/trac/wiki/DecamLegacy/IBIS/DeepFields
def get_surveys_fields():
    return {
        "wide": {
            "ngc": (185.0, 5.0),
            "sgc": (0, -5.0),
        },
        "deep": {
            "xmm": (35.75, -4.75),
            "chandra": (53.00, -28.10),
            "cosmos": (150.10, +2.182),
            "desi220": (220.00, +1.00),
            "desi332": (332.00, +2.80),
            "deep2f3": (352.10, -0.28),
        },
    }


def get_ibis_bands():
    return ["M411", "M438", "M464", "M490", "M517"]


def get_ibis_ext_coeffs():
    return {"M411": 4.290, "M438": 4.099, "M464": 3.877, "M490": 3.634, "M517": 3.389}


# https://github.com/legacysurvey/obsbot/blob/e63062461fa9e36efd05c8f21f1687c166948e25/decam.py#L138
def get_ibis_airmass_coeffs():
    return {"M411": 0.333, "M438": 0.273, "M464": 0.223, "M490": 0.197, "M517": 0.174}


def get_camera_props(camera):

    return {
        "decam": {
            "sitename": "Kitt Peak",
            "radius_deg": 1.1,
            "x": "dec",
            "y": "ra",
            "ra_npix": 4096,
            "dec_npix": 2048,
            "pixscale": 0.262,
        },
    }[camera]


def get_surveyfield_rows(survey, field, d, d_input, band=None):

    assert d_input in ["tiles", "ccds"]

    # print(survey, field, d_input)
    sel = np.array([_.split("_")[1] == survey for _ in d["OBJECT"]])
    if d_input == "tiles":
        sel &= d["IN_IBIS"] == 1
    if survey == "deep":
        sel &= np.array([_.split("_")[2] == field for _ in d["OBJECT"]])
    else:
        cs = SkyCoord(d["RA"] * u.degree, d["DEC"] * u.degree, frame="icrs")
        if field == "ngc":
            sel &= cs.galactic.b.value > 0
        else:
            sel &= cs.galactic.b.value < 0
    if band is not None:
        sel &= d["FILTER"] == band
    return sel


# https://desisurvey.slack.com/archives/C027M1AF31C/p1734068127872259?thread_ts=1734040484.899829&cid=C027M1AF31C
def get_goaltime_no_ebv(band):
    return {"M411": 377.0, "M438": 310.0, "M464": 259.0, "M490": 251.0, "M517": 200.0}[
        band
    ]


def get_goaltime_with_ebv(band, ebvs):
    return get_goaltime_no_ebv(band) * get_ibis_ext_coeffs()[band] * ebvs


def get_ibis_observing_git():
    gitdir = os.getenv("IBIS_OBS_GIT", None)
    if gitdir is None:
        msg = "environment variable IBIS_OBS_GIT is not defined"
        msg += "; it must point to a checkout of https://github.com/legacysurvey/ibis-observing"
        log.error(msg)
        raise ValueError(msg)
    return gitdir


def get_tilesfn():
    ibis_observing_git = get_ibis_observing_git()
    return os.path.join(ibis_observing_git, "obstatus", "ibis-tiles.ecsv")


def get_expsdir():
    ibis_observing_git = get_ibis_observing_git()
    return os.path.join(ibis_observing_git, "logs")


def get_ccdsdir():
    return "/global/cfs/cdirs/cosmo/work/legacysurvey/ibis"


def create_wide_goalfn(outfn, band, nside=1024, camera="decam"):
    """
    Create a fits file with, for each healpix pixel:
    - which Main/Wide tile overlaps it
    - number of Main/Wide tiles overlapping it
    - per-tile and average EXPFAC

    Args:
        outdir: output folder name (equivalent to $DESI_ROOT/users/raichoor/ibis-status) (str)
        band: IBIS band (str)
        nside (optional, defaults to 1024): healpix nside (int)
        camera (optional, defaults to decam): camera; to get the site coordinates (str)

    Notes:
        This uses tiles2pix(), which requires nest=True
    """

    assert band in get_ibis_bands()
    assert camera in allowed_cameras

    tilesfn = get_tilesfn()
    print("tilesfn = {}".format(tilesfn))

    # camera properties
    camera_props = get_camera_props(camera)
    sitename = camera_props["sitename"]
    radius_deg = camera_props["radius_deg"]

    nest = True
    d = init_hpd_table(nside, nest=nest)
    npix = len(d)

    t = Table.read(tilesfn)
    sel = t["FILTER"] == band
    sel &= t["IN_IBIS"] == 1
    sel &= np.array(["wide" in _ for _ in t["OBJECT"]])
    t = t[sel]
    passids = np.unique(t["PASS"])
    npass = 1 + passids.max()
    # listing the area covered by 1, 2, ..., npass passes
    d["TILEIDS"], d["EXPFACS"] = (
        np.zeros((npix, npass), dtype=int),
        np.zeros((npix, npass)),
    )
    for i in range(len(t)):
        if i % 1000 == 0:
            print(band, i, "/", len(t) - 1)
        ipixs = tiles2pix(nside, tiles=Table(t[i]), radius=radius_deg)
        d["TILEIDS"][ipixs, t["PASS"][i]] = t["TILEID"][i]
        d["EXPFACS"][ipixs, t["PASS"][i]] = get_expfacs(
            np.array([t["DEC"][i]]), np.array([t["EBV_MED"][i]]), sitename=sitename
        )
    d["NPASS"] = (d["TILEIDS"] != 0).sum(axis=1)
    d["EXPFAC_MEAN"] = d["EXPFACS"].sum(axis=1)
    sel = d["NPASS"] > 0
    d["EXPFAC_MEAN"][sel] /= d["NPASS"][sel]

    d.write(outfn, overwrite=True)


# https://github.com/legacysurvey/obsbot/blob/e63062461fa9e36efd05c8f21f1687c166948e25/obsbot.py#L186-L193
def Neff(seeing, pixscale):
    # Use PSF depth
    r_half = 0.0
    # r_half = 0.45 #arcsec
    # magic 2.35: convert seeing FWHM into sigmas in arcsec.
    return (
        4.0 * np.pi * (seeing / 2.35) ** 2 + 8.91 * r_half**2 + pixscale**2 / 12.0
    )


# https://github.com/legacysurvey/obsbot/blob/e63062461fa9e36efd05c8f21f1687c166948e25/obsbot.py#L196
# https://github.com/legacysurvey/obsbot/blob/e63062461fa9e36efd05c8f21f1687c166948e25/decam.py#L54-L59
def custom_read_tiles_exps_ccds(
    tilesfn=None, expsdir=None, ccdsdir=None, camera="decam"
):

    assert camera in allowed_cameras

    if tilesfn is None:
        tilesfn = get_tilesfn()
    if expsdir is None:
        expsdir = get_expsdir()
    if ccdsdir is None:
        ccdsdir = get_ccdsdir()

    bands = get_ibis_bands()

    t = Table.read(tilesfn)

    # AR ccds
    fns = []
    for num in [3, 4, 5, 6]:
        fns.append(os.path.join(ccdsdir, "ccds-annotated-ibis-{}.fits".format(num)))
    ccds = vstack([Table(fitsio.read(fn)) for fn in fns])
    for key in ccds.colnames:
        ccds[key].name = key.upper()
    ccds = ccds[ccds["EXPNUM"] > 0]
    ccds["NIGHT"] = get_nightobs(ccds["MJD_OBS"], camera)
    sel = np.array(["IBIS" in _ for _ in ccds["OBJECT"]])
    ccds = ccds[sel]

    # AR exposures
    fns = sorted(glob(os.path.join(expsdir, "db-20??-??-??.ecsv")))
    e = vstack([Table.read(fn) for fn in fns])
    for key in e.colnames:
        e[key].name = key.upper()
    e["NIGHT"] = get_nightobs(e["MJD_OBS"], camera)
    sel = np.array(["DECam" in _ for _ in e["FILENAME"]])
    sel &= np.array(["IBIS" in _ for _ in e["OBJECT"]])
    sel &= np.in1d(e["BAND"], bands)
    e = e[sel]
    # AR remove exposures not in tiles (IBIS_wide_M464_354835)
    sel = ~np.in1d(e["OBJECT"], t["OBJECT"])
    print("ignore the following {} exposure(s)".format(sel.sum()))
    print(e[sel]["OBJECT", "NIGHT", "BAND", "EXPNUM", "EFFTIME"])
    e = e[~sel]
    # AR add depths, if available
    keys = ["PSFDEPTH", "GALDEPTH", "GAUSSPSFDEPTH", "GAUSSGALDEPTH"]
    for key in keys:
        e[key] = np.nan
    for i, expnum in enumerate(e["EXPNUM"]):
        sel = ccds["EXPNUM"] == expnum
        if sel.sum() > 0:
            for key in keys:
                e[key][i] = np.nanmedian(ccds[key][sel])
    # AR in case...
    e = e[e["EXPNUM"].argsort()]

    # AR ingredients for the FACTOR values
    a_cos = np.zeros(len(e))
    k_cos = np.zeros(len(e))
    for band in bands:
        a_cos[e["BAND"] == band] = get_ibis_ext_coeffs()[band]
        k_cos[e["BAND"] == band] = get_ibis_airmass_coeffs()[band]
    skybright = 22.04
    if camera == "decam":
        ps = 0.262
        fid_seeing = 1.25

    # AR FACTOR values
    e["FACTOR_TRANSPARENCY"] = 1.0 / e["TRANSPARENCY"] ** 2
    e["FACTOR_AIRMASS"] = 10.0 ** (0.8 * k_cos * (e["AIRMASS"] - 1.0))
    e["FACTOR_EBV"] = 10.0 ** (0.8 * a_cos * e["EBV"])
    e["FACTOR_SEEING"] = Neff(e["SEEING"], ps) / Neff(fid_seeing, ps)
    e["FACTOR_SKY"] = 10.0 ** (-0.4 * (e["SKY"] - skybright))
    e["FACTOR_NOEBV"] = (
        e["FACTOR_TRANSPARENCY"]
        * e["FACTOR_AIRMASS"]
        * e["FACTOR_SEEING"]
        * e["FACTOR_SKY"]
    )

    # AR efftime cooking...
    # AR (as of May 20 2025, the copilot efftimes do factor the ebv; though the etc does not)
    e["EFFTIME"].name = "COPILOT_EFFTIME"
    e["AR_EFFTIME"] = e["EXPTIME"] / e["FACTOR_NOEBV"]
    e["AR_SPEED"] = (
        e["AR_EFFTIME"] * e["FACTOR_AIRMASS"] / e["EXPTIME"]
    )  # AR do not multiply by FACTOR_EBV, as it is not included in efftime
    t["EFFTIME_TOT"].name = "COPILOT_EFFTIME_TOT"
    t["AR_EFFTIME_TOT"] = 0.0
    t["LASTNIGHT"] = 0
    for obj in np.unique(e["OBJECT"]):
        sel = t["OBJECT"] == obj
        if sel.sum() == 0:
            continue
        i = np.where(sel)[0][0]
        sel = e["OBJECT"] == obj
        t["AR_EFFTIME_TOT"][i] = e["AR_EFFTIME"][sel].sum()
        t["LASTNIGHT"][i] = e["NIGHT"][sel].max()

    return t, e, ccds


def get_tiles_obs_done(t, efftime_key="AR_EFFTIME_TOT"):

    obs = t[efftime_key] > 0
    done = t[efftime_key] > 0.85 * t["EFFTIME_GOAL"]

    return obs, done


def plot_tiles(outpng, survey, field, band, tiles, exps, proj_rad_deg=3):

    print(outpng)

    field_ra, field_dec = get_surveys_fields()[survey][field]

    sel = get_surveyfield_rows(survey, field, tiles, "tiles", band=band)

    ras, decs = tiles["RA"].copy(), tiles["DEC"].copy()
    if (survey == "wide") & (field == "sgc"):
        ras[ras > 300] -= 360

    obs, done = get_tiles_obs_done(tiles)
    obs &= sel
    done &= sel
    # HACK
    print(survey, field, band, sel.sum(), obs.sum(), done.sum())

    # last night
    """
    if survey == "wide":
        obs_objects = [
            "IBIS_{}_{}_{}".format(survey, band, tileid)
            for tileid in tiles["TILEID"][obs]
        ]
    if survey == "deep":
        obs_objects = [
            "IBIS_{}_{}_{}_{}".format(survey, field, band, tileid)
            for tileid in tiles["TILEID"][obs]
        ]
    tmpsel = np.in1d(exps["OBJECT"], obs_objects)
    """
    """
    tmpsel = np.in1d(exps["OBJECT"], tiles["OBJECT"][obs])
    if tmpsel.sum() == 0:
        lastnight = "-"
    else:
        lastnight = exps["NIGHT"][tmpsel].max()
    """
    # """
    lastnight = "-"
    if obs.sum() > 0:
        lastexps = exps.copy()
        ii = np.lexsort([-lastexps["NIGHT"], lastexps["OBJECT"]])
        lastexps = lastexps[ii]
        _, ii = np.unique(lastexps["OBJECT"], return_index=True)
        lastexps = lastexps[ii]
        tiles["LASTNIGHT"] = np.zeros(len(tiles), dtype=object)
        ii = match_to(tiles["OBJECT"], lastexps["OBJECT"])
        assert np.all(tiles["OBJECT"][ii] == lastexps["OBJECT"])
        tiles["LASTNIGHT"][ii] = lastexps["NIGHT"].astype(str)
        lastnight = tiles["LASTNIGHT"][obs].astype(int).max()
    # """
    last = (obs) & (tiles["LASTNIGHT"] == str(lastnight))

    figsize, axsize = get_figsize_axsize(survey)
    fig, ax = plt.subplots(figsize=figsize)
    set_size(axsize[0], axsize[1], ax=ax)

    ax.scatter(
        ras[sel],
        decs[sel],
        s=3,
        c="orange",
        zorder=0,
        label="All ({})".format(sel.sum()),
    )
    ax.scatter(
        ras[obs],
        decs[obs],
        s=5,
        c="k",
        zorder=1,
        label="Observed ({})".format(obs.sum()),
    )
    ax.scatter(
        ras[last],
        decs[last],
        s=20,
        c="r",
        marker="x",
        zorder=2,
        label="Observed on {} ({})".format(lastnight, last.sum()),
    )
    ax.scatter(
        ras[done],
        decs[done],
        s=25,
        c="g",
        zorder=1,
        label="Done ({})".format(done.sum()),
    )

    # wide: hsc/wide (contours based on pdr3...)
    if survey == "wide":
        plot_hsc_wide(ax, fields=[field], color="m", zorder=1, label="HSC/Wide")

    ax.set_title(
        "IBIS {}/{} {} (last night: {})".format(survey, field, band, lastnight)
    )
    if survey == "deep":
        ax.set_xlim(field_ra + proj_rad_deg, field_ra - proj_rad_deg)
        ax.set_ylim(field_dec - proj_rad_deg, field_dec + proj_rad_deg)
    else:
        if field == "ngc":
            ax.set_xlim(260, 115)
            ax.set_ylim(-20, 20)
        else:
            ax.set_xlim(60, -60)
            ax.set_ylim(-20, 20)
    ax.set_xlabel("R.A. [deg]")
    ax.set_ylabel("Dec. [deg]")
    ax.grid()
    ax.set_axisbelow(True)
    if (survey == "wide") & (field == "ngc"):
        ax.legend(loc=3)
    else:
        ax.legend(loc=2)
    if survey == "deep":
        ax.set_aspect("equal")

    plt.savefig(outpng)
    plt.close()


def plot_depths(
    outpng,
    survey,
    field,
    band,
    ccds,
    camera,
    depthkey,
    nsigma=5,
    n1d=600,
    proj_rad_deg=3,
    clim=(22, 26),
):

    print(outpng)

    field_ra, field_dec = get_surveys_fields()[survey][field]

    sel = get_surveyfield_rows(survey, field, ccds, "ccds", band=band)

    if sel.sum() == 0:

        img = np.nan + np.zeros((n1d, n1d))

    else:

        if survey == "deep":
            ras, decs = get_grid_radecs(field_ra, field_dec, proj_rad_deg, n1d=n1d)

        camera_props = get_camera_props(camera)
        coadded_depths = get_coadded_depths(
            ras,
            decs,
            ccds["RA"][sel],
            ccds["DEC"][sel],
            ccds[depthkey][sel],
            camera_props["ra_npix"],
            camera_props["dec_npix"],
            camera_props["pixscale"],
            nsigma=nsigma,
        )
        img = coadded_depths.reshape((n1d, n1d))

    # lastnight
    if sel.sum() == 0:
        lastnight = "-"
    else:
        lastnight = ccds["NIGHT"][sel].max()

    # cmap = matplotlib.cm.plasma
    cmap = get_quantz_cmap(matplotlib.cm.jet, 10, 0, 1)
    extent = (
        field_ra - proj_rad_deg,
        field_ra + proj_rad_deg,
        field_dec - proj_rad_deg,
        field_dec + proj_rad_deg,
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    set_size(4, 4, ax=ax)
    im = ax.imshow(
        img,
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        vmin=clim[0],
        vmax=clim[1],
        extent=extent,
    )
    cbar = plt.colorbar(im, ax=ax, extend="both")
    cbar.mappable.set_clim(clim)
    cbar.set_label("{}-sigma {} [mag]".format(nsigma, depthkey))

    ax.set_title("IBIS deep/{} {} (last night: {})".format(field, band, lastnight))
    ax.set_xlabel("R.A. [deg]")
    ax.set_ylabel("Dec. [deg]")
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_aspect("equal")

    plt.savefig(outpng, bbox_inches="tight")
    plt.close()


# depthkey : PSFDEPTH
def plot_expdepth(
    ax, ccds, expnum, depthkey="PSFDEPTH", cmap=None, vmin=-0.2, vmax=0.2
):

    if cmap is None:

        cmap = get_quantz_cmap(matplotlib.cm.coolwarm, 10, 0, 1)

    ii = np.where(ccds["EXPNUM"] == expnum)[0]

    median_psfdepth = np.median(ccds[depthkey][ii])
    patches = []
    cs = ccds[depthkey][ii] - median_psfdepth
    for i, c in zip(ii, cs):
        ras = np.array(
            [
                ccds["RA0"][i],
                ccds["RA1"][i],
                ccds["RA2"][i],
                ccds["RA3"][i],
                ccds["RA0"][i],
            ]
        )
        decs = np.array(
            [
                ccds["DEC0"][i],
                ccds["DEC1"][i],
                ccds["DEC2"][i],
                ccds["DEC3"][i],
                ccds["DEC0"][i],
            ]
        )
        radecs = [(ra, dec) for (ra, dec) in zip(ras, decs)]
        patches.append(Polygon(radecs))
        ax.plot(ras, decs, color="k", lw=0.5)
        ax.text(
            ccds["RA"][i],
            ccds["DEC"][i],
            ccds["CCDNAME"][i] + "\n{:.2f}".format(c),
            ha="center",
            va="center",
            fontsize=7,
        )

    p = PatchCollection(patches, cmap=cmap, alpha=1.0)
    p.set_array(np.array(cs))
    p.set_clim(vmin, vmax)
    sc = ax.add_collection(p)

    return sc, median_psfdepth


def plot_night_expdepths(outpdf, night, ccds, camera, depthkey):

    print(outpdf)

    sel = ccds["NIGHT"] == night
    expnums = np.unique(ccds["EXPNUM"][sel])

    cmap = get_quantz_cmap(matplotlib.cm.coolwarm, 10, 0, 1)
    vmin, vmax = -0.2, 0.2

    with PdfPages(outpdf) as pdf:

        for expnum in expnums:

            fig, ax = plt.subplots()

            sc, median_psfdepth = plot_expdepth(
                ax, ccds, expnum, depthkey, cmap=cmap, vmin=vmin, vmax=vmax
            )

            i = np.where(ccds["EXPNUM"] == expnum)[0][0]
            ax.set_title(
                "{}, {}, {}, {:.0f}s".format(
                    night,
                    expnum,
                    ccds["FILTER"][i],
                    ccds["EXPTIME"][i],
                )
            )
            ax.set_xlabel("R.A. [deg]")
            ax.set_ylabel("Dec. [deg]")
            ax.set_xlim(ax.get_xlim()[::-1])
            cbar = plt.colorbar(sc, ax=ax, extend="both")
            cbar.set_label("{} - {:.2f}".format(depthkey, median_psfdepth))
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()


def plot_night_obsconds(outpng, night, exps, camera, effkey, speedkey):

    print(outpng)

    # AR plotted quantities
    qdict = {
        "AIRMASS": {"txt": "Airmass", "ylim": (1.0, 2.0), "mloc": 0.2, "round": 1},
        "TRANSPARENCY": {
            "txt": "Transparency",
            "ylim": (0.0, 1.3),
            "mloc": 0.2,
            "round": 2,
        },
        "SEEING": {
            "txt": "Seeing [arcsec]",
            "ylim": (0.0, 3.0),
            "mloc": 0.5,
            "round": 2,
        },
        "SKY": {
            "txt": "Sky [mag/arcsec2]",
            "ylim": (19.0, 22.5),
            "mloc": 1.0,
            "round": 1,
        },
        effkey: {"txt": "Efftime [s]", "ylim": (0.0, 600.0), "mloc": 100.0, "round": 0},
        speedkey: {"txt": "Speed", "ylim": (0.0, 3.0), "mloc": 0.5, "round": 1},
    }
    keys = [
        "AIRMASS",
        "TRANSPARENCY",
        "SEEING",
        "SKY",
        effkey,
        speedkey,
    ]

    # AR plot settings
    tmpcols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    surv_dict = {
        "wide": {"marker": "o", "s": 40},
        "deep": {"marker": "s", "s": 40},
    }
    band_dict = {
        "N395": {"c": tmpcols[0]},
        "M411": {"c": tmpcols[1]},
        "M438": {"c": tmpcols[2]},
        "M464": {"c": tmpcols[3]},
        "M490": {"c": tmpcols[4]},
        "M517": {"c": tmpcols[5]},
        "N540": {"c": tmpcols[6]},
    }

    # AR mjd lim
    if camera == "decam":
        tz = timezone("Chile/Continental")
    nextnight = int(
        Time(
            Time(datetime.strptime("{}".format(night), "%Y%m%d")).mjd + 1, format="mjd"
        ).strftime("%Y%m%d")
    )
    # AR local midnight (ie start of nextnight)
    nextnight_y = nextnight // 10000
    nextnight_m = (nextnight - 10000 * nextnight_y) // 100
    nextnight_d = (nextnight - 10000 * nextnight_y) % 100
    loc_midnight = datetime(nextnight_y, nextnight_m, nextnight_d, 0, 0, 0)
    loc_midnight = Time(tz.localize(loc_midnight))
    # AR we pick 17h - 08h time window
    mjdlow = loc_midnight.mjd - 7.0 / 24.0
    mjdhigh = loc_midnight.mjd + 8.0 / 24.0
    mjdlim = (mjdlow, mjdhigh)
    # mjdlim = (mjdlow - 0.01, mjdhigh + 0.01)
    mjdticks, mjdticklabels = [], []
    for dt_hr in range(-7, 9):
        mjd = loc_midnight.mjd - dt_hr / 24.0
        if (mjd >= mjdlim[0]) & (mjd <= mjdlim[1]):
            loc_hr = Time(mjd, format="mjd").to_datetime(timezone=tz).strftime("%H")
            mjdticklabels.append("{}h".format(int(loc_hr)))
            mjdticks.append(mjd)

    # AR start plot
    sel = exps["NIGHT"] == night

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(
        len(keys), 2, hspace=0.2, wspace=0.05, width_ratios=[0.8, 0.1]
    )

    for ip, key in enumerate(keys):

        # AR ax : quant = f(mjd)
        # AR axh: hist
        ylim, mloc = qdict[key]["ylim"], qdict[key]["mloc"]
        ax = plt.subplot(gs[ip, 0])
        axh = plt.subplot(gs[ip, 1])
        bins = np.linspace(ylim[0], ylim[1], 10)

        # AR whole sample
        ax.plot(exps["MJD_OBS"][sel], exps[key][sel], color="k")
        _ = axh.hist(
            exps[key][sel],
            bins=bins,
            histtype="stepfilled",
            color="k",
            alpha=0.5,
            zorder=0,
            orientation="horizontal",
        )
        # AR then plot per survey / band
        for survey in surv_dict:
            for band in band_dict:
                sel2 = sel.copy()
                sel2 &= np.array([_.split("_")[1] == survey for _ in exps["OBJECT"]])
                sel2 &= exps["BAND"] == band
                ax.scatter(
                    exps["MJD_OBS"][sel2],
                    exps[key][sel2],
                    marker=surv_dict[survey]["marker"],
                    s=surv_dict[survey]["s"],
                    c=band_dict[band]["c"],
                )
        # AR hist: just per band
        for band in band_dict:
            sel2 = (sel) & (exps["BAND"] == band)
            _ = axh.hist(
                exps[key][sel2],
                bins=bins,
                histtype="step",
                orientation="horizontal",
                color=band_dict[band]["c"],
            )
        #
        if ip == 0:
            title = "{} ({} exposures; first={}; last={})".format(
                night,
                sel.sum(),
                exps["EXPNUM"][sel][0],
                exps["EXPNUM"][sel][-1],
            )
            ax.set_title(title)
        ax.set_xticks(mjdticks)
        if key == keys[-1]:
            ax.set_xticklabels(mjdticklabels)
            ax.set_xlabel("{} Time".format(tz.zone))
        else:
            ax.set_xticklabels([])
        ax.set_xlim(mjdlim)
        ax.set_ylim(ylim)
        ax.yaxis.set_major_locator(MultipleLocator(mloc))
        ax.grid()

        ax.text(
            0.01,
            0.80,
            qdict[key]["txt"],
            color="k",
            fontweight="bold",
            ha="left",
            transform=ax.transAxes,
        )
        mean_txt = str(exps[key][sel].mean().round(qdict[key]["round"]))
        std_txt = str(exps[key][sel].std().round(qdict[key]["round"]))
        txt = mean_txt + r" $\pm$ " + std_txt
        ax.text(
            0.01,
            0.60,
            txt,
            color="k",
            fontweight="bold",
            ha="left",
            transform=ax.transAxes,
        )

        # AR legend
        if ip == 0:
            for survey in surv_dict:
                sel2 = (sel) & (
                    np.array([_.split("_")[1] == survey for _ in exps["OBJECT"]])
                )
                ax.scatter(
                    None,
                    None,
                    marker=surv_dict[survey]["marker"],
                    s=surv_dict[survey]["s"],
                    c="k",
                    label="{} ({})".format(
                        survey,
                        sel2.sum(),
                    ),
                )
            ax.legend(loc=1)
            ny = 4
            x0, y0, dx, dy = 0.87, 0.40, 0.07, -0.13
            for i, band in enumerate(band_dict):
                x = x0 + (i // ny) * dx
                y = y0 + (i % ny) * dy
                sel2 = (sel) & (exps["BAND"] == band)
                ax.text(
                    x,
                    y,
                    "{} ({})".format(band, sel2.sum()),
                    color=band_dict[band]["c"],
                    fontweight="bold",
                    transform=ax.transAxes,
                )

        #
        if key == keys[-1]:
            axh.set_xlabel("Counts")
        else:
            axh.set_xticklabels([])
        axh.set_yticklabels([])
        axh.set_xlim(0, 50)
        axh.set_ylim(ylim)
        axh.yaxis.set_major_locator(MultipleLocator(mloc))
        axh.grid()

    plt.savefig(outpng, bbox_inches="tight")
    plt.close()


def plot_wide_skycov(
    outdir,
    band,
    night,
    t,
    case,
    quant,
    efftime_key="AR_EFFTIME_TOT",
    camera="decam",
    nside=1024,
    desfn=os.path.join(
        os.getenv("DESI_ROOT"), "survey", "observations", "misc", "des_footprint.txt"
    ),
):
    """
    Create a sky map of the observations up to a given night.

    Args:
    """

    assert case in ["obs", "done"]
    assert quant in ["ntile", "fraccov"]
    assert camera in allowed_cameras

    outpng = get_filename("wide_skycov", outdir, band=band, case=case, quant=quant)
    print(outpng)

    radius_deg = get_camera_props(camera)["radius_deg"]

    # AR KPNO
    # kpno = EarthLocation.of_site("Kitt Peak")
    # AR tiles for this program
    t_copy = t.copy()
    sel = np.array(["wide" in _ for _ in t["OBJECT"]])
    sel &= t["IN_IBIS"] == 1
    sel &= t["FILTER"] == band
    t = t[sel]
    passids = np.unique(t["PASS"])
    npass = len(passids)

    obs, done = get_tiles_obs_done(t, efftime_key="AR_EFFTIME_TOT")

    if case == "obs":
        iscase = obs
    if case == "done":
        iscase = done
    case_tileids = t["TILEID"][iscase]

    if quant == "ntile":
        cmap = get_quantz_cmap(matplotlib.cm.jet, npass + 1, 0, 1)
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmaplist[0] = ListedColormap(["lightgray"])(0)
        cmap = LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)
        cmin, cmax = -0.5, npass + 0.5
        clabel = "Covered by N tiles"
        cticks = np.arange(npass + 1, dtype=int)

    if quant == "fraccov":
        cmap = get_quantz_cmap(matplotlib.cm.jet, 11, 0, 1)
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmaplist[0] = ListedColormap(["lightgray"])(0)
        cmap = LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)
        cmin, cmax = 0, 1
        clabel = "Fraction of final coverage"
        cticks = np.arange(0, 1.1, 0.1)

    # first listing the area covered by 1, 2, ..., npass passes
    goalfn = get_filename("goal", outdir, survey="wide", band=band)
    d = fits.open(goalfn)[1].data  # fits.open is much faster than fitsio.read...
    goal_ns, goal_expfacs = d["NPASS"], d["EXPFAC_MEAN"]
    # healpix
    npix = hp.nside2npix(nside)
    pixarea = hp.nside2pixarea(nside, degrees=True)
    # AR number of tiles and summed expfacs
    ns = np.nan + np.zeros(len(d))
    ntiles = 0
    ns[goal_ns > 0] = 0.0
    expfac = 0
    for i in range(npass):
        sel = t["PASS"] == passids[i]
        sel &= np.in1d(t["TILEID"], case_tileids)
        if sel.sum() > 0:
            ipixs = tiles2pix(nside, tiles=t[sel], radius=radius_deg)
            ns[ipixs] += 1
            expfac += d["EXPFACS"][ipixs, i].sum()
        ntiles += sel.sum()
    # AR fractional coverage
    fracns = np.nan + np.zeros(len(d))
    sel = goal_ns > 0
    fracns[sel] = ns[sel] / goal_ns[sel]
    if quant == "ntile":
        cs = ns
    if quant == "fraccov":
        cs = fracns

    # AR start plotting
    title = (
        "IBIS Wide {}: {}/{}={:.0f}% {} tiles up to {} ({:.0f} deg2 completed)".format(
            band,
            ntiles,
            len(t),
            100.0 * ntiles / len(t),
            case,
            night,
            (fracns == 1).sum() * pixarea,
        )
    )
    print(title)
    t["RA"][t["RA"] > 300] -= 360
    d["RA"][d["RA"] > 300] -= 360
    fig, ax = plt.subplots(figsize=(15, 5), dpi=300)
    sel = ns >= 0
    sc = ax.scatter(
        d["RA"][sel],
        d["DEC"][sel],
        c=cs[sel],
        facecolors=ns,
        marker=".",
        s=0.2,
        linewidths=0,
        alpha=0.8,
        cmap=cmap,
        vmin=cmin,
        vmax=cmax,
        zorder=0,
    )
    ax.set_title(title)
    # AR hsc/wide
    for field, label in zip(["ngc", "sgc"], ["HSC/Wide", None]):
        plot_hsc_wide(ax, fields=[field], color="m", lw=0.5, zorder=1, label=label)
    # AR tiles completed
    lastnight = t["LASTNIGHT"][iscase].max()
    sel = (iscase) & (t["LASTNIGHT"] == lastnight)
    custom_plot_sky_circles(
        ax,
        t["RA"][sel],
        t["DEC"][sel],
        2 * radius_deg,
        ec="k",
        fc="none",
        lw=0.5,
    )
    ax.text(
        0.025,
        0.05,
        "{} tiles {} on {}".format(sel.sum(), case, lastnight),
        fontsize=10,
        transform=ax.transAxes,
    )

    # colorbar
    cbar = plt.colorbar(sc, ax=ax, fraction=0.025, orientation="vertical")
    cbar.set_label(clabel)
    cbar.set_ticks(cticks)

    ax.set_xlabel("R.A. [deg]")
    ax.set_ylabel("Dec. [deg]")
    ax.set_xlim(270, -60)
    ax.set_ylim(-20, 20)
    ax.grid()
    ax.set_axisbelow(True)
    ax.legend(loc=1)

    plt.savefig(outpng, bbox_inches="tight")
    plt.close()

    t = t_copy


# AR collapsible
def get_collapsible_names():
    collapsible_names = {}
    for survey in ["wide", "deep"]:
        collapsible_names[survey] = "collapsible_{}".format(survey)
        collapsible_names["sub{}".format(survey)] = "collapsible_sub{}".format(survey)
    return collapsible_names


def write_html(outdir, tiles, exps, ccds):

    bands = get_ibis_bands()

    htmlfn = get_filename("html", outdir)
    cssfn = get_filename("css", outdir)

    # AR need to copy the css?
    git_cssfn = str(files("desi2img") / "data" / os.path.basename(cssfn))
    if os.path.isfile(cssfn):
        f = open(cssfn, "r").read()
        git_f = open(git_cssfn, "r").read()
        if f != git_f:
            log.warning(
                "{} and {} are different; {} page may look not as expected".format(
                    cssfn, git_cssfn, htmlfn
                )
            )
    else:
        cmd = "cp -p {} {}".format(git_cssfn, cssfn)
        log.info("run {}".format(cmd))
        os.system(cmd)

    html = open(htmlfn, "w")

    # ADM set up the html file and write preamble to it.
    write_html_preamble(
        html,
        "IBIS Overview Page",
        "ibis-status.css",
    )

    # AR collapsibles
    collapsible_names = get_collapsible_names()

    surveys_fields = get_surveys_fields()
    surveys = list(surveys_fields.keys())

    for survey in surveys:

        fields = surveys_fields[survey]
        if survey == "wide":
            width = "45%"
            name = "coverage"
            quant0s = bands.copy()
            quant1s = list(fields.keys())
        else:
            width = "18%"
            name = "tiles"
            quant0s = list(fields.keys())
            quant1s = bands.copy()

        html.write(
            "<button type='button' class='{}'><strong>{} survey</strong></button>\n".format(
                collapsible_names[survey],
                survey.upper(),
            )
        )
        html.write("<div class='content'>\n")

        for quant0 in quant0s:

            # AR coverage/tiles
            html.write(
                "\t<button style='margin-left:25px;' typ='button' class='{}'><strong>{} {}</strong>: {}</button>\n".format(
                    collapsible_names["sub{}".format(survey)],
                    survey.upper(),
                    name,
                    quant0,
                )
            )
            html.write("\t<div class='content'>\n")

            html.write("\t\t<p>We display here TBD.</p>\n")
            html.write("\t\t<br>\n")

            if name == "coverage":
                outpng = path_full2web(
                    get_filename(
                        "wide_skycov",
                        outdir,
                        survey=survey,
                        band=quant0,
                        case="obs",
                        quant="fraccov",
                    )
                )
                txt = "<a href='{}'><img SRC='{}' width={} height=auto></a>".format(
                    outpng, outpng, "75%"
                )
                html.write("\t\t<td> {} </td>\n".format(txt))
            else:
                for quant1 in quant1s:
                    if quant0s[0] == bands[0]:
                        outpng = path_full2web(
                            get_filename(
                                "tiles",
                                outdir,
                                survey=survey,
                                band=quant0,
                                field=quant1,
                            )
                        )
                    else:
                        outpng = path_full2web(
                            get_filename(
                                "tiles",
                                outdir,
                                survey=survey,
                                band=quant1,
                                field=quant0,
                            )
                        )
                    txt = "<a href='{}'><img SRC='{}' width={} height=auto></a>".format(
                        outpng, outpng, width
                    )
                    html.write("\t\t<td> {} </td>\n".format(txt))

            html.write("\t\t<a&emsp;></a>\n")
            html.write("\t\t<tr>\n")
            html.write("\t\t</tr>" + "\n")
            html.write("\t\t<br>\n")
            html.write("\t\t</tr>\n")
            html.write("\t</div>\n")
            html.write("\n")

            # AR depths
            if survey == "deep":
                html.write(
                    "\t<button style='margin-left:25px;' typ='button' class='{}'><strong>{} depths</strong>: {}</button>\n".format(
                        collapsible_names["sub{}".format(survey)],
                        survey.upper(),
                        quant0,
                    )
                )
                html.write("\t<div class='content'>\n")

                html.write("\t\t<p>We display here TBD.</p>\n")
                html.write("\t\t<br>\n")
                for quant1 in quant1s:

                    if quant0s[0] == bands[0]:
                        outpng = path_full2web(
                            get_filename(
                                "depths",
                                outdir,
                                survey=survey,
                                band=quant0,
                                field=quant1,
                            )
                        )
                    else:
                        outpng = path_full2web(
                            get_filename(
                                "depths",
                                outdir,
                                survey=survey,
                                band=quant1,
                                field=quant0,
                            )
                        )
                    txt = "<a href='{}'><img SRC='{}' width={} height=auto></a>".format(
                        outpng, outpng, width
                    )
                    html.write("\t\t<td> {} </td>\n".format(txt))

                html.write("\t\t<a&emsp;></a>\n")
                html.write("\t\t<tr>\n")
                html.write("\t\t</tr>" + "\n")
                html.write("\t\t<br>\n")
                html.write("\t\t</tr>\n")
                html.write("\t</div>\n")
                html.write("\n")

        html.write("\t\t</tr>\n")
        html.write("\t</div>\n")
        html.write("\n")

    # AR Filters
    band_date = 20241112
    bands = np.array(["M411", "M438", "M464", "M490", "M517"])
    band_dir = os.path.join(
        os.getenv("DESI_ROOT"), "users", "raichoor", "laelbg", "ibis", "filt"
    )
    band_fns = {
        band: os.path.join(band_dir, "ibis-{}-{}.ecsv".format(band_date, band))
        for band in bands
    }

    html.write(
        "<button type='button' class='collapsible'><strong>Passbands</strong></button>\n"
    )
    html.write("<div class='content'>\n")

    if band_date == 20241112:
        html.write(
            "\t<p>We display here estimated total throughputs ({} version, courtesy of A. Dey) .</p>\n".format(
                band_date
            )
        )
    html.write("\t<br>\n")
    outpng = path_full2web(
        os.path.join(band_dir, "ibis-bands-{}.png".format(band_date))
    )
    print(outpng)
    txt = "<a href='{}'><img SRC='{}' width=50% height=auto></a>".format(outpng, outpng)
    html.write("\t<td> {} </td>\n".format(txt))
    html.write("\t<a&emsp;></a>\n")
    html.write("\t<tr>\n")
    html.write("\t</tr>" + "\n")
    html.write("\t<br>\n")
    html.write("\t</tr>\n")
    html.write("</div>\n")
    html.write("\n")

    # AR Per-night products:
    # AR expdepths, obsconds
    html.write(
        "<button type='button' class='collapsible'><strong>Per-night products</strong> ({} nights)</button>\n".format(
            np.unique(exps["NIGHT"]).size
        )
    )
    html.write("<div class='content'>\n")

    html.write("\t<p>\n")
    html.write("\t\tProduct:\n")
    html.write("\t\t<select id='product'>\n")
    for case in ["expdepth", "obsconds"]:
        html.write("\t\t\t<option value='{}'>{}</option>\n".format(case, case))
    html.write("\t\t</select>\n")
    html.write("\t\t&nbsp\n")

    my_nights = np.unique(exps["NIGHT"])
    # ccds_nights = np.unique(ccds["NIGHT"])
    # assert np.all(np.in1d(ccds_nights, my_nights))
    my_yyyys = np.array([str(night // 10000) for night in my_nights])
    my_mms = np.array(["{:02d}".format((night % 10000) // 100) for night in my_nights])
    my_dds = np.array(["{:02d}".format(night % 100) for night in my_nights])
    last_yyyy, last_mm, last_dd = my_yyyys[-1], my_mms[-1], my_dds[-1]

    html.write("\t\t<select name='yyyy' id='yyyy'>\n")
    html.write("\t\t\t<option value='' selected='selected'>yyyy</option>\n")
    # html.write("\t<option value='' selected='selected'>{}</option>\n".format(last_yyyy))
    html.write("\t\t</select>\n")
    html.write("\t\t<select name='mm' id='mm'>\n")
    html.write("\t\t\t<option value='' selected='selected'>mm</option>\n")
    # html.write("\t<option value='' selected='selected'>{}</option>\n".format(last_mm))
    html.write("\t\t</select>\n")
    html.write("\t\t<select name='dd' id='dd'>\n")
    html.write("\t\t\t<option value='' selected='selected'>dd</option>\n")
    # html.write("\t<option value='' selected='selected'>{}</option>\n")
    html.write("\t\t</select>\n")

    html.write("<script type='text/javascript'>\n")
    html.write("var yyyyObject = {\n")
    for yyyy in np.unique(my_yyyys):
        html.write("\t'{}': {{\n".format(yyyy))
        yyyy_sel = my_yyyys == yyyy
        for mm in np.unique(my_mms[yyyy_sel]):
            yyyymm_sel = (yyyy_sel) & (my_mms == mm)
            dds = my_dds[yyyymm_sel]
            dds = ["'{}'".format(_) for _ in dds]
            html.write("\t\t'{}': [{}],\n".format(mm, ", ".join(dds)))
        html.write("\t},\n")
    html.write("}\n")
    #
    html.write("window.onload = function() {\n")
    html.write("\tvar yyyySel = document.getElementById('yyyy');\n")
    html.write("\tvar mmSel = document.getElementById('mm');\n")
    html.write("\tvar ddSel = document.getElementById('dd');\n")
    html.write("\tfor (var x in yyyyObject) {\n")
    html.write("\t\tyyyySel.options[yyyySel.options.length] = new Option(x, x);\n")
    html.write("\t}\n")
    html.write("\tyyyySel.onchange = function() {\n")
    html.write("\tddSel.length = 1;\n")
    html.write("\tmmSel.length = 1;\n")
    html.write("\tvar mm = Object.keys(yyyyObject[yyyySel.value]);\n")
    html.write("\tmm.sort();\n")
    html.write("\tfor (var i = 0; i < mm.length; i++) {\n")
    html.write("\tmmSel.options[mmSel.options.length] = new Option(mm[i], mm[i]);\n")
    html.write("\t\t}\n")
    html.write("\t}\n")
    html.write("\tmmSel.onchange = function() {\n")
    html.write("\tddSel.length = 1;\n")
    html.write("\tvar z = yyyyObject[yyyySel.value][this.value];\n")
    html.write("\tfor (var i = 0; i < z.length; i++) {\n")
    html.write("\t\tddSel.options[ddSel.options.length] = new Option(z[i], z[i]);\n")
    html.write("\t\t}\n")
    html.write("\t}\n")
    html.write("}\n")
    html.write("</script>\n")

    html.write("&nbsp\n")
    html.write(
        "<button onclick='getnight()' id='myButton' class='btn request-callback' > Go! </button>\n"
    )
    html.write("</p>\n")
    html.write("<p id='link'></p>\n")
    html.write("<p> <span class='output'></span> </p>\n")
    html.write("<script type='text/javascript'>\n")
    html.write("function getnight() {\n")
    html.write("\tvar x;\n")
    html.write("\tvar yyyy;\n")
    html.write("\tvar mm;\n")
    html.write("\tvar dd;\n")
    html.write("\tvar ext;\n")
    html.write("\tproduct = document.querySelector('#product').value;\n")
    html.write("\tyyyy = document.querySelector('#yyyy').value;\n")
    html.write("\tmm = document.querySelector('#mm').value;\n")
    html.write("\tdd = document.querySelector('#dd').value;\n")
    html.write("\tif (product == 'expdepth'){\n")
    html.write("\t ext = 'pdf';\n")
    html.write("\t}else if (product == 'obsconds'){\n")
    html.write("\t ext = 'png';\n")
    html.write("\t} else {\n")
    html.write("\text = 'html';\n")
    html.write("\t}\n")
    html.write(
        "\tx = '{}/' + product + '/ibis-' + product + '-' + yyyy + mm + dd + '.' + ext\n".format(
            path_full2web(outdir)
        )
    )
    html.write("document.getElementById('link').innerHTML = 'Opening ' + x;\n")
    html.write("\twindow.open(x);\n")
    html.write("}\n")
    html.write("</script>\n")

    html.write("\t</tr>\n")
    html.write("</div>\n")
    html.write("\n")

    # AR lines to make collapsing sections
    for collapsible in [
        "collapsible",
        "collapsible_sub",
        "collapsible_year",
        "collapsible_month",
    ] + [collapsible_names[survey] for survey in collapsible_names]:
        write_html_collapse_script(html, collapsible)

    # ADM html postamble for main page.
    write_html_today(html)
    html.write("</html></body>\n")
    html.close()


# def check_staged_files_one_night(allexpsfn, night):
def check_staged_files_one_night(night):

    # d = Table.read(allexpsfn)
    # d = d[d["NIGHT"] == night]
    fn = "/global/cfs/cdirs/desi/users/raichoor/ibis-observing/logs/db-{}-{}-{}.ecsv".format(
        str(night)[:4], str(night)[4:6], str(night)[6:8]
    )
    d = Table.read(fn)
    for key in d.colnames:
        d[key].name = key.upper()
    bands = get_ibis_bands()
    sel = np.in1d(d["BAND"], bands)
    sel &= d["EFFTIME"] > 0
    d = d[sel]
    d["NIGHT"] = night
    fns = sorted(
        glob(
            "/global/cfs/cdirs/cosmo/staging/decam/DECam_CP-IBIS/CPIBIS{}*/*ooi*".format(
                night
            )
        )
    )
    staged_expnums = [fitsio.read_header(fn, 0)["EXPNUM"] for fn in fns]
    sel = ~np.in1d(d["EXPNUM"], staged_expnums)
    if sel.sum() == 0:
        return None
    else:
        return d[sel]


# def check_staged_files(allexpsfn, numproc):
def check_staged_files(numproc):

    # d = Table.read(allexpsfn)
    fns = sorted(
        glob(
            "/global/cfs/cdirs/desi/users/raichoor/ibis-observing/logs/db-20??-??-??.ecsv"
        )
    )
    d = vstack([Table.read(fn) for fn in fns])
    for key in d.colnames:
        d[key].name = key.upper()
    bands = get_ibis_bands()
    sel = np.in1d(d["BAND"], bands)
    sel &= d["EFFTIME"] > 0
    d = d[sel]
    d["NIGHT"] = get_nightobs(d["MJD_OBS"], "decam")
    print(d.colnames)
    nights = np.unique(d["NIGHT"])
    print(
        "check exposures for {} night: {}".format(
            nights.size, ",".join(nights.astype(str))
        )
    )
    # myargs = [(allexpsfn, night) for night in nights]
    pool = multiprocessing.Pool(numproc)
    with pool:
        # ds = pool.starmap(check_staged_files_one_night, myargs)
        ds = pool.map(check_staged_files_one_night, nights)
    ds = [d for d in ds if d is not None]
    miss_d = vstack(ds)
    print("")
    if len(miss_d) == 0:
        print("all good, no missing exposure!")
    else:
        print("found {} missing exposures:".format(len(miss_d)))
        print(
            miss_d[
                "NIGHT", "EXPNUM", "BAND", "OBJECT", "TILEID", "EXPTIME", "EFFTIME"
            ].pprint_all()
        )
    print("")
