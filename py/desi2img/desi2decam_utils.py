#!/usr/bin/env python

import os
import shutil
from copy import deepcopy
from time import time
import yaml
import fitsio
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, position_angle
from astropy import units as u
from astropy.table import Table, vstack
import numpy as np
import healpy as hp
from scipy.spatial import Delaunay
from desimodel.footprint import tiles2pix
from desiutil.log import get_logger
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator
import matplotlib
from matplotlib.path import Path
from matplotlib.patches import Polygon

# from desispec.tile_qa_plot import get_quantz_cmap
import multiprocessing

log = get_logger()
nest = True


def get_quantz_cmap(name, n, cmin=0, cmax=1):
    """
    Creates a quantized colormap.


    Args:
        name: matplotlib colormap name (e.g. "tab20") (string)
        n: number of colors
        cmin (optional, defaults to 0): first color of the original colormap to use (between 0 and 1) (float)
        cmax (optional, defaults to 1): last color of the original colormap to use (between 0 and 1) (float)


    Returns:
        A matplotlib cmap object.


    Notes:
        https://matplotlib.org/examples/api/colorbar_only.html
    """
    cmaporig = matplotlib.cm.get_cmap(name)
    mycol = cmaporig(np.linspace(cmin, cmax, n))
    cmap = matplotlib.colors.ListedColormap(mycol)
    cmap.set_under(mycol[0])
    cmap.set_over(mycol[-1])
    return cmap


def get_decam_ccdnames():
    NSs = np.array(["S" for x in range(1, 32)] + ["N" for x in range(1, 32)])
    nums = np.array([x for x in range(1, 32)] + [x for x in range(1, 32)])
    names = np.array(["{}{}".format(NS, num) for NS, num in zip(NSs, nums)])
    return names


# https://noirlab.edu/science/sites/default/files/media/archives/documents/scidoc0436.pdf
# taking value at center..
def get_decam_pixscale():
    return 0.2637


# approx. conservative (i.e. larger than real)
def get_decam_radius():
    return 1.1


def read_yaml(fn):
    with open(fn, "r") as file:
        config = yaml.safe_load(file)
    # quantities added along the way..
    for key, default in [
        ("rnside", 64),
        ("indesi", True),
        ("inflate_ra_factor", 1),
        ("anneal_nccd_min", 0),
        ("anneal_tedge_freeze", False),
        ("anneal_r_distrib", "uniform"),
    ]:
        if key not in config:
            config[key] = default

    return config


# ra,dec,airmass~0,0,1.2 (dec~0 is the "center" of the desi2 footprint)
def get_ref_fn():
    # return "/global/cfs/cdirs/cosmo/staging/decam/DECam_CP-DR11/CP20240101/c4d_240102_045550_ooi_i_v1.fits.fz" # dec~0, airmass~1.2
    return "/global/cfs/cdirs/cosmo/staging/decam/DECam_CP-DR11/CP20121019/c4d_121020_014450_ooi_g_ls11.fits.fz"


def plot_decam_radec_ccds(ax, ccds, ra_wrap_center=0, print_ccd_names=True):

    dowrap = False
    for name in ccds:
        if ccds[name]["ras"].max() - ccds[name]["ras"].min() > 180:
            dowrap = True

    for name in ccds:

        ras, decs = ccds[name]["ras"].copy(), ccds[name]["decs"].copy()
        if dowrap:
            ras[ras < ra_wrap_center - 180] += 360
            ras[ras > ra_wrap_center + 180] -= 360

        if print_ccd_names:
            ax.text(
                0.5 * (ras.max() + ras.min()),
                0.5 * (decs.max() + decs.min()),
                name,
                color="k",
                ha="center",
                va="center",
            )
        ax.plot(ras, decs, color="k", lw=0.5)
        # pp = Rectangle((ras.min(), decs.min()), ras.max() - ras.min(), decs.max() - decs.min())
        pp = Polygon(np.array([ras, decs]).T)#, True)
        pp.set_color("orange")
        pp.set_alpha(0.8)
        ax.add_artist(pp)


def get_ref_hdrs(ccd_names, ref_fn=None):

    if ref_fn is None:
        ref_fn = get_ref_fn()

    h = fits.open(ref_fn)
    exts = np.array([h[i].header["EXTNAME"] for i in range(1, len(h))])
    assert np.all(np.in1d(ccd_names, exts))
    # use the CRVAL{1,2} of S1; the CRVAL{1,2} values are the same
    #   for all CCDs, it is just the image center
    # not using CENT{RA,DEC} from the 0-extension, because it is rounded
    # ref_ra, ref_dec = h[1].header["CENTRA"], h[0].header["CENTDEC"]
    ref_ra, ref_dec = h["S1"].header["CRVAL1"], h["S1"].header["CRVAL2"]
    ref_hdrs = {ccd_name: h[ccd_name].header for ccd_name in ccd_names}

    return ref_ra, ref_dec, ref_hdrs

# "default" values are:
# - npix_msk_*=0 (ie take the whole ccds)
def get_ref_radecs(
    names,
    npix_msk_xstart,
    npix_msk_xend,
    npix_msk_ystart,
    npix_msk_yend,
    ref_fn=None,
):

    if ref_fn is None:
        ref_fn = get_ref_fn()

    h = fits.open(ref_fn)
    # CRVAL1,CRVAL2: image center for all ccds
    ref_ra, ref_dec = None, None
    ref_radecs = {}
    for name in names:
        hdr = h[name].header
        if ref_ra is None:
            ref_ra, ref_dec = hdr["CRVAL1"], hdr["CRVAL2"]
            ref_c = SkyCoord(ref_ra * u.degree, ref_dec * u.degree, frame="icrs")
        x0, x1 = npix_msk_xstart, hdr["NAXIS1"] - 1 - npix_msk_xend
        y0, y1 = npix_msk_xend, hdr["NAXIS2"] - 1 - npix_msk_yend
        xs = np.array([x0, x0, x1, x1, x0])
        ys = np.array([y0, y1, y1, y0, y0])
        wcs = WCS(hdr)
        ras, decs = wcs.all_pix2world(xs, ys, 0)
        ref_radecs[name] = {"ras": ras, "decs": decs}
    return ref_ra, ref_dec, ref_radecs


# "default" values are:
# - inflate_ra_factor=1 (ie no "inflation")
# note: we perform "inflate" here, not in get_ref_radecs(), on purpose
def get_tile_ccds_radecs(tilera, tiledec, ref_radecs, ref_tilera, ref_tiledec, inflate_ra_factor):

    # print("get_tile_ccds_radecs(): tilera, tiledec, ref_tilera, ref_tiledec, inflate_ra_factor = ",  tilera, tiledec, ref_tilera, ref_tiledec, inflate_ra_factor)

    # required offsets
    tilec = SkyCoord(tilera * u.degree, tiledec * u.degree, frame="icrs")
    ref_tilec = SkyCoord(
        ref_tilera * u.degree, ref_tiledec * u.degree, frame="icrs"
    )

    ccds_radecs = {}
    for name in ref_radecs:
        ras, decs = ref_radecs[name]["ras"], ref_radecs[name]["decs"]
        cs = SkyCoord(ras * u.degree, decs * u.degree, frame="icrs")
        pas = position_angle(
            np.radians(ref_tilera),
            np.radians(ref_tiledec),
            np.radians(ras),
            np.radians(decs),
        )
        seps = cs.separation(ref_tilec)
        tilecs = tilec.directional_offset_by(pas, seps)
        tileras, tiledecs = tilecs.ra.value, tilecs.dec.value

        # inflate?
        if inflate_ra_factor != 1:
            # take the distance in ra, between -180 and 180
            raseps = tileras - tilera
            raseps[raseps < -180] += 360
            raseps[raseps > 180] -= 360
            # inflate that distance by inflate_ra_factor
            tileras = tilera + inflate_ra_factor * raseps
            # wrap in 0 and 360
            tileras[tileras < 0] += 360
            tileras[tileras > 360] -= 360

        ccds_radecs[name] = {"ras": tileras, "decs": tiledecs}
    return ccds_radecs


# to get vertices for hp.query_polygon
def radec2vec(ras, decs):
    tmpras = ras % 360.
    thetas, phis = np.radians(90 - decs), np.radians(tmpras)
    vecs = hp.ang2vec(thetas, phis)
    return vecs
    
"""
def get_tiles_pixs(tileras, tiledecs, nside, trad=None):
    if trad is None:
        trad = get_decam_radius()
    tiles = Table()
    tiles["RA"], tiles["DEC"] = np.atleast_1d(tileras), np.atleast_1d(tiledecs)
    tpixs = tiles2pix(nside, tiles=tiles, radius=trad)
    return tpixs
"""

# ra, dec: center of the "ellipse"
# ra_radius: (projected) radius along ra, in deg
#   typically: trad * inflate_ra_factor (+ pixel_size)
# dec_radius: (projected) radius along dec, in deg
#   typically: trad (+ pixel_size)
# closed: True or False
# npt: nb of points
#
# outputs: ras, decs [0<ras<360]
def get_radec_ellipse(ra, dec, ra_radius, dec_radius, closed, npt=100):
    #
    assert closed in [True, False]
    #
    angs = np.linspace(0, 2 * np.pi, npt)
    if not closed:
        angs = angs[:-1]
    #
    decs = dec + dec_radius * np.sin(angs)
    ras = ra + ra_radius / np.cos(np.radians(decs)) * np.cos(angs)
    ras = np.array(ras) % 360

    dra = ras.max() - ras.min()
    if dra > 360:
        msg = "ill-defined problem, ell_ras.max() - ell_ras.min() = {:2f} > 360 deg; exit".format(dra)
        log.info(msg)
        raise ValueError(msg)

    return ras, decs


# c_ra, c_dec: center of the "ellipse"
# ras, decs: points we want to test
# ra_radius: (projected) radius along ra, in deg
#   typically: trad * inflate_ra_factor (+ pixel_size)
# dec_radius: (projected) radius along dec, in deg
#   typically: trad (+ pixel_size)
# closed (True/False): close the ellipse or not
# pix_radius_deg: if provided, hp pixel size, to add a safe margin,
#   which is not multiplied
def get_isin_radec_ellipse(c_ra, c_dec, ras, decs, ra_radius, dec_radius, closed, npt=100):

    assert closed in [True, False]

    # ellipse centered on ra=0
    ell_ras, ell_decs = get_radec_ellipse(0., c_dec, ra_radius, dec_radius, closed, npt=npt)
    ell_ras[ell_ras < -180] += 360
    ell_ras[ell_ras > 180] -= 360
    ell_p = Path([(ell_ra, ell_dec) for ell_ra, ell_dec in zip(ell_ras, ell_decs)])

    # center ras, decs on ra_c, dec_c
    tmpras, tmpdecs = np.atleast_1d(ras).copy(), np.atleast_1d(decs).copy()
    tmpras -= c_ra
    # clip in [-180, 180]
    tmpras[tmpras < -180] += 360
    tmpras[tmpras > 180] -= 360
    # test
    radecs = np.array([tmpras, tmpdecs]).T
    sel = ell_p.contains_points(radecs)

    return sel


# fact: see desimodel.footprint.tiles2pix
# 20240525 : added "if inflate_ra_factor == 1."
#   [backwards compatible for inflate_ra_factor != 1]
def get_tiles_pixs(tileras, tiledecs, nside, inflate_ra_factor, trad=None):

    # hp stuff
    nest = True
    fact = 2 ** 7
    inclusive = True

    tras, tdecs = np.atleast_1d(tileras), np.atleast_1d(tiledecs)
    ntile = len(tras)

    if trad is None:
        trad = get_decam_radius()

    # no inflate case, simple
    if inflate_ra_factor == 1.:
        tiles = Table()
        tiles["RA"], tiles["DEC"] = np.atleast_1d(tileras), np.atleast_1d(tiledecs)
        pixs = tiles2pix(nside, tiles=tiles, radius=trad)

    # inflate case, more complex..
    else:
        # all pixels
        npix = hp.nside2npix(nside)
        pixs = np.arange(npix, dtype=int)
        thetas, phis = hp.pix2ang(nside, pixs, nest=nest)
        ras, decs = np.degrees(phis), 90.0 - np.degrees(thetas)

        # safe: extend the radius by one half of pixel size
        pix_radius_deg = hp.nside2resol(nside, arcmin=True) / 60.

        closed = True
        sel = np.zeros(npix, dtype=bool)
        for i in range(ntile):
            #
            seli = get_isin_radec_ellipse(
                tras[i], tdecs[i],
                ras, decs,
                trad * inflate_ra_factor + pix_radius_deg / 2,
                trad + pix_radius_deg / 2,
                closed, npt=100,
            )
            pixis = pixs[seli]
            sel[pixis] = True

        pixs = pixs[sel]

    return pixs


def get_nccds(radecs, ccds_radecs):

    nccds = np.zeros(len(radecs), dtype=int)

    for name in ccds_radecs:

        ccd_ras, ccd_decs = ccds_radecs[name]["ras"].copy(), ccds_radecs[name]["decs"].copy()

        # handle ccds overlapping ra=0 line
        tmpradecs = radecs.copy()
        if ccd_ras.max() - ccd_ras.min() > 180.:
            ccd_ras[ccd_ras > 180] -= 360
            sel = tmpradecs[:, 0] > 180
            tmpradecs[sel, 0] -= 360

        p = Path([(ra, dec) for ra, dec in zip(ccd_ras, ccd_decs)])
        sel = p.contains_points(tmpradecs)
        nccds[sel] += 1

    return nccds


def get_tile_nccds(
    tilera, tiledec, ref_radecs, ref_tilera, ref_tiledec, ras, decs, pixs, nside, inflate_ra_factor,
):

    #print("get_tile_nccds(): tilera, tiledec, ref_tilera, ref_tiledec, inflate_ra_factor: ",tilera, tiledec, ref_tilera, ref_tiledec, inflate_ra_factor) 

    decam_radius = get_decam_radius()

    # ccds corners
    ccds_radecs = get_tile_ccds_radecs(
        tilera, tiledec, ref_radecs, ref_tilera, ref_tiledec, inflate_ra_factor,
    )

    # pixels overlapping the tile
    tpixs = get_tiles_pixs(tilera, tiledec, nside, inflate_ra_factor, trad=decam_radius)
    #tiles = Table()
    #tiles["RA"], tiles["DEC"] = [tilera], [tiledec]
    #tpixs = tiles2pix(nside, tiles=tiles, radius=decam_radius)

    # points inside those pixels
    ii = np.where(np.in1d(pixs, tpixs))[0]
    # print("get_tile_nccds(): {:.2f}\t{:.2f}\t{}\t{:0f} deg".format(tilera, tiledec, inflate_ra_factor, ii.size/1000))

    radecs = np.array([ras, decs]).T
    nccds = np.zeros(radecs.shape[0], dtype=int)
    nccds[ii] = get_nccds(radecs[ii], ccds_radecs)

    return nccds


# get the nccds values for a given rands_fn file
#   for a set of tiles
def get_rands_fn_tiles_nccds(
    rands_fn, tileras, tiledecs, ref_radecs, ref_tilera, ref_tiledec, inflate_ra_factor,
):

    # print("get_rands_fn_tiles_nccds(): rands_fn, tileras.size, inflate_ra_factor: ", rands_fn, tileras.size, inflate_ra_factor)

    d = fitsio.read(rands_fn)
    nside = fitsio.read_header(rands_fn, 1)["HPXNSIDE"]
    nccds = np.zeros(len(d), dtype=int)
    ii = []
    for i, (tilera, tiledec) in enumerate(zip(tileras, tiledecs)):
        nccds_i = get_tile_nccds(
            tilera,
            tiledec,
            ref_radecs,
            ref_tilera,
            ref_tiledec,
            d["RA"],
            d["DEC"],
            d["HPXPIXEL"],
            nside,
            inflate_ra_factor,
        )
        if nccds_i.max() > 0:
            ii.append(i)
        nccds += nccds_i
    # log.info("{}\t{}\t{}\t{}".format(os.path.basename(rands_fn), len(d), len(tileras), nccds.max()))
    return (nccds, ii)


def _write_fn(fn, d, overwrite):
    # log.info(os.path.basename(fn))
    d.write(fn, overwrite=overwrite)


# not-straightforward procedure, as we want
# to have thing in a nice state if the process
# "dies" during this function
#   (e.g. when an interactive node session timeouts)
def update_nccds_fns(fns, all_nccds, rmv_unobs_rands=False):

    #
    tmpdirs = np.unique([os.path.dirname(fn) for fn in fns])
    assert len(tmpdirs) == 1
    mydir = tmpdirs[0]

    if rmv_unobs_rands:
        log.info("rmv_unobs_rands=True\t-> will remove rands with NCCD=0")

    oldfns = [
        os.path.join(mydir, os.path.basename(fn).replace(".fits", "-old.fits"))
        for fn in fns
    ]
    newfns = [
        os.path.join(mydir, os.path.basename(fn).replace(".fits", "-new.fits"))
        for fn in fns
    ]

    # make a copy of each original file (preserving timestamp)
    log.info("copy {} files from {}/*.fits to {}/*-old.fits".format(len(fns), mydir, mydir))
    for fn, oldfn in zip(fns, oldfns):
        # log.info("copy {} to {}".format(fn, oldfn))
        shutil.copy2(fn, oldfn)

    # update each file, and write a new fn
    # there shouldn t be an existing one, so no overwrite=True argument
    log.info("create {} updated files {}/*-new.fits".format(len(fns), mydir))
    for fn, newfn, nccds in zip(fns, newfns, all_nccds):
        d = Table.read(fn)
        assert len(nccds) == len(d)
        d["NCCD"] = nccds
        if rmv_unobs_rands:
            sel = d["NCCD"] > 0
            d = d[sel]
        d.write(newfn, overwrite=True)
        # log.info("wrote new updated file {}".format(newfn))

    # now play around with files...
    log.info("move {} files from {}/*-new.fits to {}/*.fits".format(len(fns), mydir, mydir))
    for fn, newfn in zip(fns, newfns):
        # log.info("move {} to {}".format(newfn, fn))
        shutil.move(newfn, fn)
    log.info("delete {} files {}/*-old.fits".format(len(fns), mydir))
    for oldfn in oldfns:
        # log.info("delete {}".format(oldfn))
        os.remove(oldfn)

    return True


def get_rands_fns_from_pixs(outdir, pixs):

    rands_fns = np.array(
        [
            os.path.join(outdir, "rands", "rands-hp-{}.fits".format(pix))
            for pix in pixs
        ]
    )
    return rands_fns


def get_pixs_from_rands_fns(rands_fns):

    pixs = np.array(
        [
            int(os.path.basename(fn).replace(".", "-").split("-")[-2]) for fn in rands_fns
        ]
    )
    return pixs

def create_rands(
    outdir,
    dens,
    ramin,
    ramax,
    decmin,
    decmax,
    ccd_names,
    nside,
    numproc,
    np_rand_seed=1234,
    overwrite=False,
):

    np.random.seed(np_rand_seed)

    rands_dir = os.path.dirname(get_rands_fns_from_pixs(outdir, [0])[0])
    if not overwrite:
        os.makedirs(rands_dir, exist_ok=True)

    d = Table()
    d.meta["RNDSEED"] = np_rand_seed
    d.meta["DENSITY"] = dens
    d.meta["RAMIN"], d.meta["RAMAX"] = ramin, ramax
    d.meta["DECMIN"], d.meta["DECMAX"] = decmin, decmax
    d.meta["HPXNSIDE"], d.meta["HPXNEST"] = nside, nest
    d.meta["CCDNAMES"] = ",".join(ccd_names)
    ## first estimating the area for ramin < ra < ramax, -90 < dec < 90
    ste2sqdeg = 360.0**2 / np.pi / (4 * np.pi)
    area = 4 * np.pi * (ramax - ramin) / 360.0 * ste2sqdeg
    ## number of randoms
    nrand = np.int64(dens * area)
    # creating randoms
    ras = np.random.uniform(low=ramin, high=ramax, size=nrand)
    decs = np.degrees(
        np.arcsin(1.0 - np.random.uniform(low=0, high=1, size=nrand) * 2.0)
    )
    # cutting on decmin < dec < decmax
    sel = (decs >= decmin) & (decs <= decmax)
    d["RA"], d["DEC"] = ras[sel], decs[sel]
    log.info(
        "{} < ra < {} , {} < dec < {} -> area = {:.1f} deg2".format(
            ramin, ramax, decmin, decmax, len(d) / dens
        )
    )
    log.info("nrand = {}".format(len(d)))

    d["HPXPIXEL"] = hp.ang2pix(
        nside, np.radians(90.0 - d["DEC"]), np.radians(d["RA"]), nest=nest
    )
    d["NCCD"] = 0
    unq_pixs = np.unique(d["HPXPIXEL"])
    log.info("unq_pixs.size = {}".format(unq_pixs.size))

    myargs = [
        (
            get_rands_fns_from_pixs(outdir, [pix])[0],
            d[d["HPXPIXEL"] == pix],
            overwrite,
        )
        for pix in unq_pixs
    ]
    pool = multiprocessing.Pool(processes=numproc)
    with pool:
        _ = pool.starmap(_write_fn, myargs)
    pool.close()


def compute_nccds(rands_fns, t, config, numproc, trad=None):

    if trad is None:

        trad =  get_decam_radius()

    t["TMPTILEID"] = np.arange(len(t), dtype=int)

    hdr = Table.read(rands_fns[0]).meta
    nside = hdr["HPXNSIDE"]

    ccd_names = hdr["CCDNAMES"].split(",")
    # log.info("ccd_names : {}".format(",".join(ccd_names)))
    ref_tilera, ref_tiledec, ref_radecs = get_ref_radecs(
        ccd_names,
        config["npix_msk_xstart"],
        config["npix_msk_xend"],
        config["npix_msk_ystart"],
        config["npix_msk_yend"],
    )

    # deal with each rands-hp-*.fits file individually
    pixs = get_pixs_from_rands_fns(rands_fns)

    # for each pixel, list what tiles do overlap
    pixits = {pix: [] for pix in pixs}
    for i in range(len(t)):
        tpixs_i = get_tiles_pixs(t["RA"][i], t["DEC"][i], nside, config["inflate_ra_factor"], trad=trad)
        # print(i, t["TILEID"][i], t["RA"][i], t["DEC"][i], "{:.2f} deg2".format(tpixs_i.size*hp.nside2pixarea(nside,degrees=True)))
        # handle case where we ve discarded a pixel because
        # it was not containing any rands with NCCD>0
        sel = np.in1d(tpixs_i, pixs)
        tpixs_i = tpixs_i[sel]
        #tiles_i = Table()
        #tiles_i["RA"], tiles_i["DEC"] = [t["RA"][i]], [t["DEC"][i]]
        #pixs_i = tiles2pix(nside, tiles=tiles_i, radius=decam_radius)
        for tpix in tpixs_i:
            pixits[tpix].append(i)

    # for each pixel, create a tiles file
    pixts = {}
    for pix in pixs:
        pixts[pix] = t[pixits[pix]]
        # print("pix = {}\t-> {} tiles".format(pix, len(pixits[pix])))

    # compute nccds for each pixel
    myargs = []
    for fn, pix in zip(rands_fns, pixs):
        pixt = pixts[pix]
        tileras, tiledecs = pixt["RA"], pixt["DEC"]
        myargs.append((fn, tileras, tiledecs, ref_radecs, ref_tilera, ref_tiledec, config["inflate_ra_factor"]))
    pool = multiprocessing.Pool(processes=numproc)
    with pool:
        all_outputs = pool.starmap(get_rands_fn_tiles_nccds, myargs)
    pool.close()
    all_nccds = np.array([_[0] for _ in all_outputs], dtype=object)
    all_iis = [_[1] for _ in all_outputs]

    # get the list of tiles touching the randoms
    tids = []
    for (pix, ii) in zip(pixs, all_iis):
        tids += pixts[pix]["TMPTILEID"][ii].tolist()
    tids = np.unique(tids)
    tsel = np.in1d(t["TMPTILEID"], tids)

    t.remove_column("TMPTILEID")

    return all_nccds, tsel


def get_metric(metric_num, quants):

    metric = None

    # DJS email 04/10/24
    if metric_num == 0:

        metric_str = "mean(nccd) / std(nccd)"
        if quants is not None:
            nccds = quants["nccds"]
            if "nccd_min" in quants:
                nccd_min = quants["nccd_min"]
            else:
                nccd_min = 0
            sel = nccds >= nccd_min
            metric = nccds[sel].mean() / nccds[sel].std()

    # AR 04/13/24
    if metric_num == 1:

        metric_str = "mean(nccd) / std(nccd) / (1 + |new_area - ref_area| / ref_area)"
        if quants is not None:
            nccds = quants["nccds"]
            delta_area = quants["delta_area"]  # np.abs(new_area - ref_area) / ref_area
            if "nccd_min" in quants:
                nccd_min = quants["nccd_min"]
            else:
                nccd_min = 0
            sel = nccds >= nccd_min
            metric = nccds[sel].mean() / nccds[sel].std() / (1 + delta_area)

    # DJS email 04/14/24
    if metric_num == 2:

        metric_str = "mean(nccd) / (std(nccd)) ** 2"
        if quants is not None:
            nccds = quants["nccds"]
            if "nccd_min" in quants:
                nccd_min = quants["nccd_min"]
            else:
                nccd_min = 0
            sel = nccds >= nccd_min
            metric = nccds[sel].mean() / nccds[sel].std() ** 2

    # DJS email 04/16/24
    if metric_num == 3:
        
        metric_str = "mean(nccd) * (1 + 1 / std(nccd))"
        if quants is not None:
            nccds = quants["nccds"]
            if "nccd_min" in quants:
                nccd_min = quants["nccd_min"]
            else:
                nccd_min = 0
            sel = nccds >= nccd_min
            metric = nccds[sel].mean() * (1. + 1. / nccds[sel].std())

    return metric, metric_str


def get_init_tiles(config):
    tilesfn = config["tilesfn"]
    ramin, ramax, decmin, decmax = (
        config["ramin"],
        config["ramax"],
        config["decmin"],
        config["decmax"],
    )

    inflate_ra_factor = config["inflate_ra_factor"]
    anneal_rmax = config["anneal_rmax"]
    anneal_tedge_freeze = config["anneal_tedge_freeze"]

    decam_radius = get_decam_radius()

    t = Table.read(tilesfn)
    if "TILEID" not in t.colnames:
        t["TILEID"] = np.arange(len(t), dtype=int)

    sel = np.ones(len(t), dtype=bool)

    if config["indesi"]:
        sel = t["IN_DESI"] == 1

    if (ramin == 0.) & (ramax == 360.):
        log.info("(ramin,ramax) = (0,360) -> ignore anneal_rmax for ra")
    # TODO: handle ra=0  boundary...
    else:
        sel &= t["RA"] - ramin > (decam_radius + inflate_ra_factor * anneal_rmax) / np.cos(np.radians(t["DEC"]))
        sel &= t["RA"] - ramax < -(decam_radius + inflate_ra_factor * anneal_rmax) / np.cos(
            np.radians(t["DEC"])
        )
    if (decmin == -90.) & (decmax == 90):
        log.info("(decmin,decmax) = (-90, 90) -> ignore anneal_rmax for dec")
    else:
        sel &= t["DEC"] - decmin > decam_radius + anneal_rmax
        sel &= t["DEC"] - decmax < -(decam_radius + anneal_rmax)

    log.info("select {} / {} tiles".format(sel.sum(), len(t)))

    t = t[sel]

    if "FREEZE" not in t.colnames:
        t["FREEZE"] = False
    if anneal_tedge_freeze:

        radecs = np.array([t["RA"], t["DEC"]]).T
        edges = alpha_shape(radecs, alpha=2, only_outer=True)
        ii = []
        for i, j in edges:
            ii += [i, j]
        ii = np.unique(ii)
        t["FREEZE"][ii] = True
    log.info("{} tiles have FREEZE=True".format(t["FREEZE"].sum()))

    return t


def get_anneal_allowtpixs(t, nside, inflate_ra_factor, trad=None):

    # trad : degrees
    if trad is None:
        trad = get_decam_radius()

    npix = hp.nside2npix(nside)

    # trad = -99 -> no constraint, take full sky
    if trad == -99.:

        pixs = np.arange(npix, dtype=int)
        return pixs

    else:

        resol = hp.nside2resol(nside, arcmin=True) / 60.0
        nlayer = int(np.ceil(trad * inflate_ra_factor/ resol))

        thetas, phis = hp.pix2ang(nside, np.arange(npix, dtype=int), nest=nest)
        ras, decs = np.degrees(phis), 90.0 - np.degrees(thetas)
        cs = SkyCoord(ras * u.degree, decs * u.degree, frame="icrs")

        tcs = SkyCoord(t["RA"] * u.degree, t["DEC"] * u.degree, frame="icrs")

        # pixels overlapping the tiles
        # pixs = tiles2pix(nside, tiles=t, radius=trad)
        pixs = get_tiles_pixs(t["RA"], t["DEC"], nside, inflate_ra_factor, trad=None)

        # "enlarge" the boundaries far enough to include
        for i in range(nlayer):
            pixs = np.unique(
                np.vstack([hp.get_all_neighbours(nside, pix, nest=nest) for pix in pixs])
            )
        # cs = cs[pixs]

        # TODO: not sure what I was doing here...
        #       not using that for now, so just disable
        #       at worst, it will just allow a larger footprint,
        #       less efficient but not conceptually problematic
        """
        sel = np.zeros(len(pixs), dtype=bool)
        for i in range(len(t)):
            # print("{}\t/ {}".format(i, len(t)-1))
            seli = cs.separation(tcs[i]).degree < trad
            sel[seli] = True
        """
        sel = np.ones(len(pixs), dtype=bool)

        return pixs[sel]


# https://stackoverflow.com/questions/50549128/boundary-enclosing-a-given-set-of-points
# AR: alpha=2 seems to work fine
def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"
    #
    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))
    #
    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges


def init_anneal_table(n):
    d = Table()
    for key in [
        "TIME",
        "NTRY",
        "TILEID",
        "OLD_RA",
        "OLD_DEC",
        "OLD_AREA",
        "OLD_METRIC",
        "PA_DEG",
        "SEP_DEG",
        "NEW_RA",
        "NEW_DEC",
        "NEW_AREA",
        "NEW_METRIC",
    ]:
        if key in ["NTRY", "TILEID"]:
            d[key] = np.zeros(n, dtype=int)
        else:
            d[key] = np.zeros(n, dtype=float)
    return d


def anneal_run(rands_fns, t, np_rand_seed, config, prev_a, numproc):

    #
    outdir = config["outdir"]
    tilesfn = config["tilesfn"]
    metric_num = config["metric_num"]
    anneal_tallow_rad = config["anneal_tallow_rad"]
    anneal_niter = config["anneal_niter"]
    anneal_rmax = config["anneal_rmax"]
    anneal_nt_per_proc = config["anneal_nt_per_proc"]
    nccd_min = config["anneal_nccd_min"]
    r_distrib = config["anneal_r_distrib"]
    inflate_ra_factor = config["inflate_ra_factor"]

    np.random.seed(np_rand_seed)
    trad = get_decam_radius()

    # read rands catalogs
    # TODO: indiv. rands files can be read in parallel
    pixs = get_pixs_from_rands_fns(rands_fns)
    rs = {pix : Table.read(fn) for (pix, fn) in zip(pixs, rands_fns)}
    hdr = rs[pixs[0]].meta
    rnside = hdr["HPXNSIDE"]
    randdens = hdr["DENSITY"]
    assert hdr["HPXNEST"] == nest

    # allowed pixels for the tile centers [using the orig. tiles file]
    # using a reasonably small pixel size
    #   (different than the one for the rands)
    fn = os.path.join(config["outdir"], "tiles-allowtpixs.fits")
    if not os.path.isfile(fn):
        d = Table()
        tnside = 512
        d.meta["HPXNSIDE"], d.meta["HPXNEST"] = tnside, nest
        orig_t = get_init_tiles(config)
        d["HPXPIXEL"] = get_anneal_allowtpixs(orig_t, tnside, inflate_ra_factor, trad=anneal_tallow_rad)
        d.write(fn)
    d = Table.read(fn)
    tnside = d.meta["HPXNSIDE"]
    assert d.meta["HPXNEST"] == nest
    allow_tpixs = d["HPXPIXEL"]

    # reference ccds
    ccd_names = hdr["CCDNAMES"].split(",")
    ref_tilera, ref_tiledec, ref_radecs = get_ref_radecs(                                                                                                                                          
        ccd_names,
        config["npix_msk_xstart"],
        config["npix_msk_xend"],
        config["npix_msk_ystart"],
        config["npix_msk_yend"],
    )       

    tcs = SkyCoord(t["RA"] * u.degree, t["DEC"] * u.degree, frame="icrs")

    i_iter = 0
    touched_rand_fns = []
    start = time()
    if prev_a is None:
        a = init_anneal_table(0)
        t0 = 0.0
        curr_nccds = deepcopy(np.hstack([rs[pix]["NCCD"] for pix in pixs]))
        curr_area = (curr_nccds > 0).sum() / randdens
        curr_quants = {
            "nccds": curr_nccds,
            "delta_area": 0.,
            "nccd_min": nccd_min,
        }
        curr_metric, _ = get_metric(metric_num, curr_quants)
    else:
        a = prev_a.copy()
        t0 = a["TIME"][-1]
        curr_area = a["NEW_AREA"][-1]
        curr_metric = a["NEW_METRIC"][-1]
    new_metric, updated = None, False

    while i_iter < anneal_niter:

        curr_i_iter = i_iter
        ntry = 0

        while i_iter == curr_i_iter:

            # pick numproc non-overlapping (including offset) tiles
            # we consider the touched rands pixels
            # to decide overlap or no

            tmpstart = time()

            touched_rpixs = []
            ii_avail = np.arange(len(t), dtype=int)
            if "FREEZE" in t.colnames:
                log.info("exclude from annealing {} tiles with FREEZE=True".format(t["FREEZE"].sum()))
                ii_avail = ii_avail[~t["FREEZE"]]
            ii, pas, seps = [], [], []
            new_tras, new_tdecs = [], []

            while len(ii) < anneal_nt_per_proc:

                ok = False

                # pick a tile and an offset
                # inflate_ra_factor: inflate the dra offset
                # i.e. sep will be in [0, anneal_rmax]
                i = np.random.choice(ii_avail, size=1)[0]
                pa = np.random.uniform(low=0, high=360.0, size=1)[0] * u.degree
                if r_distrib == "uniform":
                    sep = np.random.uniform(low=0, high=anneal_rmax, size=1)[0] * u.degree
                if r_distrib == "gaussian":
                    sep = np.abs(np.random.normal(loc=0, scale=anneal_rmax, size=1))[0] * u.degree

                # offset position
                new_tc = tcs[i].directional_offset_by(pa, sep)
                new_tra, new_tdec = new_tc.ra.value, new_tc.dec.value
                dra = new_tra - t["RA"][i]
                # clip offset in [-180, 180]
                if dra > 180:
                    dra -= 360
                if dra < 180:
                    dra += 360
                new_tra = t["RA"][i] + inflate_ra_factor * dra
                new_tra %= 360

                # touched rands pixels
                # consider *both* old + new tiles
                new_rpixs = get_tiles_pixs([t["RA"][i], new_tra], [t["DEC"][i], new_tdec], rnside, inflate_ra_factor)
                # restrict to "existing" pixels [as some may have been discarded,
                #   as no rands had NCCD>0 for the initial tiling)
                sel = np.in1d(new_rpixs, pixs)
                new_rpixs = new_rpixs[sel]

                # to check allow_tpixs
                new_tpix = hp.ang2pix(
                    tnside,
                    np.radians(90.0 - new_tdec),
                    np.radians(new_tra),
                    nest=nest,
                )

                # offset tile ok?
                ok = (
                    (np.in1d(new_rpixs, touched_rpixs).sum() == 0)
                    &
                    (new_tpix in allow_tpixs)
                )

                # print("len(ii) = {}\tlen(touched_rpixs)={}\t(tra,tdec)=({:.1f},{:.1f})\tok={}".format(len(ii), len(touched_rpixs), new_tra, new_tdec, ok))

                if ok:

                    ii.append(i)
                    pas.append(pa)
                    seps.append(sep)
                    new_tras.append(new_tra)
                    new_tdecs.append(new_tdec)
                    ii_avail = ii_avail[ii_avail != i]
                    touched_rpixs += new_rpixs.tolist()

            # convert to numpy arrays
            ii = np.array(ii)
            pas = np.array([_.to(u.degree).value for _ in pas]) * u.degree
            seps = np.array([_.to(u.degree).value for _ in seps]) * u.degree
            new_tras, new_tdecs = np.array(new_tras), np.array(new_tdecs)
            old_tras, old_tdecs = t["RA"][ii], t["DEC"][ii]
            assert np.unique(touched_rpixs).size == len(touched_rpixs)
            touched_rpixs = np.unique(touched_rpixs) # np.array() + sorting
            touched_rands_fns = get_rands_fns_from_pixs(outdir, touched_rpixs)

            #print(
            #    "i_iter={}\t{:.1f}s\tpick {} tiles to offset (touching {} pixels)".format(i_iter, time() - tmpstart, anneal_nt_per_proc, touched_rpixs.size)
            #)
            #print(touched_rands_fns[0])

            # compute nccds for the "old" and "new" tiles
            old_t, new_t = Table(), Table()
            old_t["RA"], old_t["DEC"] = old_tras, old_tdecs
            new_t["RA"], new_t["DEC"] = new_tras, new_tdecs

            # we can deal with all of them at once because there
            # is no overlapping tiles in old_t, neither in new_t
            #tmpstart = time()
            touched_rands_old_nccds, _ = compute_nccds(touched_rands_fns, old_t, config, numproc)
            #print("i_iter={}\t{:.1f}s".format(i_iter, time() - tmpstart))
            #tmpstart = time()
            touched_rands_new_nccds, _ = compute_nccds(touched_rands_fns, new_t, config, numproc)
            #print("i_iter={}\t{:.1f}s".format(i_iter, time() - tmpstart))

            #fig, ax = plt.subplots(figsize=(15, 10))
            for j in range(anneal_nt_per_proc):

                # print(j)

                # rands pixels touched by the "old" + "new" tiles
                old_pixs_j = get_tiles_pixs(old_t["RA"][j], old_t["DEC"][j], rnside, inflate_ra_factor, trad=trad)
                sel = np.in1d(old_pixs_j, pixs)
                old_pixs_j = old_pixs_j[sel]
                new_pixs_j = get_tiles_pixs(new_t["RA"][j], new_t["DEC"][j], rnside, inflate_ra_factor, trad=trad)
                sel = np.in1d(new_pixs_j, pixs)
                new_pixs_j = new_pixs_j[sel]
                oldnew_unq_pixs_j = np.unique(old_pixs_j.tolist() + new_pixs_j.tolist())

                #ax.plot([old_t["RA"][j],new_t["RA"][j]], [old_t["DEC"][j],new_t["DEC"][j]], color="r", zorder=1, lw=2)
                new_rs = {pix : {"NCCD" : 0} for pix in oldnew_unq_pixs_j}
                assert len(list(new_rs.keys())) == len(oldnew_unq_pixs_j)
                for pix in old_pixs_j:
                    tmp_nccds = touched_rands_old_nccds[touched_rpixs == pix][0]
                    new_rs[pix]["NCCD"] -= tmp_nccds
                    #ax.scatter(new_rs[pix]["RA"], new_rs[pix]["DEC"], c=-tmp_nccds, s=1, alpha=0.1, vmin=-1, vmax=1, rasterized=True, zorder=0)
                for pix in new_pixs_j:
                    tmp_nccds = touched_rands_new_nccds[touched_rpixs == pix][0]
                    new_rs[pix]["NCCD"] += tmp_nccds
                    #ax.scatter(new_rs[pix]["RA"], new_rs[pix]["DEC"], c=tmp_nccds, s=1, alpha=0.1, vmin=-1, vmax=1, rasterized=True, zorder=0)
                # first: copy all rs NCCD arrays for all pixels
                new_nccds = deepcopy([rs[pix]["NCCD"] for pix in pixs])
                # then: update values for touched pixels
                for pix in new_rs:
                    k = np.where(pixs == pix)[0][0]
                    new_nccds[k] += new_rs[pix]["NCCD"]
                new_nccds = np.hstack(new_nccds)
                new_area = (new_nccds > 0).sum() / randdens
                new_quants = {
                    "nccds": new_nccds,
                    "delta_area": np.abs(new_area - curr_area) / curr_area,
                    "nccd_min": nccd_min,
                }
                new_metric, _ = get_metric(metric_num, new_quants)

                # print("{}/{}\toffset tile by {:.2f}\tOLD_METRIC={}\tNEW_METRIC={}".format(j, anneal_nt_per_proc-1, seps[j], curr_metric, new_metric))

                if new_metric > curr_metric:

                    # tile index in the orig. tiles table
                    i = ii[j]

                    log.info(
                        "{:.1f}s\t{}\t{}\t\t{}\t{}\t{}".format(
                            time() - start,
                            i_iter,
                            ntry,
                            t["TILEID"][i],
                            curr_metric,
                            new_metric,
                        )
                    )
                    # update annealing table
                    new_a = init_anneal_table(1)
                    new_a["TIME"] = time() - start + t0
                    new_a["NTRY"] = ntry
                    new_a["TILEID"] = t["TILEID"][i]
                    new_a["OLD_RA"], new_a["OLD_DEC"] = old_tras[j], old_tdecs[j]
                    new_a["OLD_AREA"] = curr_area
                    new_a["OLD_METRIC"] = curr_metric
                    new_a["PA_DEG"] = pas[j].to(u.degree).value
                    new_a["SEP_DEG"] = seps[j].to(u.degree).value
                    new_a["NEW_RA"], new_a["NEW_DEC"] = new_tras[j], new_tdecs[j]
                    new_a["NEW_AREA"] = new_area
                    new_a["NEW_METRIC"] = new_metric
                    a = vstack([a, new_a])
                    # update tiles
                    t["RA"][i], t["DEC"][i] = new_tras[j], new_tdecs[j]
                    tcs = SkyCoord(t["RA"] * u.degree, t["DEC"] * u.degree, frame="icrs")
                    # update list of touched rpixs / rfns
                    touched_rand_fns += list(get_rands_fns_from_pixs(outdir, oldnew_unq_pixs_j))
                    # update NCCD in rands
                    for pix in oldnew_unq_pixs_j:
                        rs[pix]["NCCD"] += deepcopy(new_rs[pix]["NCCD"])
                    # update metric
                    curr_area = new_area
                    curr_metric = new_metric

                    i_iter += 1
                    ntry = 0
                    updated = True

                else:

                    ntry += 1
                    del new_rs, new_area, new_nccds, new_quants, new_metric

                # if we reached the req. nb. of iterations, break
                if i_iter == anneal_niter:

                    break

            #ax.set_xlim(ax.get_xlim()[::-1])
            #ax.grid()
            #plt.savefig("/global/cfs/cdirs/desi/users/raichoor/tmpdir/tmp.png", bbox_inches="tight")
            #plt.close()
            #exit()

    all_nccds = np.array(
        [
            rs[pix]["NCCD"] for pix in pixs
        ], dtype=object
    )

    touched_rand_fns = np.unique(touched_rand_fns)

    return all_nccds, t, a, touched_rand_fns


def anneal_plot(outpng, a, t, title=None, xlim=None, ylim=None):

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.2, wspace=0.2)

    ax = plt.subplot(gs[0, 0])
    ax.plot(a["NTRY"], "-")
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Ntries")
    ax.grid()

    ax = plt.subplot(gs[0, 1])
    ax.plot(a["TIME"], "-")
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Time [s]")
    ax.grid()

    ax = plt.subplot(gs[0, 2])
    ax.plot(a["NEW_METRIC"], "-")
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Metric")
    ax.grid()

    ax = plt.subplot(gs[1, 0])
    ax.plot(a["PA_DEG"], "-")
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Position angle offset [deg]")
    ax.grid()
 
    ax = plt.subplot(gs[1, 1])
    ax.plot(a["SEP_DEG"], "-")
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Separation offset [deg]")
    ax.grid()
 

    plt.savefig(outpng, bbox_inches="tight")
    plt.close()


def nccd_plot(outpng, r, t, t0, metric_num=None, nccd_min=0, title=None, ralim=None, declim=None, nccd_clim=(5, 12), nplot=int(1e6)):

    sel = r["NCCD"] > 0
    r = r[sel]
    area = sel.sum() / r.meta["DENSITY"]

    mean_nocut, std_nocut = r["NCCD"].mean(), r["NCCD"].std()
    sel = r["NCCD"] >= nccd_min
    mean, std = r["NCCD"][sel].mean(), r["NCCD"][sel].std()

    if len(r) > nplot:
        np.random.seed(1234)
        ii = np.random.choice(len(r), size=nplot, replace=False)
        r = r[ii]

    if metric_num is not None:

        _, metric_str = get_metric(metric_num, None)

    assert len(t0) == len(t)
    t0cs = SkyCoord(t0["RA"] * u.degree, t0["DEC"] * u.degree, frame="icrs")
    tcs = SkyCoord(t["RA"] * u.degree, t["DEC"] * u.degree, frame="icrs")
    tseps = tcs.separation(t0cs).to(u.deg).value

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(
        2, 4, wspace=0.2, hspace=0.3, width_ratios=[1.0, 0.02, 0.02, 0.3]
    )

    ax = plt.subplot(gs[0, 0])
    # clim = (5, 12)
    cmap = get_quantz_cmap(matplotlib.cm.jet, nccd_clim[1] - nccd_clim[0] + 1, 0, 1)
    sc = ax.scatter(
        -99, -99, c=0.0, cmap=cmap, vmin=nccd_clim[0], vmax=nccd_clim[1],
        label="{}/deg2 rands over {:.0f} deg2".format(r.meta["DENSITY"], area),
    )
    ax.scatter(
        r["RA"],
        r["DEC"],
        c=r["NCCD"],
        s=0.1,
        alpha=0.1,
        zorder=0,
        cmap=cmap,
        vmin=nccd_clim[0],
        vmax=nccd_clim[1],
        rasterized=True,
    )
    ax.set_title(title)
    ax.set_xlabel("R.A. [deg]")
    ax.set_ylabel("Dec. [deg]")
    ax.set_xlim(ralim)
    ax.set_ylim(declim)
    ax.grid()
    ax.legend(loc=2)
    cbar = plt.colorbar(sc, cax=plt.subplot(gs[0, 1]), ax=ax, extend="both")
    cbar.set_label("Nccd")
    cbar.mappable.set_clim(nccd_clim)

    ax = plt.subplot(gs[0, -1])
    bins = -0.5 + np.arange(0, 17)
    _ = ax.hist(r["NCCD"], bins=bins, density=True, histtype="stepfilled", alpha=0.5)
    ax.axvline(nccd_min, color="k", ls="--")#, label="for metric stats: nccd_min = {}".format(nccd_min))
    if metric_num is not None:
        ax.text(0.95, 0.95, "metric = {}".format(metric_str), ha="right", transform=ax.transAxes)
    ax.text(0.95, 0.85, "Nccd>=0: (mean, std) = {:.2f}, {:.2f}".format(mean_nocut, std_nocut), ha="right", transform=ax.transAxes)
    if nccd_min > 0:
        ax.text(0.95, 0.75, "Nccd>={}: (mean, std) = {:.2f}, {:.2f}".format(nccd_min, mean, std), ha="right", transform=ax.transAxes)
    ax.set_xlabel("Nccd")
    ax.set_ylabel("Normalized counts")
    ax.grid(lw=0.5)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 0.5)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))

    ax = plt.subplot(gs[1, -1])
    bins = -0.5 + np.arange(0, 17)
    _ = ax.hist(r["NCCD"], bins=bins, density=True, cumulative=-1, histtype="stepfilled", alpha=0.5)
    ax.set_xlabel("Nccd")
    ax.set_ylabel("Fraction with coverage >= Nccd")
    ax.grid(lw=0.5)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))

    ax = plt.subplot(gs[1, 0])
    clim = (0, 0.5)
    cmap = get_quantz_cmap(matplotlib.cm.jet, 10, 0, 1)
    sc = ax.scatter(t["RA"], t["DEC"], c=tseps, s=1, zorder=1, cmap=cmap, vmin=clim[0], vmax=clim[1], label="{} tiles".format(len(t)))
    if title is not None:
        ax.set_title(title.replace("rands-anneal", "tiles-anneal"))
    ax.set_xlabel("R.A. [deg]")
    ax.set_ylabel("Dec. [deg]")
    ax.set_xlim(ralim)
    ax.set_ylim(declim)
    ax.grid(lw=0.5)
    ax.legend(loc=2, ncol=2, markerscale=5)
    cbar = plt.colorbar(sc, cax=plt.subplot(gs[1, 1]), ax=ax, extend="max")
    cbar.set_label("Offset from orig. position [deg]")
    cbar.mappable.set_clim(clim)

    plt.savefig(outpng, bbox_inches="tight")
    plt.close()
