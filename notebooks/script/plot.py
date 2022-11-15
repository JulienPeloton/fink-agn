import numpy as np
import pandas as pd

from collections import Counter
import healpy as hp

import matplotlib.pyplot as plt

from astropy.time import Time
import swifttools.ukssdc.data.SXPS as uds

def plot_mwd(fig, RA,Dec,color,ax, cmap='viridis', alpha=0.5, cb=True,org=0,title=None, cb_title='magnitude',projection='mollweide'):
    ''' RA, Dec are arrays of the same length.
    RA takes values in [0,360), Dec in [-90,90],
    which represent angles in degrees.
    org is the origin of the plot, 0 or a multiple of 30 degrees in [0,360).
    title is the title of the figure.
    projection is the kind of projection: 'mollweide', 'aitoff', 'hammer', 'lambert'
    '''
    x = np.remainder(RA+360-org,360) # shift RA values
    ind = x>180
    x[ind] -= 360    # scale conversion to [-180, 180]
    x=-x    # reverse the scale: East to the left
    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tick_labels = np.remainder(tick_labels+360+org,360)
    cm = ax.scatter(
        np.radians(x),
        np.radians(Dec), 
        c=color,
        alpha=alpha,
        marker='.',
        cmap=cmap
    )  # convert degrees to radians
    if cb:
        cb = fig.colorbar(cm)
        cb.set_label(cb_title)
        
    if title is not None:
        tick_labels = np.array(['', '120', '', '60', '', '0', '', '300', '', '240', ''])
        ax.set_xticklabels(tick_labels)     # we add the scale on the x axis
        ax.set_title(title)
        ax.title.set_fontsize(15)
        ax.set_xlabel("RA")
        ax.xaxis.label.set_fontsize(12)
        ax.set_ylabel("Dec")
        ax.yaxis.label.set_fontsize(12)
        ax.grid(True)

def dec2theta(dec: float) -> float:
    """ Convert Dec (deg) to theta (rad)
    """
    return np.pi / 2.0 - np.pi / 180.0 * dec

def ra2phi(ra: float) -> float:
    """ Convert RA (deg) to phi (rad)
    """
    return np.pi / 180.0 * ra

def get_ppix(pdf, nside=1024):

    pixs = hp.ang2pix(
        nside, 
        dec2theta(pdf['ra'].values),
        ra2phi(pdf['dec'].values),
        lonlat=True
    )
    
    pdict = Counter(pixs)

    ppixs = [pdict[i] for i in pixs]
    
    return ppixs


def plot_xlc(ax, data, band, band_label):

    if band in data:
        ytop = data[band]["RatePos"]
        ybot = data[band]["RateNeg"]

        xright = data[band]["TimePos"]
        xleft = data[band]["TimeNeg"]

        ax.errorbar(
            data[band]["Time"],
            data[band]["Rate"],
            yerr=[-ybot, ytop],
            xerr=[-xleft, xright],
            label=band_label,
            capsize=2, elinewidth=1, marker='.', linestyle='none'
        )

        return data[band]["Time"], data[band]["Rate"]
    else:
        return pd.Series([], dtype=np.float64), pd.Series([], dtype=np.float64)


start_fink = Time("2019-11-01")
def plot_mw_lc(data, objectId):
    current_obj = data[data["objectId"] == objectId].sort_values("jd")
    lsxps_id = current_obj["LSXPS_ID"].values[0]

    lcs = uds.getLightCurves(
        sourceID=int(lsxps_id),
        cat='LSXPS',
        binning='obsid',
        timeFormat='MJD',
        saveData=False,
        returnData=True
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))

    fig.suptitle('{} - {} - {}'.format(current_obj['objectId'].values[0],
              current_obj['fink_class'].values[0], current_obj['IAUName'].values[0]))

    dicname = {1: 'g', 2: 'r'}
    for filt in [1, 2]:
        mask = current_obj['fid'].values == filt

        if np.sum(mask) == 0:
            continue
        ax1.errorbar(
            current_obj['jd'].values[mask],
            current_obj['dcmag'].values[mask],
            current_obj['dcmag_err'].values[mask],
            ls='',
            marker='.',
            label=dicname[filt],
            color='C{}'.format(filt - 1)
        )

    ax1.set_ylabel('DC magnitude')
    ax1.set_xlabel('Days since the beginning of Fink')
    ax1.invert_yaxis()

    plt.sca(ax1)
    plt.vlines(
        start_fink.jd,
        np.min(current_obj['dcmag'].values),
        np.max(current_obj['dcmag'].values),
        colors="red",
        linestyles='dotted',
        label="FINK start date"
    )
    plt.xticks(
        current_obj['jd'].values[::45], 
        (current_obj['jd'].values - start_fink.jd)[::45].astype(int),
        rotation=90
    )
    ax1.legend()

    soft_time, soft_rate = plot_xlc(ax2, lcs, "Soft_rates", "0.3 - 1 keV")
    medium_time, medium_rate = plot_xlc(ax2, lcs, "Medium_rates", "1 - 2 keV")
    hard_time, hard_rate = plot_xlc(ax2, lcs, "Hard_rates", "2 - 10 keV")

    xray_time = pd.concat([soft_time, medium_time, hard_time]).sort_values()
    xray_rate = pd.concat([soft_rate, medium_rate, hard_rate])
    jd_xray = Time(xray_time, format="mjd").jd

    plt.sca(ax2)
    plt.vlines(
        start_fink.mjd, 
        np.min(xray_rate),
        np.max(xray_rate), 
        colors="red",
        linestyles='dotted',
        label="FINK start date"
    )
    plt.xticks(
        xray_time, 
        (jd_xray - start_fink.jd).astype(int),
        rotation=90
    )

    ax2.legend()
    ax2.set_ylabel('Rate')
    ax2.set_xlabel('Days since the beginning of Fink')

    plt.tight_layout()
    plt.show()
