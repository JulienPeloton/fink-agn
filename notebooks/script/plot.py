import numpy as np

from collections import Counter
import healpy as hp

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