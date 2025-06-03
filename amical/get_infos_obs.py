"""
@author: Anthony Soulain (University of Sydney)

-------------------------------------------------------------------------
AMICAL: Aperture Masking Interferometry Calibration and Analysis Library
-------------------------------------------------------------------------

Instruments and mask informations.
--------------------------------------------------------------------
"""

import sys

import numpy as np
from rich import print as rprint

from amical.tools import mas2rad

if sys.version_info >= (3, 9):
    import importlib.resources as importlib_resources
else:
    import importlib_resources


def get_mask(ins, mask, first=0):
    """Return dictionnary containning saved informations about masks."""

    pupil_visir = 8.0
    pupil_visir_mm = 17.67
    off = 0.3
    dic_mask = {
        "NIRISS": {
            "g7": np.array(
                [
                    [0, -2.64],
                    [-2.28631, 0],
                    [2.28631, -1.32],
                    [-2.28631, 1.32],
                    [-1.14315, 1.98],
                    [2.28631, 1.32],
                    [1.14315, 1.98],
                ]
            ),
            "g7_bis": np.array(
                [
                    [0, 2.9920001],
                    [2.2672534, 0.37400016],
                    [-2.2672534, 1.6829998],
                    [2.2672534, -0.93499988],
                    [1.1336316, -1.5895000],
                    [-2.2672534, -0.93500012],
                    [-1.1336313, -1.5895000],
                ]
            ),
            "g7_sb": np.array(
                [
                    [0, -2.64],  # 0
                    [-2.28631, 0],  # 1
                    [-2.28631 + off, 0],
                    [-2.28631 - off / np.sqrt(2), 0 + off / np.sqrt(2)],
                    [-2.28631 - off / np.sqrt(2), 0 - off / np.sqrt(2)],
                    [2.28631, -1.32],  # 2
                    [-2.28631, 1.32],  # 3
                    [-1.14315, 1.98],  # 4
                    [-1.14315 + off, 1.98],
                    [-1.14315 - off / np.sqrt(2), 1.98 + off / np.sqrt(2)],
                    [-1.14315 - off / np.sqrt(2), 1.98 - off / np.sqrt(2)],
                    [2.28631, 1.32],  # 5
                    [2.28631 + off, 1.32],
                    [2.28631 - off / np.sqrt(2), 1.32 + off / np.sqrt(2)],
                    [2.28631 - off / np.sqrt(2), 1.32 - off / np.sqrt(2)],
                    [1.14315, 1.98],  # 6
                ]
            ),
        },
        "GLINT": {
            "g4": np.array(
                [[2.725, 2.317], [-2.812, 1.685], [-2.469, -1.496], [-0.502, -2.363]]
            )
        },
        "NACO": {
            "g7": np.array(
                [
                    [-3.51064, -1.99373],
                    [-3.51064, 2.49014],
                    [-1.56907, 1.36918],
                    [-1.56907, 3.61111],
                    [0.372507, -4.23566],
                    [2.31408, 3.61111],
                    [4.25565, 0.248215],
                ]
            )
            * (8 / 10.0),
        },
        "SPHERE": {
            "g7": 1.05
            * np.array(
                [
                    [-1.46, 2.87],
                    [1.46, 2.87],
                    [-2.92, 0.34],
                    [-1.46, -0.51],
                    [-2.92, -1.35],
                    [2.92, -1.35],
                    [0, -3.04],
                ]
            )
        },
        "SPHERE-IFS": {
            "g7": 1
            * np.array(
                [
                    [-2.07, 2.71],
                    [0.98, 3.27],
                    [-3.11, -0.2],
                    [-1.43, -0.81],
                    [-2.79, -1.96],
                    [3.3, -0.85],
                    [0.58, -3.17],
                ]
            )
        },
        "VISIR": {
            "g7": (pupil_visir / pupil_visir_mm)
            * np.array(
                [
                    [-5.707, -2.885],
                    [-5.834, 3.804],
                    [0.099, 7.271],
                    [7.989, 0.422],
                    [3.989, -6.481],
                    [-3.790, -6.481],
                    [-1.928, -2.974],
                ]
            ),
        },
        "VAMPIRES": {
            "g18": np.array(
                [
                    [0.821457, 2.34684],
                    [-2.34960, 1.49034],
                    [-2.54456, 2.55259],
                    [1.64392, 3.04681],
                    [2.73751, -0.321102],
                    [1.38503, -3.31443],
                    [-3.19337, -1.68413],
                    [3.05126, 0.560011],
                    [-2.76083, 1.14035],
                    [3.02995, -1.91449],
                    [0.117786, 3.59025],
                    [-0.802156, 3.42140],
                    [-1.47228, -3.28982],
                    [-1.95968, -0.634178],
                    # [-3.29085, -1.15300],
                    [0.876319, -3.13328],
                    [2.01253, -1.55220],
                    [-2.07847, -2.57755],
                ]
            ),
            "g7":np.array( # Mask coords at M1 (m), hole diam at M1 = 1.1m
                    [
                        [2.52145  ,-1.52360],
                        [2.91280  , 1.66366],
                        [1.43484  , 1.03630],
                        [1.63051  , 2.62994],
                        [-0.434472,  -2.77832],
                        [-1.12974 ,  2.96885],
                        [-2.80338 , 0.747867],
                    ]
            ),
            # at M1 (m)
            # hole diam at M1: 1.307m 
            'jewel_4x5_0': np.array(
                [   [ 1.89027635, -2.55336464],
                    [-0.04912034,  3.45349192],
                    [-3.34931425,  0.96661779],
                    [-1.91351673, -0.93875002],
                    [ 2.82247451, -0.35724367],
                ]
            ),
            'jewel_4x5_1': np.array(
                    [   [ 0.79021178, -3.38232252],
                        [ 1.72240994, -1.18620186],
                        [ 1.21881071,  2.91528702],
                        [-3.01358143, -1.76770824],
                        [-1.14918498,  2.62453376],
                    ]
            ),
            'jewel_4x5_2': np.array(
                [   [ 0.11874607,  2.08632914],
                    [ 2.6546081,   1.00991928],
                    [-2.08138314,  0.4284129 ],
                    [-0.47771927, -2.8441179 ],
                    [ 2.99034092, -1.72440649],
                ]
            ),
            'jewel_4x5_3': np.array(
                    [   [ 2.48674169,  2.37708212],
                        [-2.24924968,  1.79557588],
                        [ 0.62234537, -2.01515961],
                        [-1.74565032, -2.30591286],
                        [-3.18144784, -0.40054519],
                    ]
            ),

            # at M1 (m)
            # hole diam at M1: 0.89
            'jewel_7x6_0': np.array(
                [   [-1.48631408, -0.33270991],
                    [-2.97830983, -1.26677547],
                    [-1.61216721,  3.18556796],
                    [ 1.56060417, -0.2237179 ],
                    [ 1.62353053, -1.98285707],
                    [-0.05724477,  2.36049474],
                ]
            ),
            'jewel_7x6_1': np.array(
                [   [ 2.51331683,  1.57175169],
                    [-0.40774817, -2.05551838],
                    [-3.04123646,  0.49236365],
                    [ 1.14717421, -2.88059174],
                    [-1.54924071,  1.42642908],
                    [ 2.57624373, -0.18738731],
                ]
            ),
            'jewel_7x6_2': np.array(
                [   [ 0.60789118, -2.0191878 ],
                    [-3.51759279, -0.40537136],
                    [ 2.16281349, -2.84426129],
                    [ 1.43475091,  3.29456013],
                    [-2.02559704,  0.52869417],
                    [ 1.49767768,  1.53542111],
                ]
            ),
            'jewel_7x6_3': np.array(
                    [   [ 1.08424764, -1.12145286],
                        [ 0.95839472,  2.39682519],
                        [ 2.63917009, -1.94652622],
                        [-1.96267055, -1.23044475],
                        [-0.53360123,  1.46275966],
                        [-1.89974392, -2.98958391],
                    ]
            ),
            'jewel_7x6_4': np.array(
                [   [-0.94703113, -1.19411417],
                    [ 2.09988713, -1.0851222 ],
                    [-1.07288418,  2.32416375],
                    [ 2.03696063,  0.67401689],
                    [ 0.41911176,  3.25822941],
                    [-2.5019535,  -0.36904057],
                ]
            ),
            'jewel_7x6_5': np.array(
                    [   [-2.56487986,  1.39009835],
                        [ 1.97403387,  2.43315591],
                        [-2.43902687, -2.12817969],
                        [ 3.59188274, -0.15105679],
                        [-0.88410463, -2.95325319],
                        [ 0.48203819,  1.49909052],
                    ]
            ),
            'jewel_7x6_6': np.array(
                [   [ 3.05259978,  0.71034748],
                    [ 3.11552668, -1.04879155],
                    [-1.42338759, -2.09184897],
                    [ 0.13153479, -2.91692247],
                    [-2.08852354,  2.28783329],
                    [-0.59652766,  3.22189869],
                ]
            ),

        },
    }

    #

    try:
        xycoords = dic_mask[ins][mask]
        nrand = [first]
        for x in np.arange(len(xycoords)):
            if x not in nrand:
                nrand.append(x)
        xycoords_sel = xycoords[nrand]
    except KeyError:
        rprint(
            f"[red]\n-- Error: maskname ({mask}) unknown for {ins}.", file=sys.stderr
        )
        xycoords_sel = None
    return xycoords_sel


def get_wavelength(ins, filtname):
    """Return dictionnary containning saved informations about filters."""
    from astropy.io import fits

    datadir = importlib_resources.files("amical") / "internal_data"
    YJfile = datadir / "ifs_wave_YJ.fits"
    YJHfile = datadir / "ifs_wave_YJH.fits"

    try:
        with fits.open(YJfile) as fd:
            wave_YJ = fd[0].data
        with fits.open(YJHfile) as fd:
            wave_YJH = fd[0].data
    except:
        pass
        # rprint(
        #     f"[blue]\n-- Warning: missing internal data files (fits) for SPHERE-IFS filters.",
        #     file=sys.stderr,
        # )
        # wave_YJ = None
        # wave_YJH = None

    dic_filt = {
        "NIRISS": {
            "F277W": [2.776, 0.715],
            "F380M": [3.828, 0.205],
            "F430M": [4.286, 0.202],
            "F480M": [4.817, 0.298],
        },
        "SPHERE": {
            "H2": [1.593, 0.052],
            "H3": [1.667, 0.054],
            "H4": [1.733, 0.057],
            "K1": [2.110, 0.102],
            "K2": [2.251, 0.109],
            "CntH": [1.573, 0.023],
            "CntK1": [2.091, 0.034],
            "CntK2": [2.266, 0.032],
        },
        # "SPHERE-IFS": {"YJ": wave_YJ, "YH": wave_YJH},
        "GLINT": {"F155": [1.55, 0.01], "F430": [4.3, 0.01]},
        "VISIR": {"10_5_SAM": [10.56, 0.37], "11_3_SAM": [11.23, 0.55]},
        "VAMPIRES": {
            "750-50": [0.75, 0.05],
            "F610": [0.612, 0.06],
            "F670": [0.67, 0.043], 
            "F720": [0.719, 0.046],
            "F760":[0.760, 0.032],
        },
    }

    if ins not in dic_filt.keys():
        raise KeyError(
            f"--- Error: instrument <{ins}> not found ---\n"
            "Available: %s" % list(dic_filt.keys())
        )
    if filtname not in dic_filt[ins]:
        raise KeyError(
            f"Missing input: filtname <{filtname}> not found for {ins} (Available: {list(dic_filt[ins])})"
        )
    return np.array(dic_filt[ins][filtname]) * 1e-6


def get_pixel_size(ins):
    saved_pixel_detector = {
        "NIRISS": 65.6,
        "SPHERE": 12.27,
        "VISIR": 45,
        "SPHERE-IFS": 7.46,
        "VAMPIRES": 5.9, #mas/px Lucas et. al. 2024
    }
    try:
        p = mas2rad(saved_pixel_detector[ins])
    except KeyError:
        p = np.NaN
    return p


def get_ifu_table(
    i_wl, filtname="YH", instrument="SPHERE-IFS", verbose=False, display=False
):
    """Get spectral information for the given instrumental IFU setup.
    `i_wl` can be an integer, a list of 2 integers (to get a range between those
    two) or a list of integers (>= 3) used to display the
    requested spectral channels."""
    wl = get_wavelength(instrument, filtname) * 1e6

    if verbose:
        print(f"\nInstrument: {instrument}, spectral range: {filtname}")
        print("-----------------------------")
        print(
            f"spectral coverage: {wl[0]:2.2f} - {wl[-1]:2.2f} µm (step = {np.diff(wl)[0]:2.2f})"
        )

    one_wl = True
    multiple_wl = False
    if isinstance(i_wl, list) & (len(i_wl) == 2):
        one_wl = False
        wl_range = wl[i_wl[0] : i_wl[1]]
        sp_range = np.arange(i_wl[0], i_wl[1], 1)
    elif isinstance(i_wl, list) & (len(i_wl) > 2):
        multiple_wl = True
        one_wl = False
    elif i_wl is None:
        one_wl = False
        sp_range = np.arange(len(wl))
        wl_range = wl

    if display:
        from matplotlib import pyplot as plt

        plt.figure(figsize=(4, 3))
        plt.title("--- SPECTRAL INFORMATION (IFU)---")
        plt.plot(wl, label="All spectral channels")
        if one_wl:
            plt.plot(
                np.arange(len(wl))[i_wl],
                wl[i_wl],
                "ro",
                label=f"Selected ({wl[i_wl[0]]:2.2f} µm)",
            )
        elif multiple_wl:
            plt.plot(
                i_wl,
                wl[i_wl],
                "ro",
                label="Selected",
            )
        else:
            plt.plot(
                sp_range,
                wl_range,
                lw=5,
                alpha=0.5,
                label=f"Selected ({wl_range[0]:2.2f}-{wl_range[-1]:2.2f} µm)",
            )
        plt.legend()
        plt.xlabel("Spectral channel")
        plt.ylabel("Wavelength [µm]")
        plt.tight_layout()

    if one_wl:
        output = wl[i_wl]
    elif multiple_wl:
        output = np.array(wl[i_wl])
    else:
        output = wl_range
    return output
