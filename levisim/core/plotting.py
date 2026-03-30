import matplotlib.pyplot as plt

__all__ = [
    'LATEX_FIG_WIDTH',
    'set_plt_rcParams_to_xmgrace_style',
    'get_latex_doc_lims',
]

# match the width (inches) of my thesis document
# latex commands to find out the values are:
# textwidth in cm: \printinunitsof{cm}\prntlen{\textwidth}
# LATEX_FIG_WIDTH = 13.998 # centimetres
# textwidth in in: \printinunitsof{in}\prntlen{\textwidth}
LATEX_FIG_WIDTH = 5.5129 # inches


# https://matplotlib.org/stable/users/explain/customizing.html
def set_plt_rcParams_to_xmgrace_style():
    """
    Resets matplotlib plotting to a style inspired by xmgrace
    """
    plt.style.use('default')

    # custom styling to match xmgrace
    plt.rcParams.update({
        # https://matplotlib.org/stable/users/explain/axes/constrainedlayout_guide.html
        # 'figure.constrained_layout.use': True,

        # https://matplotlib.org/stable/users/explain/text/usetex.html
        # NOTE: do not use the below (which sets *all* text to use TeX font)
        # 'text.usetex': True,

        # Instead, define the separate `mathtext.fontset`
        # https://matplotlib.org/stable/gallery/text_labels_and_annotations/usetex_baseline_test.html#sphx-glr-gallery-text-labels-and-annotations-usetex-baseline-test-py
        # https://matplotlib.org/stable/gallery/text_labels_and_annotations/mathtext_fontfamily_example.html
        'mathtext.fontset': 'cm',
        'font.family': 'Arial',
        'font.weight': 'normal',

        'axes.linewidth': 1.5,
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',

        'axes.spines.right': True,

        'xtick.top': True,

        'ytick.left': True,
        'ytick.right': True,
        'ytick.minor.left': True,
        'ytick.minor.right': True,

        'xtick.labelsize': 12,
        'ytick.labelsize': 12,

        'xtick.color': 'black',
        'ytick.color': 'black',

        'xtick.direction': 'in',
        'ytick.direction': 'in',

        'xtick.major.size': 8,
        'ytick.major.size': 8,

        'xtick.minor.size': 4,
        'ytick.minor.size': 4,

        'lines.linewidth': 1,
        'lines.markersize': 8,

        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',

        'animation.embed_limit': 2**128,
    })


def get_latex_doc_lims(
    scale_factor,
    order_of_magnitude,
    doc_width_inches=LATEX_FIG_WIDTH,
    doc_height_inches=3.0,
):
    """
    Get limits that allow for equal scaling (i.e. a circle looks like a circle)
    but with the desired aspect ratio for latex documents
    """
    sf, e10 = scale_factor, order_of_magnitude
    dwi, dhi = doc_width_inches, doc_height_inches
    xlims = -(sf * dwi) * e10, (sf * dwi) * e10
    ylims = -(sf * dhi) * e10, (sf * dhi) * e10
    return xlims, ylims
