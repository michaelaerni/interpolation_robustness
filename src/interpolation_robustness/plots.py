import typing

import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.container
import matplotlib.pyplot
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

import interpolation_robustness as ir

DEFAULT_PPI = 300.0
FONT_SIZE_PT = 10
FONT_SIZE_SMALL_PT = 8
FONT_SIZE_LARGE_PT = 14
TEX_PT_PER_IN = 72.27
LINE_WIDTH = 4
LINE_WIDTH_PT = 1

# Colors based on https://davidmathlogic.com/colorblind
DEFAULT_COLORMAP = matplotlib.colors.ListedColormap(
    colors=(
        '#1E88E5',
        '#FFC107',
        '#B51751',
        '#81CBE6',
        '#4AB306',
        '#004D40'
    ),
    name='cvd_friendly'
)

MARKER_MAP = (
    'o',
    'd',
    'x',
    '+'
)

LINESTYLE_MAP = (
    'solid',
    'dashed',
    'dotted',
    (0, (3, 1, 1, 1, 1, 1))  # dash dot dot
)


def setup_plotly_template(ppi: float = DEFAULT_PPI, show_titles: bool = True):
    # Define the template used for plots and export
    pio.templates['export'] = go.layout.Template(
        data_scatter=[go.Scatter(line_width=LINE_WIDTH)],
        layout_yaxis_gridcolor='lightgray',
        layout_xaxis_gridcolor='lightgray',
        layout_legend_font_size=pt_to_px(FONT_SIZE_PT, ppi),
        layout_showlegend=True,
        layout_legend_bgcolor='rgba(0,0,0,0)',
        layout_title_font_size=pt_to_px(FONT_SIZE_LARGE_PT, ppi),
        layout_xaxis_title_font_size=pt_to_px(FONT_SIZE_PT, ppi),
        layout_yaxis_title_font_size=pt_to_px(FONT_SIZE_PT, ppi),
        layout_xaxis_tickfont_size=pt_to_px(FONT_SIZE_SMALL_PT, ppi),
        layout_yaxis_tickfont_size=pt_to_px(FONT_SIZE_SMALL_PT, ppi),
        layout_xaxis_tickangle=0,
        layout_yaxis_tickangle=0,
        layout_width=1536,
        layout_height=576,
        layout_margin_t=100 if show_titles else 20,
        layout_margin_l=120,
        layout_margin_b = 120
    )

    # Remove grids and set template as default
    pio.templates.default = 'xgridoff+ygridoff+export'


def setup_matplotlib(show_titles: bool = True):
    matplotlib.pyplot.rcdefaults()

    # Use colormap which works for people with CVD and greyscale printouts
    matplotlib.cm.register_cmap(cmap=DEFAULT_COLORMAP)

    matplotlib.rcParams.update({
        'text.usetex': True,
        'image.cmap': DEFAULT_COLORMAP.name,
        'axes.prop_cycle': matplotlib.rcsetup.cycler(
            'color', DEFAULT_COLORMAP.colors
        ),
        'font.family': 'sans-serif',
        'font.sans-serif': ['Open Sans'],
        'figure.dpi': DEFAULT_PPI,
        'axes.titlesize': FONT_SIZE_LARGE_PT,
        'axes.labelsize': FONT_SIZE_PT,
        'lines.linewidth': LINE_WIDTH_PT,
        'xtick.labelsize': FONT_SIZE_SMALL_PT,
        'ytick.labelsize': FONT_SIZE_SMALL_PT,
        'lines.markersize': 3,
        'scatter.edgecolors': 'black',
        'legend.frameon': False,
        'legend.fontsize': FONT_SIZE_PT,
        'legend.handlelength': 1.0,
        'legend.borderpad': 0.1,
        'legend.borderaxespad': 0.1,
        'legend.labelspacing': 0.2,
        'savefig.dpi': DEFAULT_PPI,
        'savefig.pad_inches': 0.1,
        'savefig.transparent': True,
        'figure.constrained_layout.use': True,
        'grid.color': '#b0b0b0',
        'grid.linestyle': '-',
        'grid.linewidth': 0.3,
        'grid.alpha': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.5
    })


def matplotlib_scatter_marker_settings() -> typing.Dict[str, typing.Any]:
    return {
        'linewidths': 1.0,
        's': 16.0
    }


def marker_settings() -> typing.Dict[str, typing.Any]:
    return{
        'marker_size': 16,
        'marker_line_width': 4
    }


def pt_to_px(value_pt: float, ppi: float = DEFAULT_PPI) -> float:
    return value_pt / TEX_PT_PER_IN * ppi


def find_best_metrics(
        metrics: ir.mlflow.MetricsMap,
        metric_key: str,
        maximize: bool
) -> typing.Dict[str, float]:
    comp = np.max if maximize else np.min
    return {run_id: comp(run_metrics[metric_key]) for run_id, (_, run_metrics) in metrics.items()}


def find_last_metrics(metrics: ir.mlflow.MetricsMap, metric_key: str, average_over: int) -> typing.Dict[str, np.ndarray]:
    return {run_id: run_metrics[metric_key][-average_over:] for run_id, (_, run_metrics) in metrics.items()}


def cmap() -> typing.List[str]:
    return DEFAULT_COLORMAP.colors


def errorbar_legend(
        ax: typing.Union[matplotlib.pyplot.Axes, matplotlib.pyplot.Figure],
        handles: typing.List = None,
        labels: typing.List = None,
        **kwargs
):
    if handles is None and labels is None:
        handles, labels = ax.get_legend_handles_labels()
    elif handles is None or labels is None:
        raise ValueError('If either handles or labels is specified, the other needs to be provided too')

    # Extract lines only from errorbar handles in order to avoid rendering the bars in the legend
    def process_handle(handle):
        if isinstance(handle, matplotlib.container.ErrorbarContainer):
            return handle[0]
        else:
            return handle

    handles = list(map(process_handle, handles))
    ax.legend(
        handles,
        labels,
        **kwargs
    )
