"""Unified visualization for typhoon precipitation fields.

Provides Cartopy-based map rendering with consistent styling suitable
for publication-quality figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import patheffects
from scipy.ndimage import gaussian_filter
from typing import Optional, Tuple, List, Dict

# Try to import cartopy (optional — graceful fallback)
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.mpl.ticker as cticker
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

# Precipitation colormap (white-blue-green-yellow-orange-red)
_PRECIP_COLORS = [
    '#ffffff', '#bde9bf', '#adddb0', '#9ed0a0', '#8ec491',
    '#7fb882', '#70ac74', '#60a065', '#519457', '#418849',
    '#307c3c', '#1c712e', '#f7f370', '#fbdf65', '#fecb5a',
    '#ffb650', '#ffa146', '#ff8b3c', '#ff6b3c', '#ff4b3c',
]
PRECIP_CMAP = mcolors.ListedColormap(_PRECIP_COLORS)

# Precipitation contour levels (mm/h)
PRECIP_LEVELS = np.concatenate([
    np.arange(0.1, 1, 0.1),
    np.arange(1, 2, 0.2),
    np.arange(2, 10, 1),
    np.arange(10, 60, 10),
])
PRECIP_NORM = mcolors.BoundaryNorm(PRECIP_LEVELS, len(_PRECIP_COLORS))

OUTLINE_EFFECT = [
    patheffects.withStroke(linewidth=2.5, foreground='white')
]


def set_style():
    """Set global matplotlib rcParams for consistent paper-quality figures."""
    plt.rcParams.update({
        'axes.linewidth': 1.5,
        'font.size': 10,
        'figure.dpi': 150,
        'figure.facecolor': 'white',
        'mathtext.fontset': 'dejavuserif',
        'font.family': 'sans-serif',
    })


# ---------------------------------------------------------------------------
# Main plotting functions
# ---------------------------------------------------------------------------

def plot_precipitation_field(
    precipitation: np.ndarray,
    center_lon: float = 0.0,
    center_lat: float = 0.0,
    radius_deg: float = 5.75,
    extent: Tuple[float, float, float, float] = None,
    ax: plt.Axes = None,
    title: str = None,
    add_colorbar: bool = True,
    smooth_sigma: float = 1.2,
    cmap=None,
    norm=None,
    figsize: Tuple[int, int] = (11, 8),
    track_lons: List[float] = None,
    track_lats: List[float] = None,
    label_text: str = None,
) -> plt.Axes:
    """Plot a precipitation field on a geographic map.

    Args:
        precipitation: [H, W] precipitation array (mm/h).
        center_lon, center_lat: Typhoon center coordinates.
        radius_deg: Half-width of local grid in degrees.
        extent: Map extent [lon_min, lon_max, lat_min, lat_max].
        ax: Existing axis (creates new if None).
        title: Plot title.
        add_colorbar: Add precipitation colorbar.
        smooth_sigma: Gaussian smoothing sigma (0 = no smoothing).
        cmap: Colormap override.
        norm: Normalization override.
        figsize: Figure size.
        track_lons, track_lats: Track line to overlay.
        label_text: Text annotation at center (e.g., pressure).

    Returns:
        Matplotlib axis.
    """
    if cmap is None:
        cmap = PRECIP_CMAP
    if norm is None:
        norm = PRECIP_NORM

    # Smooth
    if smooth_sigma > 0:
        precip = gaussian_filter(precipitation, sigma=smooth_sigma)
    else:
        precip = precipitation

    # Geographic coordinates
    if extent is None:
        extent = [
            center_lon - 8, center_lon + 8,
            center_lat - 8, center_lat + 8,
        ]

    lon_1d = np.linspace(
        center_lon - radius_deg, center_lon + radius_deg, precip.shape[1]
    )
    lat_1d = np.linspace(
        center_lat - radius_deg, center_lat + radius_deg, precip.shape[0]
    )
    lon_mesh, lat_mesh = np.meshgrid(lon_1d, lat_1d)

    # Create figure / axis
    if ax is None:
        fig = plt.figure(figsize=figsize)
        if HAS_CARTOPY:
            ax = plt.axes(projection=ccrs.PlateCarree())
        else:
            ax = plt.gca()

    # Map features
    if HAS_CARTOPY:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(
            cfeature.NaturalEarthFeature('physical', 'land', '50m'),
            facecolor='#f0f0f0', zorder=-1, alpha=0.8,
        )
        ax.add_feature(cfeature.OCEAN, facecolor='#e0f3f8', zorder=-1)
        ax.coastlines(linewidth=1.2, color='#333333')

    # Precipitation contour fill
    cf = ax.contourf(
        lon_mesh, lat_mesh, precip,
        PRECIP_LEVELS, norm=norm, cmap=cmap,
        transform=ccrs.PlateCarree() if HAS_CARTOPY else None,
        extend='max',
    )

    # Typhoon center marker
    if HAS_CARTOPY:
        transform = ccrs.PlateCarree()
    else:
        transform = None

    ax.text(
        center_lon, center_lat, 'L',
        color='red', size=24, ha='center', va='center',
        transform=transform,
        path_effects=OUTLINE_EFFECT,
    )

    if label_text:
        ax.text(
            center_lon, center_lat - 0.8, label_text,
            color='red', size=11, ha='center', va='top',
            fontweight='bold', transform=transform,
            path_effects=OUTLINE_EFFECT,
        )

    # Track overlay
    if track_lons is not None and track_lats is not None:
        ax.plot(
            track_lons, track_lats,
            color='blue', linewidth=1.5, linestyle='--',
            transform=transform, alpha=0.7,
        )

    # Grid labels
    if HAS_CARTOPY:
        ax.set_xticks(
            np.arange(extent[0], extent[1] + 1, 5),
            crs=ccrs.PlateCarree(),
        )
        ax.set_yticks(
            np.arange(extent[2], extent[3] + 1, 5),
            crs=ccrs.PlateCarree(),
        )
        ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
        ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.minorticks_on()
    ax.tick_params(
        which='major', length=6, width=1.2,
        top=True, right=True, direction='in',
    )

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    if add_colorbar:
        cbar = plt.colorbar(
            cf, ax=ax, orientation='vertical',
            pad=0.02, shrink=0.8,
        )
        cbar.set_label('Precipitation (mm/h)', fontweight='bold')

    return ax


def plot_comparison(
    P_pred: np.ndarray,
    P_true: np.ndarray,
    center_lon: float,
    center_lat: float,
    save_path: str = None,
    title: str = "Prediction vs Ground Truth",
    figsize: Tuple[int, int] = (24, 7),
):
    """Side-by-side comparison: prediction, ground truth, error.

    Args:
        P_pred: [H, W] predicted precipitation.
        P_true: [H, W] ground truth precipitation.
        center_lon, center_lat: Typhoon center.
        save_path: Output file path.
        title: Overall figure title.
        figsize: Figure dimensions.
    """
    if HAS_CARTOPY:
        fig, axes = plt.subplots(
            1, 3, figsize=figsize,
            subplot_kw={'projection': ccrs.PlateCarree()},
        )
    else:
        fig, axes = plt.subplots(1, 3, figsize=figsize)

    radius_deg = 5.75

    # Prediction
    plot_precipitation_field(
        P_pred, center_lon, center_lat, radius_deg,
        ax=axes[0], title='Prediction', add_colorbar=False,
    )

    # Ground truth
    plot_precipitation_field(
        P_true, center_lon, center_lat, radius_deg,
        ax=axes[1], title='Ground Truth', add_colorbar=False,
    )

    # Error
    error = P_pred - P_true
    vmax = max(abs(error.min()), abs(error.max()), 1e-6)
    plot_precipitation_field(
        error, center_lon, center_lat, radius_deg,
        ax=axes[2], title='Error (Pred − True)',
        cmap=plt.cm.RdBu_r,
        norm=mcolors.Normalize(vmin=-vmax, vmax=vmax),
        add_colorbar=False,
    )

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")

    plt.close()


def plot_metric_bars(
    metrics: Dict[str, Dict[str, float]],
    metric_names: List[str],
    save_path: str = None,
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (12, 6),
):
    """Grouped bar chart comparing models across metrics.

    Args:
        metrics: {model_name: {metric_name: value}}.
        metric_names: List of metric names to plot.
        save_path: Output file path.
        title: Plot title.
        figsize: Figure dimensions.
    """
    model_names = list(metrics.keys())
    n_models = len(model_names)
    n_metrics = len(metric_names)

    x = np.arange(n_models)
    width = 0.8 / n_metrics
    colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))

    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    for i, metric in enumerate(metric_names):
        values = [metrics[m].get(metric, float('nan')) for m in model_names]
        ax.bar(
            x + i * width, values, width,
            label=metric, color=colors[i], alpha=0.85,
        )

    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(loc='best')
    ax.set_ylabel('Value')
    ax.set_title(title, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
    plt.close()
