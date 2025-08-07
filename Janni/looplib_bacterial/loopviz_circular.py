import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Circle
from matplotlib.path import Path

def pos_to_angle(pos, genome_size):
    """
    Convert genome position to angle in radians.
    0 position starts at top (90 degrees), and increases clockwise.
    """
    return 2 * np.pi * (pos / genome_size)

def angle_to_xy(angle, radius, center=(0, 0)):
    """
    Convert angle and radius to x, y coordinates.
    """
    x = center[0] + radius * np.cos(angle - np.pi / 2)  # Start at top
    y = center[1] + radius * np.sin(angle - np.pi / 2)
    return x, y

def draw_genome_circle(ax, radius=1.0, center=(0, 0)):
    """
    Draw the circular genome as a ring.
    """
    circle = Circle(center, radius, color='gray', lw=3, fill=False)
    ax.add_artist(circle)

    ax.set_aspect('equal')
    ax.set_xlim(center[0] - radius - 0.5, center[0] + radius + 0.5)
    ax.set_ylim(center[1] - radius - 0.5, center[1] + radius + 0.5)
    ax.axis('off')

def plot_circular_loop(
    ax,
    l,
    r,
    genome_size,
    radius=1.0,
    center=(0, 0),
    color='tomato',
    alpha=0.6,
    lw=2,
    bend_outward=0.3
):
    """
    Plot a single loop as a Bezier curve along the circular genome.
    """
    # Convert genome positions to angles
    angle_l = pos_to_angle(l, genome_size)
    angle_r = pos_to_angle(r, genome_size)

    # Handle wraparound (for circularity)
    if angle_r < angle_l:
        angle_r += 2 * np.pi

    # Coordinates on the circle
    x0, y0 = angle_to_xy(angle_l, radius, center)
    x1, y1 = angle_to_xy(angle_r, radius, center)

    # Control point for curvature (mid-angle, farther out)
    angle_mid = (angle_l + angle_r) / 2
    ctrl_radius = radius + bend_outward
    cx, cy = angle_to_xy(angle_mid, ctrl_radius, center)

    # Create Bezier curve path
    path_data = [
        (Path.MOVETO, (x0, y0)),
        (Path.CURVE3, (cx, cy)),
        (Path.CURVE3, (x1, y1))
    ]
    codes, verts = zip(*path_data)
    path = Path(verts, codes)
    patch = PathPatch(path, facecolor='none', edgecolor=color, lw=lw, alpha=alpha)
    ax.add_patch(patch)

def plot_circular_genome_with_loops(
    l_sites,
    r_sites,
    genome_size,
    radius=1.0,
    center=(0, 0),
    colors='tomato',
    bend_outward=0.3,
    lw=2,
    alpha=0.6
):
    """
    Plot a circular genome with multiple loops.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_genome_circle(ax, radius=radius, center=center)

    if isinstance(colors, str):
        colors = [colors] * len(l_sites)

    for l, r, color in zip(l_sites, r_sites, colors):
        plot_circular_loop(
            ax=ax,
            l=l,
            r=r,
            genome_size=genome_size,
            radius=radius,
            center=center,
            color=color,
            alpha=alpha,
            lw=lw,
            bend_outward=bend_outward
        )

    plt.show()
