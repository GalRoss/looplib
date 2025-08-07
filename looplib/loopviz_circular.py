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
    height_factor=0.005,
    max_bend=0.5,
    bend=None
):
    """
    Plot a single loop as a Bezier curve on a circular genome, using height_factor to control curvature.
    """
    # Convert genome positions to angles
    angle_l = pos_to_angle(l, genome_size)
    angle_r = pos_to_angle(r, genome_size)

    # Handle wraparound for circular genome
    if angle_r < angle_l:
        angle_r += 2 * np.pi

    # Calculate loop length in genomic units
    loop_length = (r - l) % genome_size

    # Determine curvature (bend_outward)
    bend_outward = bend if bend is not None else min(max_bend, loop_length * height_factor)

    # Coordinates on the genome circle
    x0, y0 = angle_to_xy(angle_l, radius, center)
    x1, y1 = angle_to_xy(angle_r, radius, center)

    # Control point for Bezier curve
    angle_mid = (angle_l + angle_r) / 2
    ctrl_radius = radius + bend_outward
    cx, cy = angle_to_xy(angle_mid, ctrl_radius, center)

    # Draw Bezier curve
    path_data = [
        (Path.MOVETO, (x0, y0)),
        (Path.CURVE3, (cx, cy)),
        (Path.CURVE3, (x1, y1))
    ]
    path = Path([p for _, p in path_data], [c for c, _ in path_data])
    patch = PathPatch(path, facecolor='none', edgecolor=color, lw=lw, alpha=alpha)
    ax.add_patch(patch)

def plot_circular_genome_with_loops(
    l_sites,
    r_sites,
    genome_size,
    radius=1.0,
    center=(0, 0),
    colors='tomato',
    height_factor=0.005,
    max_bend=0.5,
    lw=2,
    alpha=0.6,
    ax=None  # <-- Add this
):
    """
    Plot a circular genome with multiple loops, using height_factor to control loop curvature.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))  # Only create if not passed

    draw_genome_circle(ax, radius=radius, center=center)

    # Ensure colors is iterable
    if isinstance(colors, str):
        colors = [colors] * len(l_sites)

    l_sites = np.asarray(l_sites).flatten()
    r_sites = np.asarray(r_sites).flatten()

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
            height_factor=height_factor,
            max_bend=max_bend
        )

    if ax is None:
        plt.show()

