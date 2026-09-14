#!/usr/bin/env python3
"""Scene2 double-bottleneck layout visualisation.

Generates a publication-quality figure of the Scene2 world showing:
  - Boundary walls (10×10 m enclosure) and six interior obstacles
  - Three bottleneck zones (upper, lower, central) as translucent rectangles
  - Two representative head-on evaluation templates with start/goal markers
  - Passage direction annotations

Output: scene2_layout.png  (raster)
        scene2_layout.pdf  (vector, for papers)

Usage:
    python3 vis_scene2_layout.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

# ---------------------------------------------------------------------------
# Global style — tuned for a clean paper figure.
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.figsize": (10, 9),
    "figure.dpi": 150,
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ---------------------------------------------------------------------------
# 1. Boundary walls (extracted from 10by10/wall_model.sdf)
# ---------------------------------------------------------------------------
# Each wall is a box 10 × 0.15 m, positioned to form a ~9.85 m square enclosure.
WALL_SPECS = [
    # (x_center, y_center, width, height, label)
    (-4.925,  0.000,  0.15, 10.0,  None),   # Wall_0 — left   (vertical)
    ( 4.925,  0.000,  0.15, 10.0,  None),   # Wall_2 — right  (vertical)
    ( 0.000,  4.925, 10.0,  0.15,  None),   # Wall_1 — top    (horizontal)
    ( 0.000, -4.925, 10.0,  0.15,  None),   # Wall_3 — bottom (horizontal)
]

# Interior free-space bounds (inner wall faces).
INTERIOR_XLIM = (-4.85, 4.85)
INTERIOR_YLIM = (-4.85, 4.85)

# ---------------------------------------------------------------------------
# 2. Interior obstacles (extracted from .world file)
# ---------------------------------------------------------------------------
# Each entry: type, (cx, cy), (w, h) or radius, color, label
OBSTACLES = [
    # Upper route
    ("box",    (-2.70,  2.75),  (1.15, 0.90),  "obstacle1\n(upper-left)"),
    ("circle", ( 2.30,  2.70),  0.55,           "obstacle2\n(upper-right)"),
    # Lower route
    ("circle", (-2.30, -2.70),  0.55,           "obstacle3\n(lower-left)"),
    ("box",    ( 2.70, -2.75),  (1.15, 0.90),  "obstacle4\n(lower-right)"),
    # Central S-passage
    ("box",    (-1.10,  0.50),  (0.55, 1.20),  "obstacle5\n(central-left)"),
    ("box",    ( 1.10, -0.50),  (0.55, 1.20),  "obstacle6\n(central-right)"),
]

# ---------------------------------------------------------------------------
# 3. Bottleneck zones (from environment code is_in_bottleneck_zone_xy)
# ---------------------------------------------------------------------------
BOTTLENECK_ZONES = [
    # (x_min, x_max, y_min, y_max, label, facecolor, edgecolor)
    (-3.45, 3.45,  1.15,  2.35,  "upper passage\n(bottleneck)",   "#FFE4B5", "#CD853F"),
    (-3.45, 3.45, -2.35, -1.15,  "lower passage\n(bottleneck)",   "#B5E8FF", "#4682B4"),
    (-2.15, 2.15, -1.65,  1.65,  "central S-passage\n(bottleneck)", "#E0FFE0", "#6B8E23"),
]

# ---------------------------------------------------------------------------
# 4. Head-on eval templates (representative subset)
# ---------------------------------------------------------------------------
# Each template: [(start0_x, start0_y), (start1_x, start1_y),
#                  (goal0_x,  goal0_y),  (goal1_x,  goal1_y)]
TEMPLATES = {
    "Template A: upper-passage\nexplicit head-on": [
        (-3.40, 1.45), (3.40, 1.45),   # starts
        ( 1.55, 1.45), (-1.55, 1.45),  # goals
    ],
    "Template B: central-passage\nvertical head-on": [
        (0.00,  3.40), (0.00, -3.40),   # starts
        (0.00, -1.55), (0.00,  1.55),   # goals
    ],
}

# Colours for the two robots in each template.
ROBOT_COLORS = ["#D62728", "#1F77B4"]   # red, blue

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def add_walls(ax):
    """Draw boundary walls as filled grey rectangles."""
    for cx, cy, w, h, _label in WALL_SPECS:
        rect = mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0",
            facecolor="#CCCCCC", edgecolor="#888888", linewidth=1.0, zorder=1,
        )
        ax.add_patch(rect)


def add_obstacle_box(ax, cx, cy, w, h):
    """Draw a rectangular obstacle."""
    rect = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0",
        facecolor="#E8D5B7", edgecolor="#8B7355", linewidth=1.2, zorder=3,
    )
    ax.add_patch(rect)


def add_obstacle_circle(ax, cx, cy, r):
    """Draw a cylindrical obstacle."""
    circle = mpatches.Circle(
        (cx, cy), r,
        facecolor="#E8D5B7", edgecolor="#8B7355", linewidth=1.2, zorder=3,
    )
    ax.add_patch(circle)


def add_obstacles(ax):
    """Draw all six interior obstacles."""
    for spec in OBSTACLES:
        kind = spec[0]
        if kind == "box":
            cx, cy = spec[1]
            w, h = spec[2]
            add_obstacle_box(ax, cx, cy, w, h)
        elif kind == "circle":
            cx, cy = spec[1]
            r = spec[2]
            add_obstacle_circle(ax, cx, cy, r)


def add_zones(ax):
    """Draw the three bottleneck zones as translucent rectangles."""
    for x0, x1, y0, y1, label, fc, ec in BOTTLENECK_ZONES:
        rect = mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            facecolor=fc, edgecolor=ec, linewidth=1.5, linestyle="--",
            alpha=0.35, zorder=2,
        )
        ax.add_patch(rect)


def add_template(ax, starts, goals, label, robot_colors):
    """Add one head-on template: start markers, goal markers, arrows."""
    rcolors = robot_colors
    for ri in range(2):
        sx, sy = starts[ri]
        gx, gy = goals[ri]

        # Start: filled circle.
        ax.plot(sx, sy, marker="o", color=rcolors[ri], markersize=10,
                markeredgecolor="white", markeredgewidth=1.0, zorder=5)

        # Goal: star.
        ax.plot(gx, gy, marker="*", color=rcolors[ri], markersize=14,
                markeredgecolor=rcolors[ri], markeredgewidth=0.8, zorder=5)

        # Arrow connecting start → goal (gentle curve for readability).
        ax.annotate(
            "", xy=(gx, gy), xytext=(sx, sy),
            arrowprops=dict(
                arrowstyle="->", color=rcolors[ri], lw=2.0,
                connectionstyle=f"arc3,rad={0.08 if ri == 0 else -0.08}",
                alpha=0.85,
            ),
            zorder=4,
        )

    # Thin dashed line connecting the two starts to hint at the passage axis.
    ax.plot([starts[0][0], starts[1][0]], [starts[0][1], starts[1][1]],
            color="grey", linewidth=0.6, linestyle=":", alpha=0.5, zorder=2)

    # Add template label as text annotation near the passage zone.
    if label:
        mid_x = (starts[0][0] + starts[1][0]) / 2
        mid_y = (starts[0][1] + starts[1][1]) / 2
        ax.annotate(
            label, xy=(mid_x, mid_y),
            fontsize=7, color="#444444", fontstyle="italic",
            ha="center", va="center", zorder=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#AAAAAA", alpha=0.85),
        )


# ---------------------------------------------------------------------------
# Build the figure
# ---------------------------------------------------------------------------
def main():
    fig, ax = plt.subplots()

    # ---- Draw world elements ----
    add_walls(ax)
    add_obstacles(ax)
    add_zones(ax)

    # ---- Draw templates ----
    for tname, coords in TEMPLATES.items():
        s0, s1, g0, g1 = coords
        add_template(ax, [s0, s1], [g0, g1], tname, ROBOT_COLORS)

    # ---- Annotations ----
    # Label obstacle numbers.
    obstacle_labels = {
        "obstacle1": (-2.70, 2.75, "1"),
        "obstacle2": ( 2.30, 2.70, "2"),
        "obstacle3": (-2.30, -2.70, "3"),
        "obstacle4": ( 2.70, -2.75, "4"),
        "obstacle5": (-1.10, 0.50, "5"),
        "obstacle6": ( 1.10, -0.50, "6"),
    }
    for key, (ox, oy, num) in obstacle_labels.items():
        ax.annotate(num, (ox, oy), color="black", fontsize=8, fontweight="bold",
                    ha="center", va="center", zorder=6,
                    bbox=dict(boxstyle="circle,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.85))

    # Label passages with arrows.
    passage_annotations = [
        ("upper passage\n(horizontal)", (0, 1.75), (0, 1)),
        ("lower passage\n(horizontal)", (0, -1.75), (0, -1)),
        ("central S-passage\n(vertical)", (0, 0), (0, 1)),
    ]
    for text, xy, direction in passage_annotations:
        ax.annotate(text, xy=xy, fontsize=9, fontstyle="italic", color="#555555",
                    ha="center", va="center", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="none", alpha=0.80))

    # Mark entry directions on edges.
    edge_markers = [
        ("←  left\nentrance", (-4.60,  0.0), "left"),
        ("right  →\nentrance", ( 4.60,  0.0), "right"),
        ("top entrance  ↓", ( 0.0,  4.60), "top"),
        ("↑  bottom\nentrance", ( 0.0, -4.60), "bottom"),
    ]
    for text, xy, _side in edge_markers:
        ax.annotate(text, xy=xy, fontsize=8, color="#777777",
                    ha="center", va="center", zorder=7)

    # ---- Legend ----
    legend_elements = [
        mpatches.Patch(facecolor="#CCCCCC", edgecolor="#888888",
                       label="Boundary wall (thickness 0.15 m)"),
        mpatches.Patch(facecolor="#E8D5B7", edgecolor="#8B7355",
                       label="Obstacle (box / cylinder)"),
        mpatches.Patch(facecolor="#FFE4B5", edgecolor="#CD853F", alpha=0.35,
                       linestyle="--", label="Bottleneck zone (upper / lower / central)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
               markersize=9, label="Robot start position"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="grey",
               markersize=12, label="Robot goal position"),
        Line2D([0], [0], color="black", lw=2.0,
               label="Navigation path (start → goal)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left",
              bbox_to_anchor=(1.005, 1.0), framealpha=0.9, edgecolor="#CCCCCC")

    # ---- Axis setup ----
    ax.set_xlim(-5.3, 5.3)
    ax.set_ylim(-5.3, 5.3)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        "Scene2 Double-Bottleneck Layout\n"
        "Obstacles, Bottleneck Zones & Head-on Evaluation Templates",
        fontweight="bold", pad=12,
    )
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.set_xticks(np.arange(-5, 6, 1))
    ax.set_yticks(np.arange(-5, 6, 1))

    # ---- Save ----
    for ext in ("png", "pdf"):
        fname = f"scene2_layout.{ext}"
        fig.savefig(fname, dpi=300 if ext == "png" else None)
        print(f"✅ Saved {fname}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
