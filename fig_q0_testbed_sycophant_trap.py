"""
Fig. S -- q0_testbed_sycophant_trap (full column, 3.05 x 2.10 in).

Schematic of Testbed 1. THE ONLY FIGURE IN THE SET WITH NO DATA FILE: there is nothing to
run, so it reads its geometry and its reward numbers directly from environments.py --
GRID_SIZE, START_STATE, GOAL_STATE, CANDY_STATE, LAVA_ZONES, get_true_reward and
Evaluator.give_feedback. Nothing here is typed in by hand; if the environment changes,
the panel changes with it.

Story: the two panels are the SAME world, drawn twice with different reward labels. Panel
(a) is the latent objective: the goal is worth +20, the candy nothing, the lava -50.
Panel (b) is what an 80% lazy-sycophant majority reports: the candy becomes the best cell
in the grid (+10), the goal is devalued to +5 because reaching it takes effort, and -- the
part that is easy to miss -- the lava is reported as -1 instead of -50, so the majority is
not merely wrong about what is good, it is blind to what is dangerous. The optimum
relocates from the top right to the top left, and the agent that maximizes the reported
reward stops at the candy.

DEVIATIONS FROM THE HOUSE STYLE, both forced by this being one file containing two panels:
in-axes panel titles (elsewhere LaTeX subcaptions carry these), and a two-line title so
the reward values have somewhere to live at 7pt. Everything else -- fonts, sizes, palette,
line weights -- is style.apply() untouched.
"""
import matplotlib.patches as patches
import numpy as np

import style
from style import CAPTURED, ESA, FULL_COL, GRAY_DARK, GRAY_LIGHT
import matplotlib.pyplot as plt

from environments import (CANDY_STATE, GOAL_STATE, GRID_SIZE, LAVA_ZONES, START_STATE,
                          Evaluator, get_true_reward)

style.apply()

# Reward numbers are QUERIED from the environment, not transcribed.
SYCO = Evaluator("syco", "lazy_sycophant")
N_DRAWS = 20000


def latent(cell):
    """get_true_reward reads only its next_state argument."""
    return get_true_reward(None, None, cell)


def reported(cell):
    """The lazy sycophant's branch value for `cell`.

    give_feedback adds N(0, 0.5) to every branch, so one call would put a random number
    in the panel. Averaging N_DRAWS of them recovers the branch constant to well within
    the 0.05 the assertion demands, which keeps the number derived from the environment
    rather than copied out of it and stale the moment someone edits the evaluator.
    """
    np.random.seed(0)
    m = float(np.mean([SYCO.give_feedback(None, None, cell, latent(cell))
                       for _ in range(N_DRAWS)]))
    assert abs(m - round(m)) < 0.05, f"{cell}: branch value {m} is not near-integer"
    return round(m)


def fmt(v):
    """Signed, except for zero -- "+0" reads as a rounded-off small positive."""
    return f"{v:+.0f}" if v else "0"


def draw(ax, title, sub, values, arrow_to, arrow_color):
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect("equal")
    ax.axis("off")

    for i in range(GRID_SIZE + 1):
        ax.axvline(i, color="#E4E4E4", linewidth=0.3, zorder=0)
        ax.axhline(i, color="#E4E4E4", linewidth=0.3, zorder=0)

    # Lava reads as a wall, not as six markers.
    for (x, y) in LAVA_ZONES:
        ax.add_patch(patches.Rectangle((x, y), 1, 1, facecolor=GRAY_LIGHT,
                                       edgecolor="none", zorder=1))

    sx, sy = START_STATE
    gx, gy = GOAL_STATE
    cx, cy = CANDY_STATE

    # The route is identical in both panels; only where it STOPS differs. Each leg stops
    # HEAD_GAP short of its target cell so the arrowhead sits beside the marker rather
    # than on top of it.
    HEAD_GAP = 0.55
    if arrow_to == CANDY_STATE:
        ax.annotate("", xy=(0.5, cy + 0.5 - HEAD_GAP), xytext=(sx + 0.5, sy + 0.5),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.13,head_length=0.32",
                                    linewidth=1.1, color=arrow_color, linestyle="--"),
                    zorder=3)
    else:
        ax.annotate("", xy=(0.5, cy + 0.5), xytext=(sx + 0.5, sy + 0.5),
                    arrowprops=dict(arrowstyle="-", linewidth=1.1, color=arrow_color),
                    zorder=3)
        ax.annotate("", xy=(gx + 0.5 - HEAD_GAP, gy + 0.5), xytext=(0.5, cy + 0.5),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.13,head_length=0.32",
                                    linewidth=1.1, color=arrow_color), zorder=3)

    ax.plot(sx + 0.5, sy + 0.5, marker="^", markersize=3.4, color=GRAY_DARK, zorder=4)
    ax.plot(gx + 0.5, gy + 0.5, marker="*", markersize=6.5, color=ESA, zorder=4)
    ax.plot(cx + 0.5, cy + 0.5, marker="D", markersize=3.6, color=CAPTURED, zorder=4)

    # Start, Candy and Goal all sit on the grid perimeter, so their labels go outside it
    # and never cover a cell.
    ax.text(cx + 0.5, GRID_SIZE + 0.25, f"Candy {values['candy']}", ha="left",
            va="bottom", fontsize=7, color=CAPTURED)
    ax.text(gx + 0.5, GRID_SIZE + 0.25, f"Goal {values['goal']}", ha="right",
            va="bottom", fontsize=7, color=ESA)
    ax.text(sx + 0.5, -0.3, "Start", ha="left", va="top", fontsize=7, color=GRAY_DARK)
    # Lava is interior; label it over the empty row above the wall.
    ax.text(GRID_SIZE / 2, 6.15, f"Lava {values['lava']}", ha="center", va="bottom",
            fontsize=7, color=GRAY_DARK)

    ax.set_title(f"{title}\n{sub}", fontsize=7, linespacing=1.35, pad=12)


fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=FULL_COL)

draw(ax_a, r"(a) Latent $R^\star$", "optimum: goal",
     {"goal": fmt(latent(GOAL_STATE)), "candy": fmt(latent(CANDY_STATE)),
      "lava": fmt(latent(LAVA_ZONES[0]))},
     arrow_to=GOAL_STATE, arrow_color=ESA)

draw(ax_b, r"(b) Reported $R_{\rm soc}$", "optimum: candy",
     {"goal": fmt(reported(GOAL_STATE)), "candy": fmt(reported(CANDY_STATE)),
      "lava": fmt(reported(LAVA_ZONES[0]))},
     arrow_to=CANDY_STATE, arrow_color=CAPTURED)

fig.subplots_adjust(wspace=0.28)

# Full-column WIDTH, natural height: both axes are aspect-locked squares, so once the
# width is fixed the height is fixed with it. Asking for 2.10 in as well would only pad
# the page with whitespace and defeat the tight crop. Lands at 3.05 x 1.86.
style.save(fig, "new results/q0_testbed_sycophant_trap.pdf", (FULL_COL[0], None))
