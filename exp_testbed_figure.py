"""
v2 rewrite of generate_testbed_figure.py -- schematic of Testbed 1 (the Sycophant
Trap) drawn to match the ACTUAL v2 environment in environments.py:
  GRID 10x10, START (0,0), GOAL (9,9)=+20, CANDY (0,9)=0 true / +10 social(fake),
  LAVA = row y=5 for x in 2..7 (=-50). (v1's figure used a different layout.)

Two panels: (a) latent objective R* and (b) social proxy R_soc, showing how the
sycophantic majority relocates the apparent optimum to the Candy cell. Pure
matplotlib, no compute.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from environments import GRID_SIZE, START_STATE, GOAL_STATE, CANDY_STATE, LAVA_ZONES


def draw_grid(ax, title, cells, arrows):
    ax.set_title(title, fontsize=14, pad=12, fontweight="bold", family="serif")
    ax.set_xlim(0, GRID_SIZE); ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect("equal"); ax.axis("off")
    for i in range(GRID_SIZE + 1):
        ax.axvline(i, color="#D3D3D3", lw=0.7)
        ax.axhline(i, color="#D3D3D3", lw=0.7)
    for (x, y, bg, shape, label, val) in cells:
        ax.add_patch(patches.Rectangle((x, y), 1, 1, edgecolor="#E0E0E0", facecolor=bg))
        cx, cy = x + 0.5, y + 0.5
        if shape == "agent":
            ax.add_patch(patches.RegularPolygon((cx, cy), numVertices=3, radius=0.3, color="#1565C0"))
        elif shape == "goal":
            ax.plot(cx, cy, marker="*", markersize=22, color="#2E7D32", mec="white", mew=1)
        elif shape == "candy":
            ax.plot(cx, cy, marker="D", markersize=13, color="#F9A825", mec="white", mew=1)
        elif shape == "lava":
            ax.plot(cx, cy, marker="x", markersize=11, color="#C62828", mew=2.5)
        if label:
            ax.text(cx, y + 0.82, label, ha="center", va="center", fontsize=10,
                    fontweight="bold", color="#333", family="serif")
        if val:
            ax.text(cx, y + 0.2, val, ha="center", va="center", fontsize=10,
                    color="#444", family="serif")
    for (start, end, style, color, label) in arrows:
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.4",
                                    lw=2.5, color=color, ls=style))
        if label:
            ax.text((start[0]+end[0])/2, (start[1]+end[1])/2, label, ha="center",
                    fontsize=9, color=color, fontweight="bold", family="serif",
                    bbox=dict(facecolor="white", edgecolor="none", pad=2, alpha=0.9))


def main():
    lava = [(x, y, "#FFEBEE", "lava", "", "") for (x, y) in LAVA_ZONES]
    gx, gy = GOAL_STATE
    cx, cy = CANDY_STATE
    sx, sy = START_STATE

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    cells_a = [(sx, sy, "#F5F5F5", "agent", "Start", ""),
               (gx, gy, "#E8F5E9", "goal", "Goal", "+20"),
               (cx, cy, "#FFFFFF", "candy", "Candy", "0")] + lava
    # robust path: up the clear left edge (x=0), across the top to the goal
    arrows_a = [((sx+0.5, sy+0.5), (0.5, gy+0.5), "-", "#2E7D32", ""),
                ((0.5, gy+0.5), (gx+0.5, gy+0.5), "-", "#2E7D32", "Robust path")]
    draw_grid(ax1, r"(a) Latent Objective $R^*$", cells_a, arrows_a)

    cells_b = [(sx, sy, "#F5F5F5", "agent", "Start", ""),
               (gx, gy, "#E8F5E9", "goal", "Goal", "+20"),
               (cx, cy, "#FFFDE7", "candy", "Candy", "+10 (fake)")] + lava
    arrows_b = [((sx+0.5, sy+0.5), (cx+0.5, cy+0.5), "--", "#D32F2F", "Sycophantic detour")]
    draw_grid(ax2, r"(b) Social Proxy $R_{soc}$ (80% sycophants)", cells_b, arrows_b)

    plt.tight_layout()
    os.makedirs("paper/figures", exist_ok=True)
    plt.savefig("paper/figures/q0_testbed1_sycophant_trap.png", dpi=200, bbox_inches="tight")
    print("saved paper/figures/q0_testbed1_sycophant_trap.png")


if __name__ == "__main__":
    main()
