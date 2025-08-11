import json
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Button, Slider
import bisect


def load_debug_data() -> dict:
    """Load debugData.json from competition_code/debugData/ or current folder."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "debugData", "debugData.json"),
        os.path.join(script_dir, "debugData.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    raise FileNotFoundError(
        f"Could not find debugData.json. Tried: {candidates}"
    )


def to_points(debug_dict: dict):
    """Convert debug dict keyed by ticks -> iterable of (lap, x, y, speed, section, section_ticks, tick_idx)."""
    points = []
    # Sort by numeric tick key to preserve temporal order
    items = []
    for k, v in debug_dict.items():
        # ignore non-tick metadata like "meta"
        if isinstance(k, str) and k.isdigit() and isinstance(v, dict):
            items.append((int(k), v))
    items.sort(key=lambda kv: kv[0])

    for tick_idx, v in items:
        loc = v.get("loc")
        speed = v.get("speed")
        lap = v.get("lap")
        section = v.get("section")
        section_ticks = v.get("section_ticks")
        if (
            isinstance(loc, (list, tuple))
            and len(loc) >= 2
            and isinstance(speed, (int, float))
            and isinstance(lap, (int, float))
        ):
            x, y = loc[0], loc[1]
            points.append(
                (
                    int(lap),
                    float(x),
                    float(y),
                    float(speed),
                    int(section) if isinstance(section, (int, float)) else None,
                    int(section_ticks) if isinstance(section_ticks, (int, float)) else None,
                    int(tick_idx),
                )
            )
    return points


def main():
    debug_dict = load_debug_data()
    points = to_points(debug_dict)
    if not points:
        print("No points with loc/speed/lap found in debugData.json")
        return

    # Group points by lap
    laps = sorted({p[0] for p in points})
    lap_to_points = {lap: [p for p in points if p[0] == lap] for lap in laps}
    if not laps:
        print("No laps found in debug data.")
        return

    # Build interactive figure
    fig, ax = plt.subplots(figsize=(11, 11))
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.18)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.axis((-1100, 1100, -1100, 1100))

    colors6 = ["purple", "blue", "green", "yellow", "orange", "red"]
    cmap = mcolors.ListedColormap(colors6)

    current_idx = 0
    sec_markers = []
    sec_annots = []
    cbar = None

    def compute_boundaries(_speeds_arr: np.ndarray):
        # Fixed bands from 0 to 300 km/h into 6 bins
        boundaries = np.linspace(0.0, 300.0, 7)
        norm_local = mcolors.BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)
        # Ticks at bin midpoints; last label uses '+' to indicate open-ended
        mids = (boundaries[:-1] + boundaries[1:]) / 2.0
        labels = [f"{boundaries[i]:.0f}–{boundaries[i+1]:.0f}" for i in range(len(boundaries) - 2)]
        labels.append(f"{boundaries[-2]:.0f}+")
        return boundaries, norm_local, mids, labels

    # Initialize with first lap
    init_lap = laps[current_idx]
    init_pts = lap_to_points.get(init_lap, [])
    x_coords = np.array([p[1] for p in init_pts])
    y_coords = np.array([p[2] for p in init_pts])
    speeds = np.array([p[3] for p in init_pts])

    if speeds.size == 0:
        speeds = np.array([0.0])
        x_coords = np.array([0.0])
        y_coords = np.array([0.0])

    boundaries, norm, mids, labels = compute_boundaries(speeds)
    sc = ax.scatter(x_coords, y_coords, c=speeds, cmap=cmap, norm=norm, s=10)
    cbar = fig.colorbar(sc, ax=ax, boundaries=boundaries)
    cbar.set_ticks(mids)
    cbar.set_ticklabels(labels)
    cbar.set_label("Speed (km/h)")
    ax.set_title(f"Lap {init_lap}")

    # Bottom lap label (e.g., "Lap 1/3")
    lap_label = fig.text(0.5, 0.085, f"Lap {current_idx + 1}/{len(laps)}", ha="center", va="center")

    # Precompute cumulative time (ticks) per lap using section_ticks with reset at each section
    lap_time_data = {}
    for lap in laps:
        lpts = lap_to_points.get(lap, [])
        # lpts already in global time order because to_points sorted by tick_idx
        cum_ticks = []
        xs = []
        ys = []
        offset = 0
        last_section = None
        last_section_max = 0
        for p in lpts:
            sec = p[4]
            st = p[5] if isinstance(p[5], (int, float)) else None
            if st is None:
                continue
            st = int(st)
            # Detect section change
            if last_section is None:
                last_section = sec
                last_section_max = st
            elif sec != last_section:
                # Close previous section by adding its max ticks to offset
                offset += last_section_max
                last_section = sec
                last_section_max = st
            else:
                if st > last_section_max:
                    last_section_max = st

            cum_ticks.append(offset + st)
            xs.append(p[1])
            ys.append(p[2])

        lap_time_data[lap] = {
            "cum": np.array(cum_ticks, dtype=float) if cum_ticks else np.array([0.0]),
            "xs": np.array(xs, dtype=float) if xs else np.array([0.0]),
            "ys": np.array(ys, dtype=float) if ys else np.array([0.0]),
        }

    # Time slider (0 ... end of lap ticks). Shows current tick value.
    ax_tslider = plt.axes([0.12, 0.045, 0.62, 0.03])
    init_time = 0
    init_lap_time = lap_time_data.get(init_lap, {"cum": np.array([0.0])})["cum"]
    tslider = Slider(ax_tslider, "Time (ticks)", 0, float(init_lap_time.max()), valinit=init_time, valstep=1)

    # Car position marker (big bright red star)
    star_scatter = ax.scatter([x_coords[0] if x_coords.size else 0.0], [y_coords[0] if y_coords.size else 0.0],
                              marker='*', s=200, c='red', zorder=5)

    def update_star_for_time(lap_value, t_value):
        data = lap_time_data.get(lap_value)
        if data is None:
            return
        cum = data["cum"]
        xs = data["xs"]
        ys = data["ys"]
        if cum.size == 0:
            x, y = 0.0, 0.0
        else:
            # Find rightmost index where cum <= t_value
            idx = bisect.bisect_right(cum, t_value) - 1
            if idx < 0:
                idx = 0
            x, y = xs[idx], ys[idx]
        star_scatter.set_offsets(np.column_stack([[x], [y]]))

    def on_time_change(val):
        update_star_for_time(laps[current_idx], val)
        fig.canvas.draw_idle()

    tslider.on_changed(on_time_change)

    # Section markers for initial lap
    def draw_sections(lap_pts):
        nonlocal sec_markers, sec_annots
        # clear existing
        for m in sec_markers:
            try:
                m.remove()
            except Exception:
                pass
        for a in sec_annots:
            try:
                a.remove()
            except Exception:
                pass
        sec_markers = []
        sec_annots = []

        first_by_section = {}
        for p in lap_pts:
            sec, sec_ticks = p[4], p[5]
            if sec is None or sec_ticks is None:
                continue
            if sec not in first_by_section or (sec_ticks < first_by_section[sec][2]):
                first_by_section[sec] = (p[1], p[2], sec_ticks)

        for sec, (sx, sy, _) in sorted(first_by_section.items(), key=lambda kv: kv[0]):
            marker = ax.scatter([sx], [sy], c="black", s=20, marker="x")
            ann = ax.annotate(
                f"S{sec}",
                (sx, sy),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
                color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.6),
            )
            sec_markers.append(marker)
            sec_annots.append(ann)

    draw_sections(init_pts)

    # Update function when changing lap
    def update_to_index(idx: int):
        nonlocal current_idx, cbar, norm
        idx = max(0, min(idx, len(laps) - 1))
        if idx == current_idx:
            return
        current_idx = idx
        lap = laps[current_idx]
        lap_pts = lap_to_points.get(lap, [])

        xs = np.array([p[1] for p in lap_pts])
        ys = np.array([p[2] for p in lap_pts])
        sp = np.array([p[3] for p in lap_pts])
        if sp.size == 0:
            xs = np.array([0.0])
            ys = np.array([0.0])
            sp = np.array([0.0])

        # Update scatter data
        sc.set_offsets(np.column_stack([xs, ys]))
        sc.set_array(sp)

        # Update colormap/norm & colorbar to reflect current lap's speed range
        boundaries_new, norm_new, mids_new, labels_new = compute_boundaries(sp)
        sc.set_norm(norm_new)
        sc.set_cmap(cmap)
        # Replace colorbar
        if cbar is not None:
            try:
                cbar.remove()
            except Exception:
                pass
        cbar = fig.colorbar(sc, ax=ax, boundaries=boundaries_new)
        cbar.set_ticks(mids_new)
        cbar.set_ticklabels(labels_new)
        cbar.set_label("Speed (km/h)")

        # Update sections and title
        draw_sections(lap_pts)
        ax.set_title(f"Lap {lap}")
        lap_label.set_text(f"Lap {current_idx + 1}/{len(laps)}")
        fig.canvas.draw_idle()

    # Prev/Next buttons
    ax_prev = plt.axes([0.76, 0.12, 0.08, 0.05])
    ax_next = plt.axes([0.86, 0.12, 0.08, 0.05])
    btn_prev = Button(ax_prev, "Prev")
    btn_next = Button(ax_next, "Next")

    def on_prev(event):
        update_to_index(current_idx - 1)

    def on_next(event):
        update_to_index(current_idx + 1)

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)

    # Keyboard shortcuts: left/right arrows
    def on_key(event):
        if event.key == "left":
            on_prev(event)
        elif event.key == "right":
            on_next(event)

    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()


if __name__ == "__main__":
    main()