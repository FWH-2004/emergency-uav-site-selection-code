import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
points = {
    "S1/D1": (0, 0),
    "S2/D2": (6, 0),
    "S3/D3": (0, 8),
    "S4/D4": (6, 8),
    "S5/D5": (3, 4),
}
selected_sites = ["S1/D1", "S3/D3", "S5/D5"]
w_star = 11

def ellipse_from_foci(f1, f2, sum_dist, n_points=800):
    x1, y1 = f1
    x2, y2 = f2
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    focal_dist = np.hypot(x2 - x1, y2 - y1)
    c = focal_dist / 2
    a = sum_dist / 2
    if a <= c:
        raise ValueError("sum_dist 必须大于两个焦点之间的距离，才能构成椭圆。")
    b = np.sqrt(a**2 - c**2)
    theta = np.arctan2(y2 - y1, x2 - x1)
    t = np.linspace(0, 2 * np.pi, n_points)
    x = cx + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
    y = cy + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
    return x, y
f1 = points["S1/D1"]
f2 = points["S5/D5"]
x_e1, y_e1 = ellipse_from_foci(f1, f2, w_star)
f3 = points["S3/D3"]
f5 = points["S5/D5"]
x_e2, y_e2 = ellipse_from_foci(f3, f5, w_star)
fig, ax = plt.subplots(figsize=(9, 7))
for name, (x, y) in points.items():
    ax.scatter(x, y, color="blue", s=80, zorder=3)
    if name == "S1/D1":
        ax.text(x + 0.2, y + 0.25, f"{name} ({x},{y})", color="blue", fontsize=12)
    elif name == "S2/D2":
        ax.text(x + 0.2, y + 0.25, f"{name} ({x},{y})", color="blue", fontsize=12)
    elif name == "S3/D3":
        ax.text(x + 0.2, y - 0.1, f"{name} ({x},{y})", color="blue", fontsize=12)
    elif name == "S4/D4":
        ax.text(x + 0.2, y + 0.1, f"{name} ({x},{y})", color="blue", fontsize=12)
    else:
        ax.text(x + 0.2, y - 0.1, f"{name} ({x},{y})", color="blue", fontsize=12)
for site in selected_sites:
    x, y = points[site]
    circle = plt.Circle((x, y), 0.22, fill=False, color="red", linewidth=2, zorder=4)
    ax.add_patch(circle)
ax.plot(x_e1, y_e1, color="orange", linewidth=2.5,
        label=r"Ellipse: Foci $S1,S5$,  $d_1+d_2=11$")
ax.plot(x_e2, y_e2, color="green", linewidth=2.5,
        label=r"Ellipse: Foci $S3,S5$,  $d_1+d_2=11$")
ax.plot([f1[0], f2[0]], [f1[1], f2[1]], '--', color='orange', alpha=0.5)
ax.plot([f3[0], f5[0]], [f3[1], f5[1]], '--', color='green', alpha=0.5)
ax.axhline(0, color='black', linewidth=1.2)
ax.axvline(0, color='black', linewidth=1.2)
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlabel("x / km", fontsize=14)
ax.set_ylabel("y / km", fontsize=14)
ax.set_aspect('equal', adjustable='box')
all_x = np.concatenate([
    np.array([p[0] for p in points.values()]),
    x_e1, x_e2
])
all_y = np.concatenate([
    np.array([p[1] for p in points.values()]),
    y_e1, y_e2
])

margin = 0.8
ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
ax.legend(loc="upper right", fontsize=10)

plt.tight_layout()
plt.show()
