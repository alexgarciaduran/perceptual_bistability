# %% cartoon cloth

# full_script_two_populations_final.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter, zoom

# ----------------- Parameters -----------------
n_neurons_side = 6
num_neurons = n_neurons_side**2
neuron_radius = 0.28
plane_z = 0.0
stimulus_z = -5.0
square_size = 0.5
square_offset = -2.5
connection_fraction = 0.21
np.random.seed(42)

# plotting colors
GRAY = np.array([0.85, 0.85, 0.85])
NEURON_FILL = np.array([0.95, 0.95, 0.95])
BORDER = np.array([0.0, 0.0, 0.0])
# GROUP_A_COLOR = np.array([0.15, 0.75, 0.15])   # green
GROUP_A_COLOR = np.array([0.1, 0.1, 0.1])   # green
# GROUP_B_COLOR = np.array([0.55, 0.15, 0.75])   # purple
GROUP_B_COLOR = np.array([0.5, 0.5, 0.5])   # purple
WITHIN_GROUP_COLOR = np.array([0.9, 0.1, 0.1]) # red (excitatory)
BETWEEN_GROUP_COLOR = np.array([0.1, 0.3, 0.9])# blue (inhibitory)
DEFAULT_EDGE_COLOR = np.array([0.8, 0.8, 0.8]) # light gray for other connections

# ----------------- Generate neuron positions (grid) -----------------
x_neurons = np.linspace(-3, 3, n_neurons_side)
y_neurons = np.linspace(-3, 3, n_neurons_side)
xx, yy = np.meshgrid(x_neurons, y_neurons)
xx = xx.flatten()
yy = yy.flatten()
neuron_positions = np.vstack([xx, yy]).T  # shape (64,2)

# ----------------- All neighbor pairs (grid adjacency) -----------------
pairs = []
for i in range(n_neurons_side):
    for j in range(n_neurons_side):
        idx = i*n_neurons_side + j
        # right neighbor
        if j < n_neurons_side-1:
            idx2 = i*n_neurons_side + (j+1)
            pairs.append((idx, idx2))
        # down neighbor
        if i < n_neurons_side-1:
            idx2 = (i+1)*n_neurons_side + j
            pairs.append((idx, idx2))

# ----------------- Choose two radial subgroups (~k neurons each) -----------------
k = 10  # approx size per subgroup
center = np.array([0.0, 0.0])
dists = np.linalg.norm(neuron_positions - center, axis=1)

# pick two subgroup centers on a circle of radius ~1.5 at opposite angles
angleA = np.deg2rad(90)
angleB = angleA + np.pi
radius_group_center = 1.5
centerA = np.array([radius_group_center * np.cos(angleA),
                    radius_group_center * np.sin(angleA)])
centerB = np.array([radius_group_center * np.cos(angleB),
                    radius_group_center * np.sin(angleB)])

# select k nearest neurons to each center (disjoint)
orderA = np.argsort(np.linalg.norm(neuron_positions - centerA, axis=1))
groupA_idx = []
for idx in orderA:
    if len(groupA_idx) >= k: break
    groupA_idx.append(int(idx))

orderB = np.argsort(np.linalg.norm(neuron_positions - centerB, axis=1))
groupB_idx = []
for idx in orderB:
    if int(idx) in groupA_idx: continue
    if len(groupB_idx) >= k: break
    groupB_idx.append(int(idx))

# # if overlap or too small, pad from central nearest neighbors (rare)
# if len(groupA_idx) < k or len(groupB_idx) < k:
#     near_idx = np.argsort(dists)
#     combined = [i for i in near_idx if i not in groupA_idx and i not in groupB_idx]
#     for i in combined:
#         if len(groupA_idx) < k:
#             groupA_idx.append(int(i))
#         elif len(groupB_idx) < k:
#             groupB_idx.append(int(i))
#         if len(groupA_idx) >= k and len(groupB_idx) >= k:
#             break

# if overlap or too small, pad from remaining neurons — RANDOM assignment
if len(groupA_idx) < k or len(groupB_idx) < k:
    near_idx = np.argsort(dists)
    combined = [int(i) for i in near_idx
                if i not in groupA_idx and i not in groupB_idx]

    # shuffle so assignment is random
    rng = np.random.default_rng(12345)
    for i in combined:
        # randomly decide A or B
        if rng.random() < 0.5:
            # try A first
            if len(groupA_idx) < k:
                groupA_idx.append(i)
            elif len(groupB_idx) < k:
                groupB_idx.append(i)
        else:
            # try B first
            if len(groupB_idx) < k:
                groupB_idx.append(i)
            elif len(groupA_idx) < k:
                groupA_idx.append(i)

        if len(groupA_idx) >= k and len(groupB_idx) >= k:
            break

# group_of lookup
group_of = np.full(num_neurons, -1, dtype=int)
for idx in groupA_idx:
    group_of[idx] = 0
for idx in groupB_idx:
    group_of[idx] = 1


#------------to get random positions in the center
# # distances to each group center
# distA = np.linalg.norm(neuron_positions - centerA, axis=1)
# distB = np.linalg.norm(neuron_positions - centerB, axis=1)

# # sort neurons by how "close" they are to either subgroup center
# order = np.argsort(np.minimum(distA, distB))

# groupA_idx = []
# groupB_idx = []

# for idx in order:
#     # randomly choose group for this neuron
#     if np.random.rand() < 0.5:
#         choice = 0
#     else:
#         choice = 1

#     # assign respecting capacity k
#     if choice == 0 and len(groupA_idx) < k:
#         groupA_idx.append(idx)
#     elif choice == 1 and len(groupB_idx) < k:
#         groupB_idx.append(idx)
#     else:
#         # enforce filling the other group
#         if len(groupA_idx) < k:
#             groupA_idx.append(idx)
#         elif len(groupB_idx) < k:
#             groupB_idx.append(idx)

#     # stop when both filled
#     if len(groupA_idx) == k and len(groupB_idx) == k:
#         break

# # final group assignment
# group_of = np.full(num_neurons, -1)
# group_of[groupA_idx] = 0
# group_of[groupB_idx] = 1


# ----------------- Edge color map rules -----------------
edge_color_map = {}
for (i1, i2) in pairs:
    g1 = group_of[i1]
    g2 = group_of[i2]
    if g1 == g2 and g1 != -1:
        color = WITHIN_GROUP_COLOR
    elif (g1 == 0 and g2 == 1) or (g1 == 1 and g2 == 0):
        color = BETWEEN_GROUP_COLOR
    else:
        color = DEFAULT_EDGE_COLOR
    edge_color_map[(i1, i2)] = color

# ----------------- Select subset of neighbor pairs to emphasize (for cloth) -----------------
n_select = max(1, int(len(pairs) * connection_fraction))
selected_idxs = np.random.choice(len(pairs), n_select, replace=False)
selected_pairs = [pairs[i] for i in selected_idxs]

# Also build a mapping of which selected pair gets what color (use edge rules)
selected_colors = [edge_color_map[p] for p in selected_pairs]

# ----------------- Create display positions (jitter group neurons into blobs) ----------
# We'll keep original grid positions for non-group neurons; for group members add small random jitter
display_positions = neuron_positions.copy()
rng = np.random.default_rng(12345)
def jitter_point(base, max_radius=0.18):
    # random polar jitter inside radius
    r = max_radius * np.sqrt(rng.random())
    theta = rng.random() * 2*np.pi
    return base + np.array([r*np.cos(theta), r*np.sin(theta)])

# For each group, move display positions slightly toward the group's center (to form a blob)
for idx in groupA_idx:
    base = neuron_positions[idx]
    # move slightly toward centerA and add jitter
    direction = (centerA - base)
    display_positions[idx] = base + 0.05 * direction + jitter_point(np.zeros(2), max_radius=0.05)

for idx in groupB_idx:
    base = neuron_positions[idx]
    direction = (centerB - base)
    display_positions[idx] = base + 0.05 * direction + jitter_point(np.zeros(2), max_radius=0.05)

# ----------------- Panel A: neuron cloth (left) -----------------
def plot_neuron_cloth(ax):
    n = 400
    x = np.linspace(-3.5, 3.5, n)
    y = np.linspace(-3.5, 3.5, n)
    X, Y = np.meshgrid(x, y)

    # Cloth surface (smooth random)
    np.random.seed(21)
    Z_random = np.random.randn(*X.shape)
    Z_smooth = gaussian_filter(Z_random, sigma=90)
    Z_smooth *= 0.01
    thickness = 0.000015
    Z_top = Z_smooth
    Z_bottom = Z_smooth - thickness

    # High-res texture for smooth circles/lines
    high_res = 800
    x_hr = np.linspace(-3.5, 3.5, high_res)
    y_hr = np.linspace(-3.5, 3.5, high_res)
    X_hr, Y_hr = np.meshgrid(x_hr, y_hr)
    sheet_hr = np.ones((high_res, high_res, 3))  # white base

    # Helpers
    x_to_idx = lambda v: np.argmin(np.abs(x_hr - v))
    y_to_idx = lambda v: np.argmin(np.abs(y_hr - v))
    thin = max(1, int(high_res * 0.003))
    thick = max(1, int(high_res * 0.010))

    # Draw default thin gray edges for all neighbor pairs
    for (i1, i2), color in edge_color_map.items():
        # draw thin line for every edge
        x1, y1 = display_positions[i1]
        x2, y2 = display_positions[i2]
        dx, dy = x2 - x1, y2 - y1
        dist = np.hypot(dx, dy)
        if dist < 1e-8:
            continue
        ux, uy = dx/dist, dy/dist
        Ax, Ay = x1 + neuron_radius*ux, y1 + neuron_radius*uy
        Bx, By = x2 - neuron_radius*ux, y2 - neuron_radius*uy
        Ax_i, Ay_i = x_to_idx(Ax), y_to_idx(Ay)
        Bx_i, By_i = x_to_idx(Bx), y_to_idx(By)
        steps = max(6, int(dist * high_res / 6))
        t = np.linspace(0, 1, steps)
        xs = Ax_i + (Bx_i - Ax_i) * t
        ys = Ay_i + (By_i - Ay_i) * t
        for xi, yi in zip(xs.astype(int), ys.astype(int)):
            y0 = max(0, yi - thin); y1_ = min(high_res, yi + thin)
            x0 = max(0, xi - thin); x1_ = min(high_res, xi + thin)
            sheet_hr[y0:y1_, x0:x1_] = DEFAULT_EDGE_COLOR

    # Overpaint selected special edges with thicker colored strokes (red/blue)
    for (i1, i2), color in edge_color_map.items():
        if np.allclose(color, DEFAULT_EDGE_COLOR):
            continue
        x1, y1 = display_positions[i1]
        x2, y2 = display_positions[i2]
        dx, dy = x2 - x1, y2 - y1
        dist = np.hypot(dx, dy)
        if dist < 1e-8:
            continue
        ux, uy = dx/dist, dy/dist
        Ax, Ay = x1 + neuron_radius*ux, y1 + neuron_radius*uy
        Bx, By = x2 - neuron_radius*ux, y2 - neuron_radius*uy
        Ax_i, Ay_i = x_to_idx(Ax), y_to_idx(Ay)
        Bx_i, By_i = x_to_idx(Bx), y_to_idx(By)
        steps = max(8, int(dist * high_res / 4))
        t = np.linspace(0, 1, steps)
        xs = Ax_i + (Bx_i - Ax_i) * t
        ys = Ay_i + (By_i - Ay_i) * t
        for xi, yi in zip(xs.astype(int), ys.astype(int)):
            y0 = max(0, yi - thick); y1_ = min(high_res, yi + thick)
            x0 = max(0, xi - thick); x1_ = min(high_res, xi + thick)
            sheet_hr[y0:y1_, x0:x1_] = color

    # Draw neurons themselves (fill + border) and color group members differently; use display_positions
    border_thickness_world = 0.04  # world units
    for idx, (nx0, ny0) in enumerate(display_positions):
        # distance in high-res world coords:
        distance = np.sqrt((X_hr - nx0)**2 + (Y_hr - ny0)**2)
        fill_mask = distance < neuron_radius
        if group_of[idx] == 0:
            sheet_hr[fill_mask] = GROUP_A_COLOR
        elif group_of[idx] == 1:
            sheet_hr[fill_mask] = GROUP_B_COLOR
        else:
            sheet_hr[fill_mask] = NEURON_FILL
        # thin black border
        border_mask = np.logical_and(distance >= neuron_radius - border_thickness_world,
                                     distance <= neuron_radius)
        sheet_hr[border_mask] = BORDER

    # Interpolate down for plotting
    factor = n / high_res
    sheet = zoom(sheet_hr, (factor, factor, 1), order=1)

    ax.plot_surface(X, Y, Z_top, facecolors=sheet, edgecolor=None, antialiased=True,
                    rstride=1, cstride=1, shade=False)
    ax.plot_surface(X, Y, Z_bottom, color='white', edgecolor=None, antialiased=True)

    # thick black frame edges for cloth
    ax.plot(X[0, :], Y[0, :], Z_top[0, :], color='black', linewidth=5)
    ax.plot(X[-1, :], Y[-1, :], Z_top[-1, :], color='black', linewidth=5)
    ax.plot(X[:, 0], Y[:, 0], Z_top[:, 0], color='black', linewidth=5)
    ax.plot(X[:, -1], Y[:, -1], Z_top[:, -1], color='black', linewidth=5)

    # side faces for thickness
    edges_list = [(0, slice(None)), (-1, slice(None)), (slice(None), 0), (slice(None), -1)]
    for edge in edges_list:
        if isinstance(edge[0], int):
            xs = X[edge[0], :]; ys = Y[edge[0], :]; zt = Z_top[edge[0], :]; zb = Z_bottom[edge[0], :]
        else:
            xs = X[:, edge[1]]; ys = Y[:, edge[1]]; zt = Z_top[:, edge[1]]; zb = Z_bottom[:, edge[1]]
        verts = [list(zip(xs, ys, zb)) + list(zip(xs[::-1], ys[::-1], zt[::-1]))]
        ax.add_collection3d(Poly3DCollection(verts, facecolor='black', edgecolor=None))

    ax.view_init(elev=35, azim=25)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_box_aspect([1, 1, 0.3])
    ax.dist = 12.5
    ax.axis('off')

# ----------------- Panel B: factor graph (population shading) -----------------
def plot_factor_graph(ax):
    # draw neurons as small circles at display positions (so panels correspond visually)
    def draw_circle(ax, x0, y0, z0, r, facecolor, edgecolor='black', lw=1.2):
        theta = np.linspace(0, 2*np.pi, 60)
        xs = x0 + r * np.cos(theta)
        ys = y0 + r * np.sin(theta)
        zs = np.full_like(xs, z0)
        verts = [list(zip(xs, ys, zs))]
        ax.add_collection3d(Poly3DCollection(verts, facecolor=facecolor, edgecolor=edgecolor, linewidth=lw))

    # Draw shaded population regions as soft disks (semi-transparent)
    def draw_shaded_disk(ax, center, radius, color, z=0.0, npoints=120, alpha=0.35):
        theta = np.linspace(0, 2*np.pi, npoints)
        xs = center[0] + radius * np.cos(theta)
        ys = center[1] + radius * np.sin(theta)
        zs = np.full_like(xs, z)
        verts = [list(zip(xs, ys, zs))]
        ax.add_collection3d(Poly3DCollection(verts, facecolor=np.append(color, alpha), edgecolor=None))
        # ax.plot(xs, ys, zs, color=color, linewidth=3, alpha=0.8)
    
    # # Draw neuron-to-neuron connections first
    for (i1, i2) in pairs:  # or `pairs` if you want all connections
        x1, y1 = display_positions[i1]
        x2, y2 = display_positions[i2]
        g1, g2 = group_of[i1], group_of[i2]
        if g1 == g2 and g1 != -1:
            color = WITHIN_GROUP_COLOR  # WITHIN_GROUP_COLOR  # red
            lw = 3.0
        elif (g1 != -1 and g2 != -1) and (g1 != g2):
            color = BETWEEN_GROUP_COLOR  # BETWEEN_GROUP_COLOR  # blue
            lw = 3.0
        else:
            color = DEFAULT_EDGE_COLOR
            lw = 1.5
        ax.plot([x1, x2], [y1, y2], [0.0, 0.0], color=color, linewidth=lw, alpha=1)

    # draw neurons on plane with same colors (use original display positions)
    for idx, (nx, ny) in enumerate(display_positions):
        if group_of[idx] == 0:
            color = GROUP_A_COLOR
        elif group_of[idx] == 1:
            color = GROUP_B_COLOR
        else:
            color = NEURON_FILL
        draw_circle(ax, nx, ny, 0.0, neuron_radius, facecolor=color, edgecolor='black', lw=0.8)

    # shade group regions: compute centroid and radius approx
    centroidA = display_positions[groupA_idx].mean(axis=0)
    centroidB = display_positions[groupB_idx].mean(axis=0)
    # radius for shaded disk: cover group spread (plus margin)
    rA = np.max(np.linalg.norm(display_positions[groupA_idx] - centroidA, axis=1)) + 0.35
    rB = np.max(np.linalg.norm(display_positions[groupB_idx] - centroidB, axis=1)) + 0.35

    draw_shaded_disk(ax, centroidA, rA, GROUP_A_COLOR, z=0.0, alpha=0.4)
    draw_shaded_disk(ax, centroidB, rB, GROUP_B_COLOR, z=0.0, alpha=0.4)

    # Stimulus plane (below)
    xs = np.linspace(-4, 4, 2)
    ys = np.linspace(-4, 4, 2)
    XS, YS = np.meshgrid(xs, ys)
    ZS = np.full_like(XS, stimulus_z)
    ax.plot_surface(XS, YS, ZS, color='lightgray', alpha=0.12)
    ax.plot(XS[0,:], YS[0,:], ZS[0,:], color='black', linewidth=3)
    ax.plot(XS[-1,:], YS[-1,:], ZS[-1,:], color='black', linewidth=3)
    ax.plot(XS[:,0], YS[:,0], ZS[:,0], color='black', linewidth=3)
    ax.plot(XS[:,-1], YS[:,-1], ZS[:,-1], color='black', linewidth=3)
    
    ZS = np.full_like(XS, 0)
    ax.plot_surface(XS, YS, ZS, color='lightgray', alpha=0.02)
    ax.plot(XS[0,:], YS[0,:], ZS[0,:], color='black', linewidth=3)
    ax.plot(XS[-1,:], YS[-1,:], ZS[-1,:], color='black', linewidth=3)
    ax.plot(XS[:,0], YS[:,0], ZS[:,0], color='black', linewidth=3)
    ax.plot(XS[:,-1], YS[:,-1], ZS[:,-1], color='black', linewidth=3)
    
    # Single central factor connecting both populations: place between centroids
    factor_pos = (centroidA + centroidB) / 2.0
    fx, fy = factor_pos
    fz = 0.0 + square_offset  # place factor below neuron plane
    s = square_size / 2.0
    # make factor a small vertical slab (thin box)
    verts = [(fx - 0.02, fy - s, fz - s), (fx - 0.02, fy + s, fz - s),
             (fx + 0.02, fy + s, fz + s), (fx + 0.02, fy - s, fz + s)]
    ax.add_collection3d(Poly3DCollection([verts], facecolor='white', edgecolor='black'))

    # lines from each group's neurons to factor (colored by group)
    x, y = centroidA
    ax.plot([x, fx], [y, fy], [0.0, fz], color='k', linewidth=4, alpha=0.9)
    x, y = centroidB
    ax.plot([x, fx], [y, fy], [0.0, fz], color='k', linewidth=4, alpha=0.9)

    # between-group inhibitory visual (centroid to centroid) as thick blue line on neuron plane
    # ax.plot([centroidA[0], centroidB[0]], [centroidA[1], centroidB[1]], [0.0, 0.0],
    #         color=BETWEEN_GROUP_COLOR, linewidth=4, linestyle='-', alpha=0.9)
    # plot RFs
    theta = np.linspace(0, 2*np.pi, 200)
    for centroid, r, color in zip([centroidA, centroidB],
                                  [rA, rB],
                                  [GROUP_A_COLOR, GROUP_B_COLOR]):
        xs = centroid[0] + r * np.cos(theta)
        ys = centroid[1] + r * np.sin(theta)
        zs = np.full_like(xs, stimulus_z)
        ax.plot(xs, ys, zs, linestyle='--', color=color,
                linewidth=3, alpha=0.8)
    # vertical line from factor down to stimulus plane
    ax.plot([fx, fx], [fy, fy], [fz, stimulus_z], color='k', linewidth=4, alpha=0.9)

    ax.set_box_aspect([1, 1, 0.5])
    ax.view_init(elev=20, azim=25)
    ax.dist = 12.5
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.axis('off')

# ----------------- Compose figure -----------------
fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
plot_neuron_cloth(ax1)

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
plot_factor_graph(ax2)

plt.tight_layout()
plt.show()

SV_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/hysteresis/parameters/' # Alex
print('Saving PNG')
fig.savefig(SV_FOLDER + 'cartoon_neural_sheet.png', dpi=200, bbox_inches='tight')
print('Saving SVG')
fig.savefig(SV_FOLDER + 'cartoon_neural_sheet.svg', dpi=200, bbox_inches='tight')

#%% cartoon rates
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Number of dimensions (N)
N = 5  # Example: 5-dimensional neural activity
labels = [fr'$r_{i+1}$' for i in range(N)]

# Generate a simple 3D trajectory as a cartoon (just smooth curves)
t = np.linspace(0, np.pi, 200)
x = np.sin(t)**2/np.sinh(t)**2            # r1
y = np.cos(t)*np.tanh(t)            # r2
z = np.sin(2*t)/np.cosh(t)          # r3

# For visualization, we can pick only 3 axes to plot
axes_to_plot = [0, 1, 2]  # indices of r1, r2, r3
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(x, y, z, color='k', lw=3, label='neural trajectory')

ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect([1,1,1])
# Label axes
ax.set_xlabel(labels[axes_to_plot[0]])
ax.set_ylabel(labels[axes_to_plot[1]])
ax.set_zlabel(labels[axes_to_plot[2]])

# Draw axes as arrows
arrow_length = 1.5
ax.quiver(0,0,0, arrow_length*0.5,0,0, color='black', arrow_length_ratio=0.1)
ax.quiver(0,0,0, 0,-arrow_length*0.5,0, color='black', arrow_length_ratio=0.1)
ax.quiver(0,0,0, 0,0,arrow_length*0.5, color='black', arrow_length_ratio=0.1)
# Label axes
ax.text(arrow_length/2+0.1,0,0, labels[0], fontsize=12)
ax.text(0,-arrow_length/2-0.1,0, labels[1], fontsize=12)
ax.text(0,0,arrow_length/2+0.1, labels[2], fontsize=12)

ax.axis('off')
plt.show()