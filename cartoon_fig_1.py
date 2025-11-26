import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom


if __name__ == '__main__':
    # ----------------- Shared parameters -----------------
    n_neurons_side = 4
    num_neurons = n_neurons_side**2
    neuron_radius = 0.2
    plane_z = 0.0
    stimulus_z = -5.0
    square_size = 0.5
    square_offset = -3.5
    connection_fraction = 0.21
    np.random.seed(42)
    
    # ----------------- Generate neuron positions -----------------
    x_neurons = np.linspace(-1.5, 1.5, n_neurons_side)
    y_neurons = np.linspace(-1.5, 1.5, n_neurons_side)
    xx, yy = np.meshgrid(x_neurons, y_neurons)
    xx = xx.flatten()
    yy = yy.flatten()
    neuron_positions = list(zip(xx, yy))
    
    # ----------------- Generate neighbor pairs -----------------
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
    
    # ----------------- Randomly select subset -----------------
    n_select = max(1,int(len(pairs)*connection_fraction))
    selected_idxs = np.random.choice(len(pairs), n_select, replace=False)
    selected_pairs = [pairs[i] for i in selected_idxs]
    
    # ----------------- Assign colors -----------------
    colors_list = [np.array([0.1,0.8,0.1]), np.array([0.8,0.1,0.1]), np.array([0.2,0.2,0.2])]
    connection_colors = [colors_list[np.random.choice(3, p=[0.5, 0.5, 0.])] for _ in selected_pairs]
    
    # ----------------- Panel A: 3D neuron cloth -----------------
    def plot_neuron_cloth(ax, neuron_positions, selected_pairs, connection_colors):
        n = 400
        x = np.linspace(-2,2,n)
        y = np.linspace(-2,2,n)
        X, Y = np.meshgrid(x,y)
    
        # Cloth surface
        np.random.seed(21)
        Z_random = np.random.randn(*X.shape)
        Z_smooth = gaussian_filter(Z_random, sigma=90)
        Z_smooth *= 0.01
        thickness = 0.000015
        Z_top = Z_smooth
        Z_bottom = Z_smooth - thickness
    
        # High-res texture
        high_res = 800
        x_hr = np.linspace(-2,2,high_res)
        y_hr = np.linspace(-2,2,high_res)
        X_hr, Y_hr = np.meshgrid(x_hr,y_hr)
        sheet_hr = np.ones((high_res, high_res, 3))
    
        # Convenience
        x_to_idx = lambda v: np.argmin(np.abs(x_hr - v))
        y_to_idx = lambda v: np.argmin(np.abs(y_hr - v))
        line_thickness = int(high_res*0.007)
    
        # Draw colored connections in texture
        Nside = n_neurons_side
        for pair in pairs:
            color = [0.8]*3
            idx1, idx2 = pair
            x1, y1 = neuron_positions[idx1]
            x2, y2 = neuron_positions[idx2]
            dx, dy = x2-x1, y2-y1
            dist = np.sqrt(dx*dx + dy*dy)
            ux, uy = dx/dist, dy/dist
            Ax, Ay = x1 + neuron_radius*ux, y1 + neuron_radius*uy
            Bx, By = x2 - neuron_radius*ux, y2 - neuron_radius*uy
            Ax_i, Ay_i = x_to_idx(Ax), y_to_idx(Ay)
            Bx_i, By_i = x_to_idx(Bx), y_to_idx(By)
            steps = int(dist*high_res/2)
            t = np.linspace(0,1,steps)
            xs = Ax_i + (Bx_i-Ax_i)*t
            ys = Ay_i + (By_i-Ay_i)*t
            for xi, yi in zip(xs.astype(int), ys.astype(int)):
                y0 = max(0, yi-line_thickness); y1_ = min(high_res, yi+line_thickness)
                x0 = max(0, xi-line_thickness); x1_ = min(high_res, xi+line_thickness)
                sheet_hr[y0:y1_, x0:x1_] = color
        
        # thicker for colored
        line_thickness = int(high_res*0.01)
        
        for pair, color in zip(selected_pairs, connection_colors):
            idx1, idx2 = pair
            x1, y1 = neuron_positions[idx1]
            x2, y2 = neuron_positions[idx2]
            dx, dy = x2-x1, y2-y1
            dist = np.sqrt(dx*dx + dy*dy)
            ux, uy = dx/dist, dy/dist
            Ax, Ay = x1 + neuron_radius*ux, y1 + neuron_radius*uy
            Bx, By = x2 - neuron_radius*ux, y2 - neuron_radius*uy
            Ax_i, Ay_i = x_to_idx(Ax), y_to_idx(Ay)
            Bx_i, By_i = x_to_idx(Bx), y_to_idx(By)
            steps = int(dist*high_res/2)
            t = np.linspace(0,1,steps)
            xs = Ax_i + (Bx_i-Ax_i)*t
            ys = Ay_i + (By_i-Ay_i)*t
            for xi, yi in zip(xs.astype(int), ys.astype(int)):
                y0 = max(0, yi-line_thickness); y1_ = min(high_res, yi+line_thickness)
                x0 = max(0, xi-line_thickness); x1_ = min(high_res, xi+line_thickness)
                sheet_hr[y0:y1_, x0:x1_] = color
    
        # Draw neurons
        for nx, ny in neuron_positions:
            mask = (X_hr-nx)**2 + (Y_hr-ny)**2 < neuron_radius**2
            sheet_hr[mask] = [0.8,0.8,0.8]
            distance = np.sqrt((X_hr - nx)**2 + (Y_hr - ny)**2)
            border_thickness = 0.05
            border_mask = np.logical_and(distance >= neuron_radius - border_thickness,
                                         distance <= neuron_radius)
            sheet_hr[border_mask] = [0,0,0]
    
        # Interpolate down
        factor = n/high_res
        sheet = zoom(sheet_hr, (factor,factor,1), order=1)
    
        ax.plot_surface(X, Y, Z_top, facecolors=sheet, edgecolor=None, antialiased=True,
                        rstride=1, cstride=1, shade=False)
        ax.plot_surface(X, Y, Z_bottom, color='white', edgecolor=None, antialiased=True)
        # Top edge
        ax.plot(X[0, :], Y[0, :], Z_top[0, :], color='black', linewidth=5)
        
        # Bottom edge
        ax.plot(X[-1, :], Y[-1, :], Z_top[-1, :], color='black', linewidth=5)
        
        # Left edge
        ax.plot(X[:, 0], Y[:, 0], Z_top[:, 0], color='black', linewidth=5)
        
        # Right edge
        ax.plot(X[:, -1], Y[:, -1], Z_top[:, -1], color='black', linewidth=5)
        # Thickness edges
        edges = [(0,slice(None)),(-1,slice(None)),(slice(None),0),(slice(None),-1)]
        for edge in edges:
            if isinstance(edge[0], int):
                xs = X[edge[0],:]; ys = Y[edge[0],:]; zt = Z_top[edge[0],:]; zb = Z_bottom[edge[0],:]
            else:
                xs = X[:,edge[1]]; ys = Y[:,edge[1]]; zt = Z_top[:,edge[1]]; zb = Z_bottom[:,edge[1]]
            verts = [list(zip(xs,ys,zb)) + list(zip(xs[::-1],ys[::-1],zt[::-1]))]
            ax.add_collection3d(Poly3DCollection(verts, facecolor='black', edgecolor=None))
    
        ax.view_init(elev=35, azim=25)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_box_aspect([1,1,0.3])
        ax.axis('off')
    
    # ----------------- Panel B: Factor graph -----------------
    def plot_factor_graph(ax, neuron_positions, selected_pairs, connection_colors):
        plane_z = 0.0
        stimulus_z = -5.0
        square_size = 0.1
        square_offset = -3.5
    
        # Draw neurons
        def draw_circle(ax, x0, y0, z0, r, n_points=50, facecolor='white', edgecolor='black'):
            theta = np.linspace(0, 2*np.pi, n_points)
            xs = x0 + r*np.cos(theta)
            ys = y0 + r*np.sin(theta)
            zs = np.full_like(xs, z0)
            verts = [list(zip(xs,ys,zs))]
            ax.add_collection3d(Poly3DCollection(verts, facecolor=[0.8,0.8,0.8], edgecolor=edgecolor, linewidth=4))
        for x, y in neuron_positions:
            draw_circle(ax, x, y, plane_z, neuron_radius)
    
        # Stimulus plane
        xs = np.linspace(-2,2,2)
        ys = np.linspace(-2,2,2)
        XS, YS = np.meshgrid(xs,ys)
        ZS = np.full_like(XS, stimulus_z)
        ax.plot_surface(XS,YS,ZS,color='lightgray',alpha=0.1)
        ax.plot(XS[0,:],YS[0,:],ZS[0,:],color='black',linewidth=3)
        ax.plot(XS[-1,:],YS[-1,:],ZS[-1,:],color='black',linewidth=3)
        ax.plot(XS[:,0],YS[:,0],ZS[:,0],color='black',linewidth=3)
        ax.plot(XS[:,-1],YS[:,-1],ZS[:,-1],color='black',linewidth=3)
        
        
        ZS = np.full_like(XS, 0)
        ax.plot(XS[0,:],YS[0,:],ZS[0,:],color='black',linewidth=3)
        ax.plot(XS[-1,:],YS[-1,:],ZS[-1,:],color='black',linewidth=3)
        ax.plot(XS[:,0],YS[:,0],ZS[:,0],color='black',linewidth=3)
        ax.plot(XS[:,-1],YS[:,-1],ZS[:,-1],color='black',linewidth=3)
    
        # Squares and connections
        for pair, color in zip(selected_pairs, connection_colors):
            idx1, idx2 = pair
            x1, y1 = neuron_positions[idx1]
            x2, y2 = neuron_positions[idx2]
            xm = (x1+x2)/2 + np.random.uniform(-0.1,0.1)
            ym = (y1+y2)/2 + np.random.uniform(-0.1,0.1)
            zm = plane_z + square_offset
            s = square_size/2
            verts = [(xm, ym-s, zm-2*s), (xm, ym+s, zm-2*s), (xm, ym+s, zm+2*s), (xm, ym-s, zm+2*s)]
            square = Poly3DCollection([verts], facecolor='white', edgecolor='black')
            ax.add_collection3d(square)
            # diagonal lines from neurons to square
            ax.plot([x1,xm],[y1,ym],[plane_z,zm], color=color, linewidth=3)
            ax.plot([x2,xm],[y2,ym],[plane_z,zm], color=color, linewidth=3)
            # vertical line from square to stimulus plane
            ax.plot([xm,xm],[ym,ym],[zm,stimulus_z], color=color, linewidth=3)
    
        ax.set_box_aspect([1,1,0.5])
        ax.view_init(elev=35, azim=25)
        ax.dist = 12.5
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.axis('off')
    
    # ----------------- Generate figure -----------------
    fig = plt.figure(figsize=(16,8))
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    plot_neuron_cloth(ax1, neuron_positions, selected_pairs, connection_colors)
    
    ax2 = fig.add_subplot(1,2,2, projection='3d')
    plot_factor_graph(ax2, neuron_positions, selected_pairs, connection_colors)
    
    plt.tight_layout()
    plt.show()
    
    
    SV_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/phd/folder_save/hysteresis/parameters/'  # Alex
    print('Saving PNG')
    fig.savefig(SV_FOLDER + 'cartoon_neural_sheet_all.png', dpi=200, bbox_inches='tight')
    print('Saving SVG')
    fig.savefig(SV_FOLDER + 'cartoon_neural_sheet_all.svg', dpi=200, bbox_inches='tight')
