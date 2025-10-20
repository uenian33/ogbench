#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pickle

def get_walls(maze_str):
    """Extract walls from maze string - same as in a_star_collector.py"""
    lines = maze_str.split('\\')
    ch_arrs = []

    for l in lines:
        ch_arrs.append(np.array(list(l)))

    ch_arrs = np.array(ch_arrs)

    ws = []

    for lid in range(ch_arrs.shape[0]):
        l = ch_arrs[lid]
        w = []

        for cid in range(ch_arrs.shape[1]):
            c = ch_arrs[lid][cid]

            if c == "#":
                w.append([cid, lid])
            else:
                if len(w) > 0:
                    ws.append([w[0], w[-1]])
                    w = []

            if len(w) > 0 and cid == (ch_arrs.shape[1] - 1):                    
                ws.append([w[0], w[-1]])
                w = []

    for cid in range(ch_arrs.shape[1]):
        w = []

        for lid in range(ch_arrs.shape[0]):
            c = ch_arrs[lid][cid]
            
            if c == "#":
                w.append([cid, lid])
            else:
                if len(w) > 0:                    
                    ws.append([w[0], w[-1]])
                    w = []

            if len(w) > 0 and lid == (ch_arrs.shape[0] - 1):                    
                ws.append([w[0], w[-1]])
                w = []
   
    return ws


def classify_trajectory_type(goal):
    """Classify trajectory based on goal position to assign correct color"""
    # Goal ranges from a_star_collector.py (shifted by -w/2, -h/2)
    # gx_ranges = [[20.83, 22.11], [20.17, 22.11], [20.43, 21.85]]
    # gy_ranges = [[11.02, 14.41], [1.14, 4.77],  [6.91, 8.13]]
    # After shifting: subtract (23.2/2=11.6, 15.2/2=7.6)
    
    gx_ranges_shifted = [
        [20.83 - 11.6, 22.11 - 11.6],  # [9.23, 10.51]
        [20.17 - 11.6, 22.11 - 11.6],  # [8.57, 10.51]
        [20.43 - 11.6, 21.85 - 11.6]   # [8.83, 10.25]
    ]
    gy_ranges_shifted = [
        [11.02 - 7.6, 14.41 - 7.6],    # [3.42, 6.81]
        [1.14 - 7.6, 4.77 - 7.6],      # [-6.46, -2.83]
        [6.91 - 7.6, 8.13 - 7.6]       # [-0.69, 0.53]
    ]
    
    gx, gy = goal[0], goal[1]
    
    # Check which region the goal falls into
    for i in range(3):
        if (gx_ranges_shifted[i][0] <= gx <= gx_ranges_shifted[i][1] and 
            gy_ranges_shifted[i][0] <= gy <= gy_ranges_shifted[i][1]):
            return i
    
    # If not in any specific range, use closest match
    distances = []
    for i in range(3):
        center_x = np.mean(gx_ranges_shifted[i])
        center_y = np.mean(gy_ranges_shifted[i])
        dist = np.sqrt((gx - center_x)**2 + (gy - center_y)**2)
        distances.append(dist)
    
    return np.argmin(distances)


def visualize_collected_data(file_path='A_star_buffer.pkl'):
    """Load and visualize the collected A* trajectory data"""
    
    # Trajectory colors - same as in generate_data()
    tj_colors = ['#D92332', '#56A1BF', '#592B27']
    
    # Load the data
    print(f"Loading data from {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Data keys: {data.keys()}")
    print(f"Observations shape: {data['o'].shape}")
    print(f"Actions shape: {data['u'].shape}")
    print(f"Goals shape: {data['g'].shape}")
    print(f"Achieved goals shape: {data['ag'].shape}")
    
    # The data is stored as observations (o) which are already in shifted coordinates
    # We should use 'ag' (achieved goals) which contains the actual trajectory positions
    
    # Maze configuration
    grid_size = 0.4
    robot_radius = 0.7
    
    maze_str = "########################\\" + \
               "#OOOOOOOO#OOOOOOOOOOOOO#\\" + \
               "#OOO##OOO##OOO###OO##OO#\\" + \
               "#OOOOOOOOOOOOOOOOOO#OOO#\\" + \
               "#OOO####OOOOOO#######OO#\\" + \
               "#OO0000#OOO#O##OOOOO#OO#\\" + \
               "####OOO#OO##OO#OOO######\\" + \
               "#OOOOOO#OOO#OOOOOOOOOGO#\\" + \
               "#OOOOOO###O#OOOOOOOOOGO#\\" + \
               "####OOO#OO##OO#OOO######\\" + \
               "#OOOOOO#OOO#O##OOOOO#OO#\\" + \
               "#OOO####OOO#OO#######OO#\\" + \
               "#OOOOOOOOOOOOOOOOOO#OOO#\\" + \
               "#OOO##OOO#OOOO######OOO#\\" + \
               "#OOOOOOOO#OOOOOOOOOOOOO#\\" + \
               "########################"
    
    # Get walls
    walls = get_walls(maze_str)
    
    # Calculate dimensions - need to match what A* planner calculates
    lines = maze_str.split("\\")
    
    # Count obstacles to get min/max ranges
    ox = []
    oy = []
    for lid, l in enumerate(lines):
        for cid, c in enumerate(list(l)):
            if c == "#":
                ox.append(cid)
                oy.append(lid)
    
    min_x, min_y = round(min(ox)), round(min(oy))
    max_x, max_y = round(max(ox)), round(max(oy))
    x_width = round((max_x - min_x) / grid_size)
    y_width = round((max_y - min_y) / grid_size)
    
    w = x_width * grid_size
    h = y_width * grid_size
    
    print(f"Calculated w={w}, h={h}")
    print(f"Shift: w/2={w/2}, h/2={h/2}")
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Draw walls - need to shift them
    for wal in walls:
        x1, y1 = np.array(wal[0][0]) - w/2, np.array(wal[0][1]) - h/2
        x2, y2 = np.array(wal[1][0]) - w/2, np.array(wal[1][1]) - h/2
        plt.plot([x1, x2], [y1, y2], linestyle='-', color='black', linewidth=8.5)
    
    # Plot trajectories - data is already shifted, plot directly
    num_trajectories = data['o'].shape[0]
    print(f"\nPlotting {num_trajectories} trajectories...")
    
    for i in range(num_trajectories):
        # Use achieved goals which contain the actual trajectory
        traj = data['ag'][i]  # Shape: (100, 2), already shifted
        traj_goal = data['g'][i][0]  # Goal position, already shifted
        
        # Classify trajectory type based on goal
        traj_type = classify_trajectory_type(traj_goal)
        color = tj_colors[traj_type]
        
        # Plot the trajectory directly - it's already in the right coordinate system
        rx = traj[:, 0]
        ry = traj[:, 1]
        
        plt.plot(rx, ry, color=color, alpha=0.75, linewidth=2.5)
    
    plt.grid(False)
    plt.axis("equal")
    plt.title(f"Visualized A* Trajectories (n={num_trajectories})")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=tj_colors[0], label='Type 1'),
        Patch(facecolor=tj_colors[1], label='Type 2'),
        Patch(facecolor=tj_colors[2], label='Type 3')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.show()
    
    print("\nVisualization complete!")


if __name__ == '__main__':
    visualize_collected_data('A_star_buffer.pkl')