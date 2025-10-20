#!/usr/bin/env python
# coding: utf-8

# In[5]:


"""

A* grid planning

author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""

import math

import matplotlib.pyplot as plt

import numpy as np

import random

import pickle

class AStarPlanner:

    def __init__(self, ox, oy, resolution, rr):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while 1:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node,
                                                                     open_set[
                                                                         o]))
            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        ##print("min_x:", self.min_x)
        #print("min_y:", self.min_y)
        #print("max_x:", self.max_x)
        #print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        #print("x_width:", self.x_width)
        #print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion


def get_walls(maze_str):
    # transfer maze into arrays
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

            if c=="#":
                w.append([cid, lid])
            else:
                if len(w)>0:
                    ws.append([w[0], w[-1]])
                    w = []

            if len(w)>0 and cid==(ch_arrs.shape[1]-1):                    
                ws.append([w[0], w[-1]])
                w = []

    for cid in range(ch_arrs.shape[1]):
        w = []

        for lid in range(ch_arrs.shape[0]):
            c = ch_arrs[lid][cid]
            
            if c=="#":
                w.append([cid, lid])
            else:
                if len(w)>0:                    
                    ws.append([w[0], w[-1]])
                    w = []

            if len(w)>0 and lid==(ch_arrs.shape[0]-1):                    
                ws.append([w[0], w[-1]])
                w = []
   
    return ws
    



show_animation = False

def generate_data():

    # set start points and end points
    sx_ranges = [[0.70, 2.29], [0.74, 2.68], [0.70, 2.68]]
    sy_ranges = [[0.74, 5.28], [10.68, 14.30], [7.11, 8.34]]
    gx_ranges = [[20.83, 22.11], [20.17, 22.11], [20.43, 21.85]]
    gy_ranges = [[11.02, 14.41], [1.14, 4.77],  [6.91, 8.13]]
    tj_colors = ['#D92332', '#56A1BF', '#592B27']

    t_num = 100#15
    grid_size = 0.4  # [m]
    robot_radius = 0.7  # [m]

    # make maze
    maze_str =  "########################\\"+\
                "#OOOOOOOO#OOOOOOOOOOOOO#\\"+\
                "#OOO##OOO##OOO###OO##OO#\\"+\
                "#OOOOOOOOOOOOOOOOOO#OOO#\\"+\
                "#OOO####OOOOOO#######OO#\\"+\
                "#OO0000#OOO#O##OOOOO#OO#\\"+\
                "####OOO#OO##OO#OOO######\\"+\
                "#OOOOOO#OOO#OOOOOOOOOGO#\\"+\
                "#OOOOOO###O#OOOOOOOOOGO#\\"+\
                "####OOO#OO##OO#OOO######\\"+\
                "#OOOOOO#OOO#O##OOOOO#OO#\\"+\
                "#OOO####OOO#OO#######OO#\\"+\
                "#OOOOOOOOOOOOOOOOOO#OOO#\\"+\
                "#OOO##OOO#OOOO######OOO#\\"+\
                "#OOOOOOOO#OOOOOOOOOOOOO#\\"+\
                "########################"
    lines = maze_str.split("\\")

    ox = []
    oy = []
    available_ps = []
    for lid, l in enumerate(lines):
        height = len(lines)
        for cid, c in enumerate(list(l)):
            width = len(l)
            if c=="#":
                ox.append(cid)
                oy.append(lid)
                #print(cid, l)
            else:
                available_ps.append([cid, lid])

    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    w, h = a_star.x_width*a_star.resolution, a_star.y_width*a_star.resolution

    # draw continuuous walls
    walls = get_walls(maze_str)


    for wal in walls:
        #plt.axline((w[0][0], w[0][1]), (w[1][0], w[1][1]))
        plt.plot([np.array(wal[0][0])-w/2, np.array(wal[1][0])-w/2], [np.array(wal[0][1])-h/2,np.array( wal[1][1])-h/2], linestyle = '-', color='black',linewidth=8.5)

    #plt.show()


    all_trajs = []

    # generate trajs
    for tp in range(len(sx_ranges)):
        '''
        start = random.choice(available_ps)
        g = random.choice(available_ps)
        sx = start[0]  # [m]
        sy = start[1]  # [m]
        gx = g[0]  # [m]
        gy = g[1]  # [m]
        '''
        sx_r = sx_ranges[tp]
        sy_r = sy_ranges[tp]
        gx_r = gx_ranges[tp]
        gy_r = gy_ranges[tp]

        for i in range(t_num):
            # start and goal position
            sx = np.random.uniform(sx_r[0], sx_r[1])  # [m]
            sy = np.random.uniform(sy_r[0], sy_r[1])  # [m]
            gx = np.random.uniform(gx_r[0], gx_r[1])  # [m]
            gy = np.random.uniform(gy_r[0], gy_r[1])  # [m]

            if show_animation:  # pragma: no cover
                #plt.plot(ox, oy, "s", color='black')
                plt.plot(sx, sy, "og")
                plt.plot(gx, gy, "xb")
                plt.grid(False)
                plt.axis("equal")

            a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
            rx, ry = a_star.planning(sx, sy, gx, gy)



            rx, ry = np.array(rx), np.array(ry) 
            rx, ry = rx-w/2, ry-h/2

            if np.random.uniform(1) > 0.75:
                rx[1:-1] = rx[1:-1] + np.clip((np.random.normal(0, 0.3, rx[1:-1].shape[0])), -1, 1)
                ry[1:-1] = ry[1:-1] + np.clip((np.random.normal(0, 0.3, ry[1:-1].shape[0])), -1, 1)

            if rx.shape[0]>10:
                all_trajs.append([rx, ry, tj_colors[tp]])

            if show_animation:  # pragma: no cover
                plt.plot(rx, ry, "-r")
                plt.pause(0.001)
                plt.show()

    print(len(all_trajs))

    # print shiftes walls
    for wid, wal in enumerate(walls):
        #plt.axline((w[0][0], w[0][1]), (w[1][0], w[1][1]))
        walls[wid][0][0], walls[wid][1][0], walls[wid][0][1], walls[wid][1][1] = np.array(wal[0][0])-w/2, np.array(wal[1][0])-w/2, np.array(wal[0][1])-h/2,np.array( wal[1][1])-h/2

    print("walls:\n", walls)
    for tj in all_trajs:
        rx = tj[0]
        ry = tj[1]
        color = tj[2]
        #plt.plot(ox, oy, "s")
        #plt.plot(sx, sy, "og")
        #plt.plot(gx, gy, "xb")
        plt.grid(False)
        plt.axis("equal")
        plt.plot(rx, ry, color, alpha=0.75, linewidth=2.5)
        print(rx[1:] - rx[:-1])
    
    # save to dataset
    ss, acts, ags, gs = [], [], [], []
    for tj in all_trajs[:]:
        # obs, a, next_obs, r, g, ag
        rx, ry = tj[0], tj[1]
        traj_len = len(rx)
        exact_len = 100

        if traj_len > 15:
            g = np.ones((exact_len, 2))
            g[:,0] = g[:,0]*tj[0][-1]
            g[:,1] = g[:,1]*tj[1][-1]

            # achieved goals
            ag = np.zeros((traj_len, 2))
            ag[:,0] = rx
            ag[:,1] = ry

            ag_extra = np.ones((exact_len-traj_len, 2))
            ag_extra[:,0] = ag_extra[:,0]*tj[0][-1]
            ag_extra[:,1] = ag_extra[:,1]*tj[1][-1]

            #print(ag.shape, ag_extra.shape)
            ag = np.append(ag,ag_extra, axis=0)

            # observations
            s = np.zeros((traj_len, 2))
            s[:,0] = rx
            s[:,1] = ry

            s_extra = np.ones((exact_len-traj_len, 2))
            s_extra[:,0] = s_extra[:,0]*tj[0][-1]
            s_extra[:,1] = s_extra[:,1]*tj[1][-1]

            s = np.append(s,s_extra, axis=0)

            # actions
            act = s[1:] - s[:-1]
            act_final = act[-1:]
            act = np.append(act,act_final, axis=0)

            print(traj_len)

            ss.append(s)
            acts.append(act)
            gs.append(g)
            ags.append(ag)

    # shuffle order
    new_idx = np.arange(len(ss))
    np.random.shuffle(new_idx)
    np.random.shuffle(new_idx)
    np.random.shuffle(new_idx)
    
    print(new_idx)
    ss = [ss[i] for i in new_idx]
    #ss[new_idx]
    acts = [acts[i] for i in new_idx]
    #acts[new_idx]
    gs = [gs[i] for i in new_idx]
    #gs[new_idx]
    ags = [ags[i] for i in new_idx]
    #ags[new_idx]

    #print(ag.shape, s.shape, acts.shape, g.shape)
    ss, acts, gs, ags = np.array(ss), np.array(acts), np.array(gs), np.array(ags)
    print(ss.shape, acts.shape, gs.shape, ags.shape)
    new_dict = {'o':{},'u':{},'g':{},'ag':{}}
    new_dict['o'] = ss
    new_dict['u'] = acts[:,:-1]
    new_dict['g'] = gs[:,:-1]
    new_dict['ag'] = ags


    #print(new_dict['g'], new_dict['ag'])

    out_path = 'A_star_buffer.pkl'
    
    pickle.dump( new_dict, open(out_path, "wb" ) )

    plt.show()      



def visualize_data(color, file_name):

    # set start points and end points
    sx_ranges = [[0.70, 2.29], [0.74, 2.68], [0.70, 2.68]]
    sy_ranges = [[0.74, 5.28], [10.68, 14.30], [7.11, 8.34]]
    gx_ranges = [[20.83, 22.11], [20.17, 22.11], [20.43, 21.85]]
    gy_ranges = [[11.02, 14.41], [1.14, 4.77],  [6.91, 8.13]]
    tj_colors = ['#029386', '#FAC205', '#FF6347']

    t_num = 80#15
    grid_size = 0.4  # [m]
    robot_radius = 0.7  # [m]

    # make maze
    maze_str =  "########################\\"+\
                "#OOOOOOOO#OOOOOOOOOOOOO#\\"+\
                "#OOO##OOO##OOO###OO##OO#\\"+\
                "#OOOOOOOOOOOOOOOOOO#OOO#\\"+\
                "#OOO####OOOOOO#######OO#\\"+\
                "#OO0000#OOO#O##OOOOO#OO#\\"+\
                "####OOO#OO##OO#OOO######\\"+\
                "#OOOOOO#OOO#OOOOOOOOOGO#\\"+\
                "#OOOOOO###O#OOOOOOOOOGO#\\"+\
                "####OOO#OO##OO#OOO######\\"+\
                "#OOOOOO#OOO#O##OOOOO#OO#\\"+\
                "#OOO####OOO#OO#######OO#\\"+\
                "#OOOOOOOOOOOOOOOOOO#OOO#\\"+\
                "#OOO##OOO#OOOO######OOO#\\"+\
                "#OOOOOOOO#OOOOOOOOOOOOO#\\"+\
                "########################"
    lines = maze_str.split("\\")

    ox = []
    oy = []
    available_ps = []
    for lid, l in enumerate(lines):
        height = len(lines)
        for cid, c in enumerate(list(l)):
            width = len(l)
            if c=="#":
                ox.append(cid)
                oy.append(lid)
                #print(cid, l)
            else:
                available_ps.append([cid, lid])

    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    w, h = a_star.x_width*a_star.resolution, a_star.y_width*a_star.resolution

    # draw continuuous walls
    walls = get_walls(maze_str)


    for wal in walls:
        #plt.axline((w[0][0], w[0][1]), (w[1][0], w[1][1]))
        plt.plot([np.array(wal[0][0])-w/2, np.array(wal[1][0])-w/2], [np.array(wal[0][1])-h/2,np.array( wal[1][1])-h/2], linestyle = '-', color='black',linewidth=8.5)

    #plt.show()

    with open(file_name, 'rb') as handle:
        all_trajs = pickle.load(handle)

   
    # print shiftes walls
    for wid, wal in enumerate(walls):
        #plt.axline((w[0][0], w[0][1]), (w[1][0], w[1][1]))
        walls[wid][0][0], walls[wid][1][0], walls[wid][0][1], walls[wid][1][1] = np.array(wal[0][0])-w/2, np.array(wal[1][0])-w/2, np.array(wal[0][1])-h/2,np.array( wal[1][1])-h/2

    #print("walls:\n", walls)
    print(all_trajs[0])
    all_trajs = np.array(all_trajs)
    for tj in all_trajs:
        rx = tj[:,0]
        ry = tj[:,1]
        color = color
        #plt.plot(ox, oy, "s")
        #plt.plot(sx, sy, "og")
        #plt.plot(gx, gy, "xb")
        plt.grid(False)
        plt.axis("equal")
        plt.plot(rx, ry, color, alpha=0.75, linewidth=2.5)
        #print(rx[1:] - rx[:-1])
    

    plt.show()      

if __name__ == '__main__':
    generate_data()
    #visualize_data('#1D594E','ContinuousMaze-v0-0.9-0.1-awac-adv-binary0.05-relabel0.0-200--gauss--False-ood-True-expectile-0.5.pkl')
    #visualize_data('#F29F05','ContinuousMaze-v0-0.9-0.1-td3bc_chain_hybrid_v7-binary0.05-relabel0.0-200--gauss--False-ood-True-expectile-0.5.pkl')
    #visualize_data('#F2668B','ContinuousMaze-v0-0.9-0.9-td3bc_chain_v8-binary0.05-relabel0.5-200--gauss--False-ood-True-expectile-0.5.pkl')
    #visualize_data('#03A688','ContinuousMaze-v0-0.5-0.5-td3bc-binary0.05-relabel0.5-200--gauss--False-ood-False-expectile-0.5.pkl')
   

# In[ ]:



