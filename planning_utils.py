from enum import Enum
from queue import PriorityQueue
import numpy as np
import networkx as nx
import numpy.linalg as LA
from sklearn.neighbors import KDTree
from shapely.geometry import Polygon, Point, LineString

from scipy.spatial import Voronoi
from bresenham import bresenham

def breakpoint():
    import pdb; pdb.set_trace()

class RRT:
    def __init__(self, x_init):
        # A tree is a special case of a graph with
        # directed edges and only one path to any vertex.
        self.tree = nx.DiGraph()
        self.tree.add_node(x_init)

    def add_vertex(self, x_new):
        self.tree.add_node(tuple(x_init))

    def add_edge(self, x_near, x_new, u):
        self.tree.add_edge(tuple(x_near), tuple(x_new), orientation=u)

    @property
    def vertices(self):
        return self.tree.nodes()

    @property
    def edges(self):
        return self.tree.edges()


def sample_state(grid):
    maxx, maxy = grid.shape
    while True:
        x = np.random.randint(maxx)
        y = np.random.randint(maxy)
        if grid[x,y] == 0:
            break

    return (x, y)

def nearest_neighbor(x_rand, rrt):
    mylist = np.array([*rrt.vertices])
    idx = np.linalg.norm(mylist - np.array(x_rand), axis=1).argmin()
    return mylist[idx]

def select_input(x_rand, x_near):
    return np.arctan2(x_rand[1]-x_near[1], x_rand[0]-x_near[0])

def new_state(x_near, u, dt):
    x = x_near[0] + np.cos(u)*dt
    y = x_near[1] + np.sin(u)*dt
    return (x, y)

def generate_RRT(grid, x_init, x_goal, num_vertices, dt, pathlookup=False):
    rrt = RRT(x_init)

    maxiter = 0
    found = False
    for _ in range(num_vertices):
        maxiter += 1

        # Add a biais toward the goal, in order to converge
        # rapidely
        if np.random.randint(0,100) < 10:
            x_rand = x_goal
        else:
            x_rand = sample_state(grid)
            # sample states until a free state is found
            while grid[int(x_rand[0]), int(x_rand[1])] == 1:
                x_rand = sample_state(grid)

        x_near = nearest_neighbor(x_rand, rrt)
        u = select_input(x_rand, x_near)
        x_new = new_state(x_near, u, dt)

        if x_new[0] < grid.shape[0] and x_new[1] < grid.shape[1]:
            if grid[int(x_new[0]), int(x_new[1])] == 0:
                # the orientation `u` will be added as metadata to
                # the edge
                rrt.add_edge(x_near, x_new, u)
                if pathlookup:
                    if LA.norm(np.array(x_new) - np.array(x_goal)) < dt:
                        found = True
                        print("\tPath found!")
                        break

    if pathlookup and not found:
        rrt = None

    return rrt, maxiter, x_init, x_new

def create_RRT(grid, x_init, x_goal, num_vertices, dt):
    return generate_RRT(grid, x_init, x_goal, num_vertices, dt, pathlookup=True)

def create_grid(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Initialize an empty list for Voronoi points
    points = []

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1

            # add center of obstacles to points list
            points.append([north - north_min, east - east_min])

    return grid, points, int(north_min), int(east_min)

def create_voronoi_graph(grid, points):
    """
    Returns Voronoi graph edges given obstacle data and the
    drone's altitude.
    """

    # location of obstacle centres
    graph = Voronoi(points)

    edges = []
    for v in graph.ridge_vertices:
        p1 = graph.vertices[v[0]]
        p2 = graph.vertices[v[1]]

        # If the edge does not hit on obstacle
        # add it to the list
        if not collision_check(grid, p1, p2):
            # array to tuple for future graph creation step)
            p1 = (p1[0], p1[1])
            p2 = (p2[0], p2[1])
            edges.append((p1, p2))

    VG = nx.Graph()
    for e in edges:
       p1 = e[0]
       p2 = e[1]
       dist = LA.norm(np.array(p2) - np.array(p1))
       VG.add_edge(p1, p2, weight=dist)

    return VG

def create_local_voxmap(data, center, cube):
    """
    Returns a grid representation of a 3D configuration space
    based on given obstacle data.
    """
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))
    east_min  = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max  = np.ceil(np.max(data[:, 1] + data[:, 4]))
    alt_min   = 0
    alt_max   = np.ceil(np.max(data[:, 2] + data[:, 5]))

    # minimum and maximum north coordinates
    north_local_min = np.clip(center[0]-(cube[0]//2), north_min, north_max)
    north_local_max = np.clip(center[0]+(cube[0]//2), north_min, north_max)

    # minimum and maximum east coordinates
    east_local_min = np.clip(center[1]-(cube[1]//2), east_min, east_max)
    east_local_max = np.clip(center[1]+(cube[1]//2), east_min, east_max)

    alt_local_min = np.clip(-center[2]-(cube[2]//2), alt_min, alt_max)
    alt_local_max = np.clip(-center[2]+(cube[2]//2), alt_min, alt_max)

    #print("max: ", north_max, east_max, alt_max)
    #print("min: ", north_min, east_min, alt_min)

    #print("local max: ", north_local_max, east_local_max, alt_local_max)
    #print("local min: ", north_local_min, east_local_min, alt_local_min)

    num_samples = 50
    nvals = np.random.uniform(north_local_min, north_local_max, num_samples).astype(int)
    evals = np.random.uniform(east_local_min, east_local_max, num_samples).astype(int)
    avals = np.random.uniform(alt_local_min, alt_local_max, num_samples).astype(int)
    samples = list(zip(nvals, evals, avals))
    tree = KDTree(samples)
    to_keep = samples.copy()

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = cube[0]
    east_size =  cube[1]
    alt_size =   cube[2]
    polygones = []
    voxmap = np.zeros((north_size, east_size, alt_size), dtype=np.bool)
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        #if (north+d_north < north_local_max and north-d_north > north_local_min) and \
        #   (east+d_east   < east_local_max  and east-d_east   > east_local_min):# and \
           #(alt-d_alt) > alt_local_min:
        #if (north-d_north > north_local_min and north-d_north > north_local_min)
        #    print("obstacle n:", north - d_north - north_min, north + d_north-north_min)
        #    print("obstacle e:", east - d_east - east_min, east + d_east - east_min)
        if True:
            obstacle = [
                int(np.clip(north - d_north - north_local_min, 0, north_size-1)),
                int(np.clip(north + d_north - north_local_min, 0, north_size-1)),
                int(np.clip(east - d_east - east_local_min, 0, east_size-1)),
                int(np.clip(east + d_east - east_local_min, 0, east_size-1))
            ]
            maxradius = max(d_north, d_east, d_alt)
            corners = [(obstacle[0], obstacle[2]), (obstacle[0], obstacle[3]), (obstacle[1], obstacle[3]), (obstacle[1], obstacle[2])]
            #height = int(alt+d_alt)
            height = int(np.clip(alt + d_alt, 0, alt_size-1))
            voxmap[obstacle[0]:obstacle[1], obstacle[2]:obstacle[3], 0:height] = True
            p = Polygon(corners)
            for idx in tree.query_radius([(north, east, alt)], maxradius, return_distance=False)[0]:
                pnt = samples[idx]
                if pnt in to_keep and p.contains(Point(pnt[0], pnt[1])) and pnt[2]<=height:
                    to_keep.remove(pnt)
                else:
                    polygones.append((p,height))
        # TODO: fill in the voxels that are part of an obstacle with `True`
        #
        # i.e. grid[0:5, 20:26, 2:7] = True
    graph = create_graph(to_keep, 5, polygones)

    return voxmap, graph


# Assume all actions cost the same.
class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)

    NWEST = (-1, -1, np.sqrt(2))
    NEAST = (-1, 1, np.sqrt(2))
    SWEST = (1, -1, np.sqrt(2))
    SEAST = (1, 1, np.sqrt(2))

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])


def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle

    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid_actions.remove(Action.NORTH)
    if x + 1 > n or grid[x + 1, y] == 1:
        valid_actions.remove(Action.SOUTH)
    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid_actions.remove(Action.WEST)
    if y + 1 > m or grid[x, y + 1] == 1:
        valid_actions.remove(Action.EAST)

    if (x - 1 < 0 and y - 1 < 0) or grid[x - 1, y - 1] == 1:
        valid_actions.remove(Action.NWEST)
    if (y + 1 > m and x - 1 < 0) or grid[x - 1, y + 1] == 1:
        valid_actions.remove(Action.NEAST)
    if (y - 1 < 0 and x + 1 > n) or grid[x + 1, y - 1] == 1:
        valid_actions.remove(Action.SWEST)
    if (y + 1 > m and x + 1 > n) or grid[x + 1, y + 1] == 1:
        valid_actions.remove(Action.SEAST)

    return valid_actions

def point(p):
    return np.array([p[0], p[1], 1.]).reshape(1, -1)

def collinearity_check(p1, p2, p3, epsilon=1e-6):
    m = np.concatenate((p1, p2, p3), 0)
    det = np.linalg.det(m)
    return abs(det) < epsilon

def collision_check(grid, p1, p2):
    cells = list(bresenham(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
    hit = False

    for c in cells:
        # First check if we're off the map
        if np.amin(c) < 0 or c[0] >= grid.shape[0] or c[1] >= grid.shape[1]:
            hit = True
            break
        # Next check if we're in collision
        if grid[c[0], c[1]] == 1:
            hit = True
            break

    return hit

def get_edge(pt, npt, cube):
    #TODO add support of 3d edges
    cells = list(bresenham(int(pt[0]), int(pt[1]), int(npt[0]), int(npt[1])))
    for c in cells:
        if c[0] == int(pt[0])-cube[0] or c[0] == int(pt[0])+cube[0] or \
           c[1] == int(pt[1])-cube[1] or c[1] == int(pt[1])+cube[1]: 
            return (c[0], c[1], npt[2])

# Prune the path in both directions
def prune_path(path, grid):
    # the one way to goal.
    ppath = prune_path_oneway(path, grid)
    # the way back to start.
    ppathback = prune_path_oneway(ppath[::-1], grid)

    return ppathback[::-1]

def prune_path_oneway(path, grid):
    #print(path)
    pruned_path = []

    if path is not None:
        pruned_path.append(path[0])
        i = 0
        j = 1
        while j < len(path):
            if collision_check(grid, path[i], path[j]) == True:
                pruned_path.append(path[j-1])
                i = j-1
            j += 1;

        pruned_path.append(path[-1])

    print("\tPrune path from {} to {} ".format(len(path), len(pruned_path)))

    return pruned_path

def prune_path2(path):
    pruned_path = []

    if path is not None:
        pruned_path.append(path[0])
        for i in range(len(path)-2):
            if collinearity_check(point(path[i]), point(path[i+1]), point(path[i+2])) == False:
                pruned_path.append(path[i+1])
        pruned_path.append(path[-1])

    print("prune path from {} to {} ".format(len(path), len(pruned_path)))

    return pruned_path

def can_connect(n1, n2, polygons):
    l = LineString([n1, n2])
    for p, h in polygons:
        if l.crosses(p) and h >= min(n1[2], n2[2]):
            return False

    return True

def closest_point(graph, current_point):
    """
    Compute the closest point in the `graph`
    to the `current_point`.
    """
    closest_point = None
    dist = 100000
    for p in graph.nodes:
        d = LA.norm(np.array(p) - np.array(current_point))
        if d < dist:
            closest_point = p
            dist = d
    return closest_point

def create_graph(nodes, k, polygons):
    g = nx.Graph()
    tree = KDTree(nodes)
    for n1 in nodes:
        # for each node connect try to connect to k nearest nodes
        idxs = tree.query([n1], k, return_distance=False)[0]

        for idx in idxs:
            n2 = nodes[idx]
            if n2 == n1:
                continue

            if can_connect(n1, n2, polygons):
                g.add_edge(n1, n2, weight=1)
    return g

def a_star_graph(graph, heuristic, start, goal):
    """Modified A* to work with NetworkX graphs."""

    path = []
    path_cost = 0

    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_cost = item[0]
        current_node = item[1]

        if current_node != goal:
            for next_node in graph[current_node]:
                cost = graph.edges[current_node, next_node]['weight']
                new_cost = current_cost + cost + heuristic(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    queue.put((new_cost, next_node))

                    branch[next_node] = (new_cost, current_node)
        else:
            found = True
            break

    if found and branch:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]

        path.append(branch[n][1])
        print('\tFound a path.')

    return path[::-1], path_cost


def a_star_grid(grid, h, start, goal):

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:
            current_cost = branch[current_node][0]

        if current_node == goal:
            print('\tFound a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))

    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
    return path[::-1], path_cost


def heuristic(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

