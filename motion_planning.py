import re
import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np
import numpy.linalg as LA

from sampling import Sampler
from planning_utils import *

from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local

import matplotlib.pyplot as plt
import multiprocessing as mp

class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()

def gridisplay_addpath(points=None, lines=None):
    print("\tUpdate grid")
    if lines:
        print("\t\tadd edges")
        for (n1, n2) in lines:
            plt.plot([n1[1], n2[1]], [n1[0], n2[0]], 'black')

    if points:
        print("\t\tadd points")
        for p in points:
            plt.scatter(p[1], p[0], marker='*', c='red')

    plt.draw()
    plt.pause(0.1)

def gridisplay_init(grid, start, goal):
    plt.imshow(grid, cmap='Greys', origin='lower')
    plt.scatter(start[1], start[0], marker='p', c='blue')
    plt.scatter(goal[1], goal[0], marker='X', c='blue')

    plt.draw()
    plt.pause(1)

def gridisplay_movepoint(oldplot, point):
    if oldplot:
        oldplot.remove()

    newplot = plt.scatter(point[1], point[0], marker='>', c='orange')
    plt.draw()
    plt.pause(0.1)

    return newplot

def gridisplay_worker(q):
    plt.rcParams['figure.figsize'] = 15, 10
    plt.rcParams['lines.markersize'] = 15
    fig = plt.figure()
    plt.xlabel('NORTH')
    plt.ylabel('EAST')
    plt.show(block=False)

    # Wait for expected grid object and both start and goal points then
    # initialize the display.
    grid  = q.get()
    start = q.get()
    goal  = q.get()
    gridisplay_init(grid, start, goal)
    fig.canvas.flush_events()
    fig.canvas.draw()

    # Wait for expected path object and add it to the grid.
    path  = q.get()
    edges = [[ path[i], path[i+1] ] for i in range(len(path)-1)]
    gridisplay_addpath(points=path[1:-1], lines=edges)

    # Vehicule spot update loop
    vehicule = None
    oldpoint = start
    while True:
        # Wait for new position.
        obj = q.get()
        point = (obj[0], obj[1])

        # Update vehicule spot on each 20 unit.
        if LA.norm(np.array(point) - np.array(oldpoint)) > 20:
            vehicule = gridisplay_movepoint(vehicule, point)
            oldpoint = point

        # Put final vehicule spot.
        if LA.norm(np.array(point) - np.array(goal)) < 0.1:
            gridisplay_movepoint(vehicule, point)
            break

    plt.show()

gridisp_queue = mp.Queue()
p = mp.Process(target=gridisplay_worker, args=(gridisp_queue,))
p.start()
print("started")

class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}
        self.grid = None
        self.grid_offsets = np.array([0, 0, 0])
        self.data = None
        self.edges = None

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()

        elif self.flight_state == States.WAYPOINT:
            gridisp_queue.put(self.local_to_grid(self.local_position))
            deadband = 10.0
            if len(self.waypoints) == 0:
                deadband = 0.1

            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < deadband:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def local_to_grid(self, position):
        #return 2D point in the grid.
        return (position[0] - self.grid_offsets[0],
                position[1] - self.grid_offsets[1])

    def grid_to_local(self, position):
        #return 3D point in the local coordinate.
        return (position[0] + self.grid_offsets[0],
                position[1] + self.grid_offsets[1],
                self.grid_offsets[2])

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def myplan0(self, start, goal):
        G = nx.Graph()
        for e in self.edges:
            p1 = e[0]
            p2 = e[1]
            dist = LA.norm(np.array(p2) - np.array(p1))
            G.add_edge(p1, p2, weight=dist)

        skel_start = closest_point(G, start)
        skel_goal  = closest_point(G, goal)

        path, cost = a_star_graph(G, heuristic, skel_start, skel_goal)
        path.insert(0, start)
        path.append(goal)
        #return path

        return prune_path_graph(path, self.grid)

    def myplan1(self, start, goal):
        sampler = Sampler(self.data)
        print("here1")
        polygons = sampler._polygons
        print("here2")
        nodes = sampler.sample(300)
        print("here3")

        graph = create_graph(nodes, 5, polygons)
        start = list(graph.nodes)[0]
        k = np.random.randint(len(graph.nodes))
        print(k, len(graph.nodes))
        goal = list(graph.nodes)[k]

        # Run A* to find a path from start to goal
        print('Local Start and Goal: ', start, goal)
        path, _ = a_star_graph(graph, heuristic, start, goal)
        path.insert(0, start)
        path.append(goal)

        return path

    def myplan2(self, start, goal):
        path, _ = a_star_grid(self.grid, heuristic, start, goal)
        return prune_path(path)

    def myplan3(self, start, goal):
        path =  [(316, 445), (316, 446), (317, 446), (317, 447), (318, 447), (318, 448), (319, 448), (319, 449), (320, 449), (320, 450), (321, 450), (321, 451), (322, 451), (322, 452), (323, 452), (323, 453), (324, 453), (324, 454), (325, 454), (325, 455), (326, 455)]
        return path

    def myplan4(self, start, goal):
        print("generating rrt")
        num_vertices = 300
        dt = 1
        rrt = generate_RRT(self.grid, start, num_vertices, dt)
        path, _ = a_star_graph(rrt, heuristic, start, goal)
        return prune_path(path)

    def myplan5(self, start, goal):
        sampler = Sampler(self.data)
        print("here1")
        polygons = sampler._polygons
        print("here2")
        nodes = sampler.sample(300)
        print("here3")

        graph = create_graph(nodes, 5, polygons)
        start = list(graph.nodes)[0]
        k = np.random.randint(len(graph.nodes))
        print(k, len(graph.nodes))
        goal = list(graph.nodes)[k]

        # Run A* to find a path from start to goal
        print('Local Start and Goal: ', start, goal)
        path, _ = a_star_graph(graph, heuristic, start, goal)
        path.append(goal)

    def plan_path(self):
        self.flight_state = States.PLANNING
        TARGET_ALTITUDE = 15
        SAFETY_DISTANCE = 5

        self.target_position[2] = TARGET_ALTITUDE

        # DONE: read lat0, lon0 from colliders into floating point values
        pattern = 'lat0 (-?\d+\.\d+), lon0 (-?\d+\.\d+)'
        with open('colliders.csv', 'r') as f:
            firstline = f.readline()
        match = re.search(pattern, firstline)
        lat0 = float(match[1])
        lon0 = float(match[2])

        # DONE: set home position to (lon0, lat0, 0)
        self.set_home_position(lon0, lat0, 0)

        # DONE: retrieve current global position
        # DONE: convert to current local position using global_to_local()
        current_position = global_to_local(self.global_position, self.global_home)

        print('\tGlobal home {0}\n\tGlobal position {1}\n\tlocal position {2}'
              .format(self.global_home, self.global_position, self.local_position))
        print('Loading obstacle map grid ... ')
        # Read in obstacle map
        self.data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)

        # Define a grid for a particular altitude and safety margin around obstacles
        self.grid, self.edges, north_offset, east_offset = \
                create_grid_and_edges(self.data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        self.grid_offsets = np.array([north_offset, east_offset, TARGET_ALTITUDE])
        print("\tNorth offset = {0}, east offset = {1}"
              .format(self.grid_offsets[0], self.grid_offsets[1]))

        start = (int(current_position[0]), int(current_position[1]), 0)
        grid_start = self.local_to_grid(start)

        maxn, maxe = self.grid.shape
        while True:
            n = np.random.randint(maxn)
            e = np.random.randint(maxe)
            if self.grid[n, e] == False:
                grid_goal = (n, e)
                break

        goal  = self.grid_to_local(grid_goal)

        print('\tLocal Start and Goal: ', start, goal)
        print('\tGrid Start and Goal: ', grid_start, grid_goal)

        gridisp_queue.put(self.grid)
        gridisp_queue.put(grid_start)
        gridisp_queue.put(grid_goal)

        print("Searching for a path ...")
        path = self.myplan0(grid_start, grid_goal)
        #path = self.myplan1(grid_start, grid_goal)
        #path = self.myplan2(grid_start, grid_goal)
        #path = self.myplan3(grid_start, grid_goal)
        #path = self.myplan4(grid_start, grid_goal)
        #path = self.myplan5(grid_start, grid_goal)
        print(len(path), path)

        gridisp_queue.put(path)

        # Convert path to waypoints
        waypoints = [tuple(map(int, self.grid_to_local(p))) + tuple([0]) for p in path]

        # Set self.waypoints
        self.waypoints = waypoints
        # TODO: send waypoints to sim (this is just for visualization of waypoints)
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)
    time.sleep(1)

    drone.start()
