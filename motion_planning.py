import re
import argparse
import time
import msgpack
from enum import Enum, auto
from threading import Thread

import numpy as np
import numpy.linalg as LA

from sampling import Sampler
from planning_utils import *

from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local

from gridisplay import Gridisplay

gridisp = Gridisplay()

class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()

class MotionPlanning(Drone):

    def __init__(self, connection, global_goal=None):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        self.grid = None
        self.obsdata = None
        self.grid_offsets = np.array([0, 0, 0])
        self.obspoints = None
        self.safety_distance = 0
        self.planned = False
        self.local3d = False
        self.localpath = {}
        self.localinit = False

        try:
            match = re.search('(-?\d+\.\d+), (-?\d+\.\d+), (-?\d+\.\d+)', global_goal)
            self.global_goal = (float(match[1]), float(match[2]), float(match[3]))
        except:
            self.global_goal = None

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
            gridisp.put(self.local_to_grid(self.local_position))
            deadband = 2 * self.safety_distance
            if len(self.waypoints) == 0 or self.local3d == True:
                deadband = 0.1

            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < deadband:
                if len(self.waypoints) > 0 or len(self.localpath) > 0:
                    self.local_plan()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def local_plan(self):
        if self.local3d == True:
            if len(self.localpath) > 0:
                self.localpoint_transition()
                return
            else:
                print("local replanning")
                cube = (40,40,10)
                # skip first call
                if self.localinit == True:
                    voxmap, self.graph  = create_local_voxmap(self.obsdata, self.local_position, cube)
                    next_localpoint = get_edge(self.local_position, self.nextwaypoint, cube)
                    if next_localpoint:
                        local_start = closest_point(self.graph,
                                   (int(self.local_position[0]),
                                    int(self.local_position[1]),
                                   -int(self.local_position[2])))
                        local_goal = closest_point(self.graph, next_localpoint)
                        self.localpath, _ = a_star_graph(self.graph, heuristic, local_start, local_goal)
                        prune_path2(self.localpath)
                        return
                self.localinit = True

        # transit to new waypoint if local3d algorithm is not inuse or if
        # no more next_localpoint.
        self.waypoint_transition()

    def isHovering(self):
        return (abs(self.local_velocity[0]) < 0.01 and \
                abs(self.local_velocity[1]) < 0.01 and \
                abs(self.local_velocity[2]) < 0.01)

    def velocity_callback(self):
        # We trigger disarming if the drone is hovering. to make it possible
        # to land on top of a building.
        if self.flight_state == States.LANDING:
            if self.isHovering():
                self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                if self.planned:
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

    def localpoint_transition(self):
        localpoint = self.localpath.pop(0)
        self.goto_target(localpoint)

    def goto_target(self, position):
        self.target_position = position
        print('\ttarget position', self.target_position)
        heading = np.arctan2(self.target_position[1] - self.local_position[1],
                             self.target_position[0] - self.local_position[0])
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], heading)

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.nextwaypoint = self.waypoints.pop(0)
        if self.local3d == True:
            self.local_plan()
        else:
             self.goto_target(self.nextwaypoint)

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

    # Helper to return 2D point in the grid.
    def local_to_grid(self, position):
        return (position[0] - self.grid_offsets[0],
                position[1] - self.grid_offsets[1])

    # Helper to return 3D point in the local coordinate.
    def grid_to_local(self, position):
        return (position[0] + self.grid_offsets[0],
                position[1] + self.grid_offsets[1],
                self.grid_offsets[2])

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    # Voronoi Graph + A_star planning algorithm.
    def myplan_graph(self, start, goal):
        print("\tUsing Voronoi Graph A_star algorithm")
        # Create graph from the grid and obstacle points.
        VG = create_voronoi_graph(self.grid, self.obspoints)

        # Looking for closest points of start and goal in graph.
        skel_start = closest_point(VG, start)
        skel_goal  = closest_point(VG, goal)

        if np.linalg.norm(np.array(skel_start) - np.array(skel_goal)) < 0.1:
            # Return a direct path (not null), if goal and start points are too closed.
            path = [start, goal]
        else :
            # Run the a_star algorithm to find a path.
            path, cost = a_star_graph(VG, heuristic, skel_start, skel_goal)
            if path:
                # Insert start and goal points to the path, as the path is using the closest_point.
                path.insert(0, start)
                path.append(goal)
                # Prune the path to only few 2 by 2 coolinear waypoints.
                path = prune_path(path, self.grid)

        return path

    # WARNING: this may take very long time to process: (>1300 seconds)
    def myplan_pr(self, start, goal):
        print("\tUsing Probabilistic Roadmap algorithm")
        sampler = Sampler(self.obsdata)
        polygons = sampler._polygons
        nodes = sampler.sample(300)

        # Create graph.
        graph = create_graph(nodes, 5, polygons)
        start3d = start + (self.grid_offsets[2],)
        goal3d  = goal  + (self.grid_offsets[2],)

        # Looking for closest points of start and goal in graph.
        skel_start = closest_point(graph, start3d)
        skel_goal  = closest_point(graph, goal3d)

        # Run A* to find a path from start to goal
        if np.linalg.norm(np.array(skel_start) - np.array(skel_goal)) < 0.1:
            # Return a direct path (not null), if goal and start points are too closed.
            path = [start3d, goal3d]
        else :
            path, _ = a_star_graph(graph, heuristic, skel_start, skel_goal)
            if path:
                # Insert start and goal points to the path, as the path is using the closest_point.
                #path.insert(0, start3d)
                #path.append(goal3d)
                # Prune the path to only few 2 by 2 coolinear waypoints.
                path = prune_path(path, self.grid)

        return path

    def myplan_rh(self, start, goal):
        print("\tUsing Receding Horizon algorithm")
        self.local3d = True
        path, _ = a_star_grid(self.grid, heuristic, start, goal)
        if path:
           path = prune_path(path, self.grid)
        return path

    # Planning algorithm which apply directly A_star to the grid.
    # WARNING: this may take long time to process
    def myplan_grid(self, start, goal):
        print("\tUsing Grid A_star algorithm")
        path, _ = a_star_grid(self.grid, heuristic, start, goal)
        return prune_path(path, self.grid)


    # RRT planning algorithm.
    def myplan_rrt(self, start, goal):
        print("\tUsing RRT algorithm")
        path = []
        dt = 20
        maxiteration = 5000
        rrt, iteration, snode, gnode = \
            create_RRT(self.grid, start, goal, maxiteration, dt)
        #print("generate path after {} iteration".format(iteration))
        if rrt:
            path = nx.shortest_path(rrt.tree, source=snode, target=gnode)
            path = prune_path(path, self.grid)
            if snode != goal:
                path.append(goal)

        return path

    def myplan(self, start, goal):
        t0 = time.time()

        if np.linalg.norm(np.array(start) - np.array(goal)) < self.safety_distance:
            # Return a direct path (not null), if goal and start points are too
            # close.
            path = [start, goal]
        else:
            # Choose one of the following planning algorithm:
            # 1. A* star algorithm applied to the grid.
            #path = self.myplan_grid(start, goal)

            # 2. A* star algorithm applied to a graph.
            #path = self.myplan_graph(start, goal)

            # 3. Probabilistic Roadmap algorithm.
            #path = self.myplan_pr(start, goal)

            # 4. Receding Horizon algorithm.
            #path = self.myplan_rh(start, goal)

            # 5. RRT algorithm.
            path = self.myplan_rrt(start, goal)

        time_taken = time.time() - t0
        print("\t",len(path), path)
        print("\tPlanning process took {} seconds ...".format(time_taken))

        return path

    def plan_path(self):
        self.flight_state = States.PLANNING
        TARGET_ALTITUDE = 15
        SAFETY_DISTANCE = 5

        self.safety_distance    = SAFETY_DISTANCE
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
        self.obsdata = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)

        # Define a grid for a particular altitude and safety margin around obstacles
        self.grid, self.obspoints, north_offset, east_offset = \
                create_grid(self.obsdata, TARGET_ALTITUDE, SAFETY_DISTANCE)
        self.grid_offsets = np.array([north_offset, east_offset, TARGET_ALTITUDE])
        print("\tNorth offset = {0}, east offset = {1}"
              .format(self.grid_offsets[0], self.grid_offsets[1]))

        start = (int(current_position[0]), int(current_position[1]), 0)
        grid_start = self.local_to_grid(start)

        # generate a goal randomly if not provided as argument
        if not self.global_goal:
            print("\tCannot parse global goal, generate it randomly!")
            maxn, maxe = self.grid.shape
            while True:
                n = np.random.randint(maxn)
                e = np.random.randint(maxe)
                if self.grid[n, e] == False:
                    grid_goal = (n, e)
                    break

            goal  = self.grid_to_local(grid_goal)
        else:
            print("\tglobal_goal:",self.global_goal)
            goal = global_to_local(self.global_goal, self.global_home)
            grid_goal  = self.local_to_grid(goal)

        #grid_goal = (grid_start[0]+10, grid_start[1]+10)

        print('\tLocal Start and Goal: ', start, goal)
        print('\tGrid Start and Goal: ', grid_start, grid_goal)

        gridisp.put(self.grid)
        gridisp.put(grid_start)
        gridisp.put(grid_goal)
        time.sleep(1)

        # Start the planning process in a separate thread, to not block
        # the current function call, otherwise the server will shutdown the
        # tcp connection on a 30 seconds timeout.
        t = Thread(target=self.plan_worker, args=[grid_start, grid_goal])
        t.start()

    def plan_worker(self, grid_start, grid_goal):
        print("Searching for a path ...")
        path = self.myplan(grid_start, grid_goal)

        while not path:
            self.target_position[2] += 10
            self.grid_offsets[2] = self.target_position[2]
            print("Retry generating the grid at altitude:", self.target_position[2])
            self.grid, self.obspoints, north_offset, east_offset = \
                create_grid(self.obsdata, self.target_position[2] , self.safety_distance)
            path = self.myplan(grid_start, grid_goal)

        gridisp.put(path)

        # Convert path to waypoints.
        waypoints = [tuple(map(int, self.grid_to_local(p))) + tuple([0]) for p in path]

        # Set self.waypoints.
        self.waypoints = waypoints
        # Send waypoints to sim (this is just for visualization of waypoints).
        self.send_waypoints()

        # Finaly mark the planned tag in order to proceed to next transition.
        self.planned = True

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
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--goal', type=str, default=None, help="Global goal location, i.e. '-122.397970, 37.795090, 26.190'")
    args = parser.parse_args()

    # Change the client timeout to an arbitrary high value in order to avoid
    # hanging the connection in case of planning algorithm that may take long
    # time.
    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=600)
    drone = MotionPlanning(conn, global_goal=args.goal)
    time.sleep(1)

    drone.start()
