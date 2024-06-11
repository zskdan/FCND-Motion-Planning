import matplotlib.pyplot as plt
import multiprocessing as mp

import numpy as np
import numpy.linalg as LA

class Gridisplay:
    def __init__(self):
        self.q = None
        print("Gridisplay started ... ")
        self.q = mp.Queue()
        self.p = mp.Process(target=gridisplay_worker, args=(self.q,))
        self.p.start()

    @property
    def queue(self):
        return self.q

    @property
    def process(self):
        return self.p

    def put(self, obj):
        return self.q.put(obj)

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
    if not isinstance(grid, np.ndarray):
        raise TypeError("type of object gotten from the queue is not a grid (numpy.ndarray)")

    start = q.get()
    if not isinstance(start, tuple):
        raise TypeError("type of object gotten from the queue is not a point (tuple)")

    goal  = q.get()
    if not isinstance(goal, tuple):
        raise TypeError("type of object gotten from the queue is not a point (tuple)")

    gridisplay_init(grid, start, goal)
    fig.canvas.flush_events()
    fig.canvas.draw()

    # Wait for expected path object and add it to the grid.
    path  = q.get()
    if not isinstance(path, list):
        raise TypeError("type of object gotten from the queue is not a path (list)")

    edges = [[ path[i], path[i+1] ] for i in range(len(path)-1)]
    gridisplay_addpath(points=path[1:-1], lines=edges)

    # Vehicule spot update loop
    vehicule = None
    oldpoint = start
    while True:
        # Wait for new position.
        obj = q.get()
        if not isinstance(obj, tuple):
            raise TypeError("type of object gotten from the queue is not a point (tuple)")

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

