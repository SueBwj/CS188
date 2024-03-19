from GridWorld import GridWorld
from Grid import Grid
import numpy as np
grid = Grid()
grid_world = GridWorld(grid=grid,goalVals=np.arange(10))
grid_world.simulate(np.arange(10))

