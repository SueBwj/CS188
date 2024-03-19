# Joey Velez-Ginorio
# Gridworld Implementation
# ---------------------------------

import numpy as np

class Grid():
	"""

		Defines the necessary environment elements for several variations
		of GridWorld to be easily constructed by GridWorld().

	"""

	def __init__(self, grid='bookGrid'):
		self.row = 0
		self.col = 0

		self.rewards = list()
		self.walls = list()

		if grid == 'bookGrid':
			self.getBookGrid()

		elif grid == 'testGrid':
			self.getTestGrid()


	def setGrid(self, fileName):
		""" 
			Initializes grid to the desired gridWorld configuration.
		"""
		gridBuffer = np.loadtxt(fileName, dtype=str)

		self.row = len(gridBuffer)
		self.col = len(gridBuffer[0])

		gridMatrix = np.empty([self.row,self.col], dtype=str)

		for i in range(self.row):
			gridMatrix[i] = list(gridBuffer[i])

		self.rewards = list(zip(*np.where(gridMatrix == 'R')))
		self.walls = list(zip(*np.where(gridMatrix == 'W')))


	def getBookGrid(self):
		""" 
			Builds the canonical gridWorld example from the Sutton,
			Barto book.
		"""
		fileName = 'gridWorlds/bookGrid.txt'
		self.setGrid(fileName)
		
	def getTestGrid(self):
		"""
			Builds a test grid, use this to quickly try out different
			gridworld environments. Simply modify the existing testGrid.txt
			file.
		"""
		fileName = 'gridWorlds/testGrid.txt'
		self.setGrid(fileName)