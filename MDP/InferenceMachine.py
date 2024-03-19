
# Joey Velez-Ginorio
# Gridworld Implementation
# ---------------------------------

from GridWorld import GridWorld
from Grid import Grid
from scipy.stats import uniform
from scipy.stats import beta
from scipy.stats import expon
import numpy as np
import pyprind
import matplotlib.pyplot as plt


class InferenceMachine():
	"""
		Conducts inference via MDPs for GridWorld.
	"""
	def __init__(self, grid, space=10, discount=.99, tau=.01, epsilon=.01):
		self.sims = list()

		self.likelihood = None
		self.posterior = None
		self.prior = None

		self.discount = discount
		self.tau = tau
		self.epsilon = epsilon

		self.grid = grid

		self.space = space
		self.test = list()

		for i in range(space):
			for j in range(space):
				self.test.append([i,j])

		self.buildBiasEngine()


	def inferSummary(self, state, action):
		self.inferLikelihood(state, action)
		self.inferPosterior(state, action)
		self.expectedPosterior()
		# self.plotDistributions()

	def buildBiasEngine(self):
		""" 
			Simulates MDPs with varying bias to build a bias inference engine.
		"""

		print( "Loading MDPs...\n")

		# Unnecessary progress bar for terminal
		bar = pyprind.ProgBar(len(self.test))
		for i in self.test:
			self.sims.append(GridWorld(self.grid, i, self.discount, self.tau, self.epsilon))
			bar.update()

		print ("\nDone loading MDPs...")


	def inferLikelihood(self, state, action):
		"""
			Uses inference engine to inferBias predicated on an agents'
			actions and current state.
		"""

		self.state = state
		self.action = action

		self.likelihood = list()
		for i in range(len(self.sims)):
			self.likelihood.append(self.sims[i].policy[state][action])


	def inferPosterior(self, state, action, prior='uniform'):
		"""
			Uses inference engine to compute posterior probability from the 
			likelihood and prior (beta distribution).
		"""

		if prior == 'beta':
			# Beta Distribution
			self.prior = np.linspace(.01,1.0,101)
			self.prior = beta.pdf(self.prior,1.4,1.4)
			self.prior /= self.prior.sum()

		elif prior == 'shiftExponential':
			# Shifted Exponential
			self.prior = np.zeros(101)
			for i in range(50):
				self.prior[i + 50] = i * .02
			self.prior[100] = 1.0
			self.prior = expon.pdf(self.prior)
			self.prior[0:51] = 0
			self.prior *= self.prior
			self.prior /= self.prior.sum()

		elif prior == 'shiftBeta':
			# Shifted Beta
			self.prior = np.linspace(.01,1.0,101)
			self.prior = beta.pdf(self.prior,1.2,1.2)
			self.prior /= self.prior.sum()
			self.prior[0:51] = 0

		elif prior == 'uniform':
			# Uniform
			self.prior = np.zeros(len(self.sims))	
			self.prior = uniform.pdf(self.prior)
			self.prior /= self.prior.sum()


		self.posterior = self.likelihood * self.prior
		self.posterior /= self.posterior.sum()


	def plotDistributions(self):
		""" 
		Plots the prior, likelihood, and posterior distributions. 
		"""
		pass


	def expectedPosterior(self):
		"""
			Calculates expected value for the posterior distribution.
		"""
		expectation_a = 0
		expectation_b = 0
		aGreaterB = 0
		aLessB = 0
		aEqualB = 0

		x = range(len(self.posterior))

		for i in range(len(self.posterior)):

			e_a = self.test[i][0] * self.posterior[i]
			e_b = self.test[i][1] * self.posterior[i]

			expectation_a += e_a
			expectation_b += e_b

			# print "R_A: {}, R_B: {}".format(self.test[i][0], self.test[i][1])
			# print "E_a: {}".format(e_a)
			# print "E_b: {}\n".format(e_b)

			
			if self.test[i][0] > self.test[i][1]:
				aGreaterB += self.posterior[i]

			elif self.test[i][0] < self.test[i][1]:
				aLessB += self.posterior[i]

			elif self.test[i][0] == self.test[i][1]:
				aEqualB += self.posterior[i]
		
		# print aGreaterB


		print ("Chance that agent prefers A over B: {}".format(aGreaterB))
		print ("Chance that agent prefers B over A: {}".format(aLessB))
		print ("Chance that agent prefers A and B equally: {}".format(aEqualB))

		print ("Expectation of Goal A: {}".format(expectation_a))
		print ("Expectation of Goal B: {}".format(expectation_b))
