import matplotlib.pyplot as plt
import numpy as np
import random
import pyprind
from scipy.stats import beta
from scipy.stats import expon
from scipy.stats import uniform
from abc import ABCMeta
from abc import abstractmethod

class MDP(object):
    """ 
		Defines an Markov Decision Process containing:
	
		- States, s 
		- Actions, a
		- Rewards, r(s,a)
		- Transition Matrix, t(s,a,_s)

		Includes a set of abstract methods for extended class will
		need to implement.

	"""
    __metaclass__ = ABCMeta
    
    def __init__(self, states=None, actions=None, rewards=None, transition=None, discount=.99, tau=.01, epsilon=.01) -> None:
        self.s = np.array(states)
        self.a = np.array(actions)
        self.r = np.array(rewards)
        self.t = np.array(transition)
        
        self.discount = discount
        self.tau = tau
        self.epsilon = epsilon
        
        # Value iteration will update this
        self.values = None
        self.policy = None
        
    @abstractmethod
    def isTerminal(self,state):
        """
                Checks if MDP is in terminal state.
        """
        raise NotImplementedError()

    def getTransitionStatesAndProbs(self,state,action):
        """
                Return the list of transition probabilities
        """
        # t(s,a,_s)
        return self.t[state][action][:]
    
    def getReward(self,state):
        """
                Get reward for transition from state->action->nextState
        """
        return self.r[state]
    
    def takeAction(self,state,action):
        """
                Take an action in an MDP, return the next state
                
                Chooses according to probability distribution of state transitions, contingent on actions
        """
        return np.random.choice(self.s,p=self.getTransitionStatesAndProbs(state,action))
    
    def valueIteration(self):
        """
                Perform value iteration to populate the values of all states in the MDP
                
                Params:
                    -epsilon: Determines limit of convergence
        """
        # Initialize V_0 t0 zero
        self.values = np.zeros(len(self.s))
        self.policy = np.zeros([len(self.s),len(self.a)])
        
        policy_switch = 0
        
        # loop until convergence
        while True:
            
            # oldPolicy = np.argmax(self.policy, axis=1)
			# self.extractPolicy()
			# newPolicy = np.argmax(self.policy, axis=1)


			# if not np.array_equal(oldPolicy, newPolicy):
			# 	policy_switch += 1

			# print "Policy switch count: {}".format(policy_switch)

			# To be used for convergence check
            oldValues = np.copy(self.values)
            
            for i in range(len(self.s)-1):
                self.values[i] = self.r[i] + np.max(self.discount*np.dot(self.t[i][:][:],self.values))
            
            if np.max(np.abs(self.values - oldValues)) <= self.epsilon:
                break
            
    def extractPolicy(self):
        """
                Extract policy from values after value iteration runs
        """
        self.policy = np.zeros([len(self.s),len(self.a)])
        
        for i in range(len(self.s)-1):
            
            state_policy = np.zeros(len(self.a))
            
            state_policy = self.r[i] + self.discount*np.dot(self.t[i][:][:],self.values)
            
            # Softmax the policy
            state_policy -= np.max(state_policy)
            state_policy = np.exp(state_policy / float(self.tau))
            state_policy /= state_policy.sum()
            
            self.policy[i] = state_policy
            
    def extractDeterministicPolicy(self):
        """
                Extract policy from values after value iteration runs
        """
        self.policy = np.zeros(len(self.s))
        
        for i in range(len(self.s)-1):
            max_a = 0
            
            for j in range(len(self.a)):
                
                sum_nextState = 0
                for k in range(len(self.s)-1):
                    sum_nextState += self.getTransitionStatesAndProbs(i,j)[k] *\
                        (self.getReward(i) + self.discount*self.values[k])
                
                if sum_nextState > max_a:
                    max_a = sum_nextState
                    self.policy[i] = j
                    
    def simulate(self,state):
        
        """
                Runs the solver for the MDP, conducts value iteration, extracts policy,
			    then runs simulation of problem.

			    NOTE: Be sure to run value iteration (solve values for states) and to
		 	    extract some policy (fill in policy vector) before running simulation
        """
        # Run simulation using policy until terminal condition met
        while not self.isTerminal(state):
            
            # Determin which policy to use (non-deterministic)
            
            # NOTE:where返回的是两个数组元素相同的下标，但是为什么要返回[0][0]
            policy = self.policy[np.where(self.s == state)[0][0]]
            p_policy = self.policy[np.where(self.s == state)[0][0]]/self.policy[np.where(self.s==state)[0][0]].sum()
            
            # Get the parameters to perform one move
            stateIndex = np.where(self.s == state)[0][0]
            policyChoice = np.random.choice(policy,p=p_policy)
            actionIndex = np.choice(np.array(np.where(self.policy[state][:] == policyChoice)).ravel())
            
            # Take an action, move to next state
            nextState = self.takeAction(stateIndex,actionIndex)
            
            print( "In state: {}, taking action: {}, moving to state: {}".format(state, self.a[actionIndex], nextState))
            
            state = int(nextState)
            if self.isTerminal(state):
                
                return state

class BettingGame(MDP):
    
    """ 
		Defines the Betting Game:

		Problem: A gambler has the chance to make bets on the outcome of 
		a fair coin flip. If the coin is heads, the gambler wins as many
		dollars back as was staked on that particular flip - otherwise
		the money is lost. The game is won if the gambler obtains $100,
		and is lost if the gambler runs out of money (has 0$). This gambler
		did some research on MDPs and has decided to enlist them to assist
		in determination of how much money should be bet on each turn. Your 
		task is to build that MDP!

		Params: 

				startCash: Starting amount to bet with
				pHead: Probability of coin flip landing on heads
					- Use .5 for fair coin, else choose a bias [0,1]

	"""
    def __init__(self,pHeads=.5,discount=.99,epsilon=.1,tau=.001):
        
        MDP.__init__(self,discount=discount,tau=tau,epsilon=epsilon)
        self.pHeads = pHeads
        self.setBettingGame(pHeads)
        self.valueIteration()
        self.extractPolicy()
        
    def isTerminal(self, state):
        
        """
                Checks if MDP is in terminal state.
        """
        return True if state is 100 or state is 0 else False
    
    def setBettingGame(self,pHeads=.5):
        
        """ 
			Initializes the MDP to the starting conditions for 
			the betting game. 

			Params:
				startCash = Amount of starting money to spend
				pHeads = Probability that coin lands on head
					- .5 for fair coin, otherwise choose bias

		"""
    
        self.pHeads=pHeads
        self.s = np.arange(102)
        self.a = np.arange(101)
        self.r = np.zeros(101)
        self.r[0] = -5
        self.r[100] = 10
        
        # Initialize transition matrix
        temp = np.zeros([len(self.s),len(self.a),len(self.s)])
        
        self.t = [self.tHelper(i[0],i[1],i[2],self.pHeads) for i,x in np.ndenumerate(temp)]
        self.t = np.reshape(self.t,np.shape(temp))
        
        for x in range(len(self.a)):
            self.t[100][x] = np.zeros(len(self.s))
            self.t[100][x][101] = 1.0
            self.t[0][x] = np.zeros(len(self.s))
            self.t[0][x][101] = 1.0
   
    def tHelper(self,x,y,z,pHeads):
        
        if x + y is z and y is 0:
            return 1.0

        # If you bet more money than you have, no chance of any outcome
        elif y > x and x is not z:
            return 0

        # If you bet more money than you have, returns same state with 1.0 prob.
        elif y > x and x is z:
            return 1.0

        # Chance you lose
        elif x - y is z:
            return 1.0 - pHeads

        # Chance you win
        elif x + y is z:
            return pHeads

        # Edge Case: Chance you win, and winnings go over 100
        elif x + y > z and z is 100:
            return pHeads


        else:
            return 0 

        return 0
    
class InferenceMachine():
    """
            Conducts inference via MDPs for the BettingGame.
    """
    
    def __init__(self):
        self.sims = list()
        self.likelihood = None
        self.posterior = None
        self.prior = None
        
        self.e = None
        
        self.buildBiasEngine()
        
    def inferSummary(self,state,action):
        self.inferLikelihood(state,action)
        self.inferPosterior(state,action)
        print("Expected Value of Posterior Distribution: {}".format(self.expectedPosterior()))
        self.plotDistributions()
        
    def buildBiasEngine(self):
        """
                Simulates MDPs with varying bias to build a bias inference engine
        """
        
        print("Loading MDPs...\n")
        
        bar = pyprind.ProgBar(len(np.arange(0,1.01,.01)))
        for i in np.arange(0,1.01,.01):
            self.sims.append(BettingGame(i))
            bar.update()
            
        print("\nDone loading MDPs...")

    def inferLikelihood(self,state,action):
        """
                Uses inference engine to inferBias predicated on an agents' actions and current state.
        """
        
        self.state = state
        self.action = action
        
        self.likelihood = list()
        for i in range(len(self.sims)):
            self.likelihood.append(self.sims[i].policy[state][action])
            
    def inferPosterior(self,state,action):
        """
                Uses inference engine to compute posterior probability from the likelihood and prior (beta distribution).
        """
        self.prior = np.linspace(.01,1.0,101)
        self.prior = uniform.pdf(self.prior)
        self.prior /= self.prior.sum()
        self.prior[0:51] = 0
        
        self.posterior = self.likelihood * self.prior
        self.posterior /= self.posterior.sum()
        
    def plotDistributions(self):
        
        plt.figure(1)
        plt.subplot(221)
        plt.plot(np.linspace(.01,1.0,101),self.posterior)
        plt.ylabel('P(Action={}|State={})'.format(self.action, self.state))
        plt.xlabel('Bias')
        plt.title('Posterior Probability for Bias')

		# Plotting Likelihood
        plt.subplot(222)
        plt.plot(np.linspace(.01,1.0,101),self.likelihood)
        plt.ylabel('P(Action={}|State={})'.format(self.action,self.state))
        plt.xlabel('Bias')
        plt.title('Likelihood for Actions, States')

		# Plotting Prior
        plt.subplot(223)
        plt.plot(np.linspace(.01,1.0,101), self.prior)
        plt.ylabel('P(Bias)')
        plt.xlabel('Bias')
        plt.title('Prior Probability')
        plt.tight_layout()
        plt.show()
        
    def expectedPosterior(self):
        """
                Calculates expected value for the posterior distribution
        """
        expectation = 0
        x = np.linspace(.01,1.0,101)
        
        for i in range(len(self.posterior)):
            expectation += self.posterior[i] * x[i]
        
        return expectation