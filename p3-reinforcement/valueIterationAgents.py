# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # 迭代self.iterations次
        
        for _ in range(self.iterations):
            # print(self.mdp.getStates())
            # 这里建立一个新的table避免就地更新：也就是直接更新self.values的值；因为更新后的值会影响其他位置的计算；两种方式都可以收敛到正确值，并且就地更新的速度更快
            Vtable = util.Counter()
            for state in self.mdp.getStates()[1:]:
                # print(state)
                ActionsList= self.mdp.getPossibleActions(state)
                AnsList = []
                for action in ActionsList:
                    nextStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state,action)
                    # print(f'action: {action}')
                    # print(nextStatesAndProbs)
                    # 不根据概率分布来挑选洗一个可能的state，而是求加权平均
                    ans = 0.0
                    for item in nextStatesAndProbs:
                        statePrime, prob = item
                        reward = self.mdp.getReward(state,action,statePrime)
                        # 套用公式计算当前state的value
                        ans += prob * (reward + self.discount * self.values[statePrime])
                        # print(f'ans:{ans}')
                        
                    AnsList.append(ans)
                Vtable[state] = max(AnsList)
                # print(f'state: {state}, AnsList: {AnsList}')
                # print(f'self.values[state]: {self.values[state]}')
            self.values = Vtable
                    
                    


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        nextStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state,action)
        qvalue = 0
        for item in nextStatesAndProbs:
            statePrime, prob = item
            reward = self.mdp.getReward(state,action,statePrime)
            qvalue += prob * (reward + self.discount * self.values[statePrime])
        return qvalue            
        
        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        policy = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            policy[action] = self.getQValue(state,action)
        
        return policy.argMax()
            
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # 计算predecessor
        predecessors = {}
        for state in self.mdp.getStates():
            ActionsList= self.mdp.getPossibleActions(state)
            for action in ActionsList:
                nextStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state,action)
                for item in nextStatesAndProbs:
                    nextState, _ = item
                    if nextState in predecessors:
                        predecessors[nextState].add(state)
                    else:
                        predecessors[nextState] = {state}              
        #初始化一个优先队列 
        PQueue = util.PriorityQueue()
        for state in self.mdp.getStates()[1:]:
            bestAction = self.getPolicy(state)
            maxQvalue = self.getQValue(state,bestAction)
            diff = abs(self.values[state] - maxQvalue)
            # print(f'diff is : {diff}')
            PQueue.update(state,-diff)
        
        for _ in range(self.iterations):
            if PQueue.isEmpty():
                break
            state = PQueue.pop()
            bestAction = self.getPolicy(state)
            # value值就等于最大的qvalue值
            stateValue = self.getQValue(state,bestAction)
            self.values[state] = stateValue
            # print(f'self.values:{self.values[state]}')
            for p in predecessors[state]:
                bestAction = self.getPolicy(p)
                maxQvalue = self.getQValue(p,bestAction)
                diff = abs(self.values[p] - maxQvalue)
                # print(f'diff is : {diff}')
                if diff > self.theta:
                    PQueue.update(p,-diff)