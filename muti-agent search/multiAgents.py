# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        if gameState.getNumFood() == 1:
            print(bestIndices)
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # # getPacmanState可以返回pacman的状态，有两个function：getDirection和getPosition可以调用
        # print(f"successor state:\n {successorGameState.getPacmanState().getDirection()}")
        # print(f"newPos: {newPos}")
        # # newFood.aslist()返回一个食物地址列表
        # print(f"newFood: {newFood.asList()}")
        # # newGhostStates是一个列表，列表中是每一个ghost的state
        # print(f"newGhostStates: {newGhostStates}")
        # # newScaredTime是一个列表，返回每一个ghost的scared time
        # print(f"newScaredTime: {newScaredTimes}")

        if (successorGameState.getNumFood() != 0):

            # 距离最近食物距离
            foodDistance = min([util.manhattanDistance(newPos, newFood.asList()[
                               i]) for i in range(len(newFood.asList()))])

            # 距离所有没有被scared的ghost的最小距离
            # 每一个ghost的newPos
            newGhostPosList = [newGhostStates[i].getPosition()
                               for i in range(len(newGhostStates))]
            # 计算离ghost的距离，如果ghost没有被scare则取最小的距离，这个最小的距离越大越好，如果被scared就直接取100
            ghostDistance = min([util.manhattanDistance(newPos, newGhostPosList[i]) for i in range(
                len(newGhostPosList)) if newScaredTimes[i] == 0] or [100])

            return successorGameState.getScore() + 1.0 / foodDistance + 0.02 * ghostDistance

        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def Minimax(self, gameState, agentIndex, depth):
        """
        该函数返回agent根据miniMax策略获得的值,和返回所采取的action
        params:
            gameState:当前状态
            agentIndex:0为pacman,>=1为ghost
            depth:当前递归深度
        """
        if gameState.isWin() or gameState.isLose() or depth == 0:
            res = self.evaluationFunction(gameState), Directions.STOP

        elif agentIndex == 0:
            res = self.Maxmize(gameState, agentIndex, depth)
        else:
            res = self.Minimize(gameState, agentIndex, depth)

        return res

    def Maxmize(self, gameState, agentIndex, depth):
        if agentIndex != gameState.getNumAgents()-1:
            nextAgentIndex = agentIndex + 1
            # agent之间交替做决策算是一层，depth不需要+1
            nextDepth = depth
        if agentIndex == gameState.getNumAgents()-1:
            nextAgentIndex = 0
            # 去到下一层
            nextDepth = depth - 1
        v = -1e9
        bestAction = Directions.STOP
        pacmanActions = gameState.getLegalActions(agentIndex)
        for action in pacmanActions:
            Nextvalue = self.Minimax(gameState.generateSuccessor(
                agentIndex, action), nextAgentIndex, nextDepth)[0]
            if v < Nextvalue:
                v = Nextvalue
                bestAction = action

        return v, bestAction

    def Minimize(self, gameState, agentIndex, depth):
        if agentIndex != gameState.getNumAgents()-1:
            nextAgentIndex = agentIndex + 1
            nextDepth = depth
        if agentIndex == gameState.getNumAgents()-1:
            nextAgentIndex = 0
            nextDepth = depth - 1
        v = 1e9
        bestAction = Directions.STOP
        ghostActions = gameState.getLegalActions(agentIndex)
        for action in ghostActions:
            Nextvalue = self.Minimax(gameState.generateSuccessor(
                agentIndex, action), nextAgentIndex, nextDepth)[0]
            if v > Nextvalue:
                v = Nextvalue
                bestAction = action

        return v, bestAction

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        return self.Minimax(gameState, 0, self.depth)[1]
        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def Minimax(self, gameState, agentIndex, depth, alpha, beta):
        """
        该函数返回agent根据miniMax策略获得的值,和返回所采取的action
        params:
            gameState:当前状态
            agentIndex:0为pacman,>=1为ghost
            depth:当前递归深度
        """
        if gameState.isWin() or gameState.isLose() or depth == 0:
            res = self.evaluationFunction(gameState), Directions.STOP

        elif agentIndex == 0:
            res = self.Maxmize(gameState, agentIndex, depth, alpha, beta)
        else:
            res = self.Minimize(gameState, agentIndex, depth, alpha, beta)

        return res

    def Maxmize(self, gameState, agentIndex, depth, alpha, beta):
        if agentIndex != gameState.getNumAgents()-1:
            nextAgentIndex = agentIndex + 1
            # agent之间交替做决策算是一层，depth不需要+1
            nextDepth = depth
        if agentIndex == gameState.getNumAgents()-1:
            nextAgentIndex = 0
            # 去到下一层
            nextDepth = depth - 1
        v = -1e9
        bestAction = Directions.STOP
        pacmanActions = gameState.getLegalActions(agentIndex)
        for action in pacmanActions:
            Nextvalue = self.Minimax(gameState.generateSuccessor(
                agentIndex, action), nextAgentIndex, nextDepth, alpha, beta)[0]
            if v < Nextvalue:
                v = Nextvalue
                alpha = max(alpha, v)
                bestAction = action
            if v > beta:
                return v, action

        return v, bestAction

    def Minimize(self, gameState, agentIndex, depth, alpha, beta):
        if agentIndex != gameState.getNumAgents()-1:
            nextAgentIndex = agentIndex + 1
            nextDepth = depth
        if agentIndex == gameState.getNumAgents()-1:
            nextAgentIndex = 0
            nextDepth = depth - 1
        v = 1e9
        bestAction = Directions.STOP
        ghostActions = gameState.getLegalActions(agentIndex)
        for action in ghostActions:
            Nextvalue = self.Minimax(gameState.generateSuccessor(
                agentIndex, action), nextAgentIndex, nextDepth, alpha, beta)[0]
            if v > Nextvalue:
                v = Nextvalue
                beta = min(beta, v)
                bestAction = action
            if v < alpha:
                return v, action

        return v, bestAction

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.Minimax(gameState, 0, self.depth, -1e9, 1e9)[1]
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def Minimax(self, gameState, agentIndex, depth):
        """
        该函数返回agent根据miniMax策略获得的值,和返回所采取的action
        params:
            gameState:当前状态
            agentIndex:0为pacman,>=1为ghost
            depth:当前递归深度
        """
        if gameState.isWin() or gameState.isLose() or depth == 0:
            res = self.evaluationFunction(gameState), Directions.STOP

        elif agentIndex == 0:
            res = self.Maxmize(gameState, agentIndex, depth)
        else:
            res = self.Expectimax(gameState, agentIndex, depth)

        return res

    def Maxmize(self, gameState, agentIndex, depth):
        if agentIndex != gameState.getNumAgents()-1:
            nextAgentIndex = agentIndex + 1
            # agent之间交替做决策算是一层，depth不需要+1
            nextDepth = depth
        if agentIndex == gameState.getNumAgents()-1:
            nextAgentIndex = 0
            # 去到下一层
            nextDepth = depth - 1
        v = -1e9
        bestAction = Directions.STOP
        pacmanActions = gameState.getLegalActions(agentIndex)
        for action in pacmanActions:
            Nextvalue = self.Minimax(gameState.generateSuccessor(
                agentIndex, action), nextAgentIndex, nextDepth)[0]
            if v < Nextvalue:
                v = Nextvalue
                bestAction = action

        return v, bestAction

    def Expectimax(self, gameState, agentIndex, depth):
        if agentIndex != gameState.getNumAgents()-1:
            nextAgentIndex = agentIndex + 1
            nextDepth = depth
        if agentIndex == gameState.getNumAgents()-1:
            nextAgentIndex = 0
            nextDepth = depth - 1
        v = 0
        bestAction = Directions.STOP
        ghostActions = gameState.getLegalActions(agentIndex)
        probability = 1 / len(ghostActions)
        for action in ghostActions:
            Nextvalue = self.Minimax(gameState.generateSuccessor(
                agentIndex, action), nextAgentIndex, nextDepth)[0]
            v += probability * Nextvalue

        bestAction = random.choice(ghostActions)

        return v, bestAction

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.Minimax(gameState, 0, self.depth)[1]
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    # ghostPos是一个列表，存储着所有ghost的坐标
    ghostPos = currentGameState.getGhostPositions()
    foodPos = currentGameState.getFood().asList()
    # 距离最近食物距离
    foodDistance = 0.001
    if not currentGameState.isWin():
        foodDistance = min([util.manhattanDistance(
            pacmanPos, foodPos[i]) for i in range(len(foodPos))])
    # 计算离ghost的距离,这个最小的距离越大越好
    ghostDistance = min([util.manhattanDistance(
        pacmanPos, ghostPos[i]) for i in range(len(ghostPos))])
    return 10.0 / foodDistance + 0.1 * ghostDistance + currentGameState.getScore()

    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
