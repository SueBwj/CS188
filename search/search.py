# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # 用Stack来做DFS -> 用Stack就不用额外priority，因为stack的属性决定了每次取出最深的元素
    state_fringe = util.Stack()
    visited = set() #如果不设置visited可能会进入死胡同，即一直在两个走过的点中重复
    # 将初始位置和action list[]放入到fringe中
    state_fringe.push((problem.getStartState(),[]))
    # 当fringe不为空的时候循环
    while(not(state_fringe.isEmpty())):
        current_state, actions = state_fringe.pop()
        # 访问过则不再访问
        if current_state in visited:
            continue
        visited.add(current_state)
        # 到达destination则停止
        if(problem.isGoalState(current_state)):
            return actions
        
        else:
            # 当前状态的successor，将当前状态的子状态和action放入到fringe中
            successor_states = problem.getSuccessors(current_state)
            for state, action, _ in successor_states:
                state_fringe.push((state,actions + [action]))
    return []
            
        
        


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # 用Queue来做BFS
    state_fringe = util.Queue()
    visited = set()
    # 将初始位置和action list[]放入到fringe中
    state_fringe.push((problem.getStartState(),[]))
    # 当fringe不为空的时候循环
    while(not(state_fringe.isEmpty())):
        current_state, actions = state_fringe.pop()
        # 访问过则不再访问
        if current_state in visited:
            continue
        visited.add(current_state)
        # 到达destination则停止
        print(current_state)
        if(problem.isGoalState(current_state)):
            return actions
        
        else:
            # 当前状态的successor，将当前状态的子状态和action放入到fringe中
            successor_states = problem.getSuccessors(current_state)
            for state, action, _ in successor_states:
                state_fringe.push((state,actions + [action]))
    return []


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # 用PriorityQueue来做UCS
    state_fringe = util.PriorityQueue()
    visited = set()
    acm_cost = 0 # 表示初始的cost
    # 将初始位置和action list[]和priority放入到fringe中
    state_fringe.update((problem.getStartState(),[]),acm_cost)
    # 当fringe不为空的时候循环
    while(not(state_fringe.isEmpty())):
        current_state, actions = state_fringe.pop()
        # 访问过则不再访问
        if current_state in visited:
            continue
        visited.add(current_state)
        # 到达destination则停止
        if(problem.isGoalState(current_state)):
            return actions
        
        else:
            # 当前状态的successor，将当前状态的子状态和action放入到fringe中
            successor_states = problem.getSuccessors(current_state)
            for state, action, cost in successor_states:
                state_fringe.update((state,actions + [action] ), problem.getCostOfActions(actions) + cost)
    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
   # 用PriorityQueue来做A*
    state_fringe = util.PriorityQueue()
    visited = set()
    acm_cost = 0 + heuristic(problem.getStartState(),problem) # 表示初始的cost
    # 将初始位置和action list[]和priority放入到fringe中
    state_fringe.update((problem.getStartState(),[]),acm_cost)
    # 当fringe不为空的时候循环
    while(not(state_fringe.isEmpty())):
        current_state, actions = state_fringe.pop()
        # 访问过则不再访问
        if current_state in visited:
            continue
        visited.add(current_state)
        # 到达destination则停止
        if(problem.isGoalState(current_state)):
            return actions
        
        else:
            # 当前状态的successor，将当前状态的子状态和action放入到fringe中
            successor_states = problem.getSuccessors(current_state)
            for state, action, cost in successor_states:
                state_fringe.update((state,actions + [action] ), problem.getCostOfActions(actions) + cost + heuristic(state,problem))
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
