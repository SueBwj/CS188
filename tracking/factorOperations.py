# factorOperations.py
# -------------------
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

from typing import List
from bayesNet import Factor
import functools
from util import raiseNotDefined

def joinFactorsByVariableWithCallTracking(callTrackingList=None):


    def joinFactorsByVariable(factors: List[Factor], joinVariable: str):
        """
        Input factors is a list of factors.
        Input joinVariable is the variable to join on.

        This function performs a check that the variable that is being joined on 
        appears as an unconditioned variable in only one of the input factors.

        Then, it calls your joinFactors on all of the factors in factors that 
        contain that variable.

        Returns a tuple of 
        (factors not joined, resulting factor from joinFactors)
        """

        if not (callTrackingList is None):
            callTrackingList.append(('join', joinVariable))

        currentFactorsToJoin =    [factor for factor in factors if joinVariable in factor.variablesSet()]
        currentFactorsNotToJoin = [factor for factor in factors if joinVariable not in factor.variablesSet()]

        # typecheck portion
        numVariableOnLeft = len([factor for factor in currentFactorsToJoin if joinVariable in factor.unconditionedVariables()])
        if numVariableOnLeft > 1:
            print("Factor failed joinFactorsByVariable typecheck: ", factor)
            raise ValueError("The joinBy variable can only appear in one factor as an \nunconditioned variable. \n" +  
                               "joinVariable: " + str(joinVariable) + "\n" +
                               ", ".join(map(str, [factor.unconditionedVariables() for factor in currentFactorsToJoin])))
        
        joinedFactor = joinFactors(currentFactorsToJoin)
        return currentFactorsNotToJoin, joinedFactor

    return joinFactorsByVariable

joinFactorsByVariable = joinFactorsByVariableWithCallTracking()

########### ########### ###########
########### QUESTION 2  ###########
########### ########### ###########

def joinFactors(factors: List[Factor]):
    """
    Input factors is a list of factors.  
    
    You should calculate the set of unconditioned variables and conditioned (无条件，有条件)
    variables for the join of those factors.

    Return a new factor that has those variables and whose probability entries 
    are product of the corresponding rows of the input factors.

    You may assume that the variableDomainsDict for all the input 
    factors are the same, since they come from the same BayesNet.

    joinFactors will only allow unconditionedVariables to appear in 
    one input factor (so their join is well defined).

    Hint: Factor methods that take an assignmentDict as input 
    (such as getProbability and setProbability) can handle 
    assignmentDicts that assign more variables than are in that factor.
    
    将 assignmentDict 作为输入 
    的因子方法(如 getProbability 和 setProbability)可以处理分配的变量多于该因子中的变量的 
    赋值字典。

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """

    # typecheck portion
    setsOfUnconditioned = [set(factor.unconditionedVariables()) for factor in factors]
    if len(factors) > 1:
        intersect = functools.reduce(lambda x, y: x & y, setsOfUnconditioned)
        if len(intersect) > 0:
            print("Factor failed joinFactors typecheck: ", factor)
            raise ValueError("unconditionedVariables can only appear in one factor. \n"
                    + "unconditionedVariables: " + str(intersect) + 
                    "\nappear in more than one input factor.\n" + 
                    "Input factors: \n" +
                    "\n".join(map(str, factors)))


    "*** YOUR CODE HERE ***"
    # 构造Factor需要inputUnconditionedVariables, inputConditionedVariables, inputVariableDomainsDict
    # for factor in factors:
        # print(factor.getAllPossibleAssignmentDicts())
        # print(factor.conditionedVariables())
        # print(factor.unconditionedVariables())
        # print(factor.variableDomainsDict())
        # for item in factor.getAllPossibleAssignmentDicts():
        #     print(factor.getProbability(item))
        # print('=====================')
    
    # 根据题目中的hint，发现unconditioned变量就是每一组factor中unconditioned变量的并集，同理conditioned变量
    # 变量域同一个字典中的factor都相同，所以直接赋值即可
    # 1. 取并集获得newFactor的unconditioned和conditioned
    # 2. 计算newFactor的变量域的概率并赋值
    unconditionedVariable = set()
    conditionedVariable = set()
    inputDomain = dict()
    
    for factor in factors:
        
        inputDomain = factor.variableDomainsDict()
        
        for everyUncondition in factor.unconditionedVariables():
            unconditionedVariable.add(everyUncondition)
        for everyCondition in factor.conditionedVariables():
            conditionedVariable.add(everyCondition)
            
    # 不用copy会报错
    for conditioned in conditionedVariable.copy():
        if conditioned in unconditionedVariable:
            conditionedVariable.remove(conditioned)
    
    newFactor = Factor(unconditionedVariable,conditionedVariable,inputDomain)
    
    all_possible_assignments = newFactor.getAllPossibleAssignmentDicts()
    for assignment in all_possible_assignments:
        prob = 1.0
        # 对于每个因子，找到对应赋值的概率并乘以当前的概率
        for factor in factors:
            prob *= factor.getProbability(assignment)
        newFactor.setProbability(assignment, prob)
    
    return newFactor         
    

    "*** END YOUR CODE HERE ***"

########### ########### ###########
########### QUESTION 3  ###########
########### ########### ###########

def eliminateWithCallTracking(callTrackingList=None):

    def eliminate(factor: Factor, eliminationVariable: str):
        """
        Input factor is a single factor.
        Input eliminationVariable is the variable to eliminate from factor.
        eliminationVariable must be an unconditioned variable in factor.
        
        You should calculate the set of unconditioned variables and conditioned 
        variables for the factor obtained by eliminating the variable
        eliminationVariable.

        Return a new factor where all of the rows mentioning
        eliminationVariable are summed with rows that match
        assignments on the other variables.

        Useful functions:
        Factor.getAllPossibleAssignmentDicts
        Factor.getProbability
        Factor.setProbability
        Factor.unconditionedVariables
        Factor.conditionedVariables
        Factor.variableDomainsDict
        """
        # autograder tracking -- don't remove
        if not (callTrackingList is None):
            callTrackingList.append(('eliminate', eliminationVariable))

        # typecheck portion
        if eliminationVariable not in factor.unconditionedVariables():
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Elimination variable is not an unconditioned variable " \
                            + "in this factor\n" + 
                            "eliminationVariable: " + str(eliminationVariable) + \
                            "\nunconditionedVariables:" + str(factor.unconditionedVariables()))
        
        if len(factor.unconditionedVariables()) == 1:
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Factor has only one unconditioned variable, so you " \
                    + "can't eliminate \nthat variable.\n" + \
                    "eliminationVariable:" + str(eliminationVariable) + "\n" +\
                    "unconditionedVariables: " + str(factor.unconditionedVariables()))

        "*** YOUR CODE HERE ***"
        # 思路和上一题一致，只不过计算概率是joint和eliminate的逻辑不同
        unconditionedVariable = set()
        conditionedVariable = set()
        inputDomain = dict()
        
        unconditionedVariable = factor.unconditionedVariables()
        # print(unconditionedVariable)
        unconditionedVariable.remove(eliminationVariable)
        
        conditionedVariable = factor.conditionedVariables()
        
        inputDomain = factor.variableDomainsDict()
        inputDomain.pop(eliminationVariable)
        
        newFactor = Factor(unconditionedVariable,conditionedVariable,inputDomain)
        
        factor_assignments = factor.getAllPossibleAssignmentDicts()
        for factor_assignment in factor_assignments:
            prob = newFactor.getProbability(factor_assignment)
            prob += factor.getProbability(factor_assignment)
            newFactor.setProbability(factor_assignment,prob)
        
        return newFactor  
        "*** END YOUR CODE HERE ***"

    return eliminate

eliminate = eliminateWithCallTracking()

