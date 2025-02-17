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
In search.py, we implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import math

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).
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
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """Explore deepest paths first with cycle prevention."""

    # Node storage (LIFO) containing (position, path, level)
    stateStack = util.Stack()
    visitedStates = []  # Track processed states
    depthLimit = 10  # Exploration boundary

    fringeTracker = 0  # Was maxFringeSize
    expansionCounter = 0  # Was nodesExpanded

    initial = problem.getStartState()
    stateStack.push((initial, [], 0))

    while not stateStack.isEmpty():
        fringeTracker = max(fringeTracker, len(stateStack.list))
        position, pathSteps, currentLevel = stateStack.pop()
        expansionCounter += 1

        if position not in visitedStates:
            visitedStates.append(position)

            if problem.isGoalState(position):
                return pathSteps, fringeTracker, expansionCounter
            else:
                for nextPos, move, _ in problem.getSuccessors(position):
                    updatedPath = pathSteps + [move]
                    if (currentLevel + 1) <= depthLimit:
                        stateStack.push((nextPos, updatedPath, currentLevel + 1))

    return [], fringeTracker, expansionCounter


def breadthFirstSearch(problem):
    """Explore nearest nodes first with cost tracking."""

    nodeQueue = util.Queue()  # Was frontier
    processedStates = []  # Was exploredNodes
    fringeMax = 0  # Was maxFringeSize
    nodeCount = 0  # Was nodesExpanded

    start = problem.getStartState()
    nodeQueue.push((start, [], 0))

    while not nodeQueue.isEmpty():
        fringeMax = max(fringeMax, len(nodeQueue.list))
        currentPos, actions, accumulatedCost = nodeQueue.pop()
        nodeCount += 1

        if currentPos not in processedStates:
            processedStates.append(currentPos)

            if problem.isGoalState(currentPos):
                return actions, fringeMax, nodeCount
            else:
                for neighbor, direction, cost in problem.getSuccessors(currentPos):
                    newActions = actions + [direction]
                    totalCost = accumulatedCost + cost
                    nodeQueue.push((neighbor, newActions, totalCost))

    return [], fringeMax, nodeCount


def uniformCostSearch(problem):
    """Prioritize minimal cost paths with state tracking."""

    costQueue = util.PriorityQueue()  # Was frontier
    stateCosts = {}  # Was exploredNodes
    maxFringe = 0  # Was maxFringeSize
    processedNodes = 0  # Was nodesExpanded

    origin = problem.getStartState()
    costQueue.push((origin, [], 0), 0)

    while not costQueue.isEmpty():
        maxFringe = max(maxFringe, len(costQueue.heap))
        currentPos, path, total = costQueue.pop()
        processedNodes += 1

        if problem.isGoalState(currentPos):
            return path, maxFringe, processedNodes

        if (currentPos not in stateCosts) or (total < stateCosts.get(currentPos, float('inf'))):
            stateCosts[currentPos] = total

            for nextPos, action, stepCost in problem.getSuccessors(currentPos):
                updatedPath = path + [action]
                newTotal = total + stepCost
                costQueue.update((nextPos, updatedPath, newTotal), newTotal)

    return [], maxFringe, processedNodes


def h1_misplaced_tiles(state, problem=None):
    misplaced = 0
    goal = [[1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 0]]
    for i in range(4):
        for j in range(4):
            if state.cells[i][j] != 0 and state.cells[i][j] != goal[i][j]:
                misplaced += 1
    return misplaced


def h2_euclidean_distance(state, problem=None):
    """Computes the Euclidean distance heuristic for the 15-puzzle."""

    total_distance = 0

    for row in range(4):
        for col in range(4):
            tile = state.cells[row][col]
            if tile != 0:  # Ignore blank space
                goal_row, goal_col = divmod(tile - 1, 4)
                total_distance += math.sqrt((goal_row - row) ** 2 + (goal_col - col) ** 2)

    return total_distance

def h3_manhattan_distance(state, problem=None):
    """Computes the Euclidean distance heuristic for the 15-puzzle."""

    total_distance = 0

    for row in range(4):
        for col in range(4):
            tile = state.cells[row][col]
            if tile != 0:  # Ignore blank space
                goal_row=(tile-1)//4
                goal_col=(tile-1)%4
                total_distance +=abs(goal_row-row)+abs(goal_col-col)

    return total_distance
def h4_tiles_out(state,problem=None):
    misplaced_row = misplaced_col = 0
    for row in range(4):
        for col in range(4):
            tile=state.cells[row][col]
            if tile!=0:
                goal_row=(tile-1)//4
                goal_col=(tile-1)%4
                if(row!=goal_row):
                    misplaced_row+=1
                if(col!=goal_col):
                    misplaced_col+=1
    return misplaced_row + misplaced_col


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic,track_fringe=None, track_expansion=None):
    """Search the node that has the lowest combined cost and heuristic first."""

    #to be explored (FIFO): takes in item, cost+heuristic
    fringe = util.PriorityQueue()

    exploredNodes = [] #holds (state, cost)

    startState = problem.getStartState()
    startNode = (startState, [], 0) #(state, action, cost)

    fringe.push(startNode, heuristic(startState,problem))

    while not fringe.isEmpty():
        if track_fringe:
            track_fringe(fringe)  # Track fringe size
        #begin exploring first (lowest-combined (cost+heuristic) ) node on frontier
        currentState, actions, currentCost = fringe.pop()

        #put popped node into explored list
        currentNode = (currentState, currentCost)
        exploredNodes.append((currentState, currentCost))

        if problem.isGoalState(currentState):
            return actions
        exploredNodes.append((currentState,currentCost))
        if track_expansion:
            track_expansion()

        #list of (successor, action, stepCost)
        successors = problem.getSuccessors(currentState)

        #examine each successor
        for succState, succAction, succCost in successors:
            newAction = actions + [succAction]
            newCost = problem.getCostOfActions(newAction)
            newNode = (succState, newAction, newCost)

            #check if this successor has been explored
            already_explored = False
            for explored in exploredNodes:
                #examine each explored node tuple
                exploredState, exploredCost = explored

                if (succState == exploredState) and (newCost >= exploredCost):
                    already_explored = True

            #if this successor not explored, put on frontier and explored list
            if not already_explored:
                fringe.push(newNode, newCost + heuristic(succState, problem))
                exploredNodes.append((succState, newCost))

    return actions


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
