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
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """
    def __init__(self,start,goal,barrier):
        self.start = start
        self.goal = goal
        self.barrier = barrier

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        return self.start

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        if (state == self.goal):
            return True
        return False

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        x,y = state
        successors = []
        actions = {
            "North": (x,y+1),
            "South": (x,y-1),
            "East": (x+1,y),
            "West": (x-1,y),
            "Stop": (state)
        }

        for action, position in actions.items():
            if position not in self.barrier:
                successors.append((position, action, 1))
        return successors

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        if actions is None:
          return float('inf')  # Invalid actions = Infinite cost

        # Define action costs (customizable for different environments)
        action_costs = {"North": 1, "South": 1, "East": 2, "West": 2}

        cost = 0
        for action in actions:
            if action not in action_costs:
                return float('inf')  # Illegal move detected!
            cost += action_costs[action]  # Use predefined cost

        return cost


def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    #si da fallo es aki lol
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
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
    from util import Stack
    
    # Initialize a stack for DFS
    stack = Stack()

    # Push the starting state with an empty path and a cost of 0
    start_state = problem.getStartState()
    stack.push((start_state, [], 0))

    # Maintain a set to track visited nodes
    visited = set()

    while not stack.isEmpty():
        # Pop the top state from the stack
        state, path, cost = stack.pop()

        # If the state is the goal, return the path
        if problem.isGoalState(state):
            return path

        # If the state has not been visited, process it
        if state not in visited:
            visited.add(state)

            # Get successors and push them onto the stack
            for successor, action, step_cost in problem.getSuccessors(state):
                if successor not in visited:
                    new_path = path + [action]
                    stack.push((successor, new_path, cost + step_cost))

    return []  # Return empty list if no solution is found

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""


def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""


def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def exploration(problem):
     # Stack for DFS traversal
    from util import Stack
    stack = Stack()

    # Get the starting state
    start_state = problem.getStartState()
    stack.push((start_state, []))  # (current state, path taken)

    # Set to track visited states
    visited = set()

    # List to store the final path
    exploration_path = []

    while not stack.isEmpty():
        # Pop the top state from the stack
        state, path = stack.pop()

        # If this state has not been visited
        if state not in visited:
            visited.add(state)
            exploration_path.extend(path)  # Add path to exploration

            # Get all valid successors (next states that are not walls)
            for successor, action, step_cost in problem.getSuccessors(state):
                if successor not in visited:
                    new_path = path + [action]
                    stack.push((successor, new_path))

    return exploration_path  # Return a full exploration path




#Actividad 1:

def exploracion(problem: SearchProblem):
    from util import Queue

    queue = Queue()
    start_state = problem.getStartState()
    queue.push((start_state, [], 0))  # Estado, lista de acciones, costo acumulado

    explored = set()

    while not queue.isEmpty():
        state, actions, cost = queue.pop()

        if problem.isGoalState(state):
            return actions

        if state not in explored:
            explored.add(state)

            for successor, action, step_cost in problem.getSuccessors(state):
                new_cost = cost + step_cost
                new_actions = actions + [action]
                queue.push((successor, new_actions, new_cost))

    return []

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import Queue, PriorityQueue
    fringe = PriorityQueue()  # Fringe to manage which states to expand
    fringe.push(problem.getStartState(), 0)
    currState = fringe.pop()
    visited = []  # List to check whether state has already been visited
    tempPath = []  # Temp variable to get intermediate paths
    path = []  # List to store final sequence of directions
    pathToCurrent = PriorityQueue()  # Queue to store direction to children (currState and pathToCurrent go hand in hand)
    while not problem.isGoalState(currState):
        if currState not in visited:
            visited.append(currState)
            successors = problem.getSuccessors(currState)
            for child, direction, cost in successors:
                tempPath = path + [direction]
                costToGo = problem.getCostOfActions(tempPath) + heuristic(child, problem)
                if child not in visited:
                    fringe.push(child, costToGo)
                    pathToCurrent.push(tempPath, costToGo)
        currState = fringe.pop()
        path = pathToCurrent.pop()
    return path

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
exp = exploration
