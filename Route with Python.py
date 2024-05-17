import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Constants
GRIDSIZE = 40  # Grid size as required by user
PopSIZE = 100
MUTATIONRATE = 0.01
GENERATIONS = 100

# Grid representing the map of Lahore
obstacles = [(5, 5), (10, 15), (20, 25)]  # Obstacle locations (Change according to user)
start = (2, 3)
destination = (38, 39)

# Function to calculate distance between two points
def distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

# Function to create initial Pop
def IntializePop(size):
    Pop = []
    for _ in range(size):
        route = [start]
        CurrentPos = start
        while CurrentPos != destination:
            MovesPossible = [(CurrentPos[0] + 1, CurrentPos[1]), (CurrentPos[0] - 1, CurrentPos[1]),
                              (CurrentPos[0], CurrentPos[1] + 1), (CurrentPos[0], CurrentPos[1] - 1)]
            MovesPossible = [move for move in MovesPossible if move[0] >= 0 and move[0] < GRIDSIZE and
                              move[1] >= 0 and move[1] < GRIDSIZE and move not in obstacles]
            CurrentPos = random.choice(MovesPossible)
            route.append(CurrentPos)
        Pop.append(route)
    return Pop

# Function to evaluate fitness of each route
def FitnessEvaluation(route):
    return -distance(route[-1], destination)  # Fitness based on distance to destination

# Function for tournament selection
def SelectionArea(Pop, k=3):
    selected = []
    for _ in range(len(Pop)):
        participants = random.sample(Pop, k)
        winner = max(participants, key=FitnessEvaluation)
        selected.append(winner)
    return selected

# Function for one-point Crossover
def Crossover(parent1, parent2):
    CrossoverPoint = random.randint(1, min(len(parent1), len(parent2)) - 1)
    child1 = parent1[:CrossoverPoint] + parent2[CrossoverPoint:]
    child2 = parent2[:CrossoverPoint] + parent1[CrossoverPoint:]
    return child1, child2

# Function for mutation
def Mutate(route):
    MutatedRoute = route[:]
    for i in range(len(MutatedRoute)):
        if random.random() < MUTATIONRATE:
            MovesPossible = [(MutatedRoute[i][0] + 1, MutatedRoute[i][1]), (MutatedRoute[i][0] - 1, MutatedRoute[i][1]),
                              (MutatedRoute[i][0], MutatedRoute[i][1] + 1), (MutatedRoute[i][0], MutatedRoute[i][1] - 1)]
            MovesPossible = [move for move in MovesPossible if move[0] >= 0 and move[0] < GRIDSIZE and
                              move[1] >= 0 and move[1] < GRIDSIZE and move not in obstacles]
            MutatedRoute[i] = random.choice(MovesPossible)
    return MutatedRoute

# Genetic Algorithm
def GA():
    Pop = IntializePop(PopSIZE)
    for generation in range(GENERATIONS):
        Pop = SelectionArea(Pop)
        NextGen = []
        while len(NextGen) < PopSIZE:
            parent1, parent2 = random.sample(Pop, 2)
            child1, child2 = Crossover(parent1, parent2)
            child1 = Mutate(child1)
            child2 = Mutate(child2)
            NextGen.extend([child1, child2])
        Pop = NextGen
    return max(Pop, key=FitnessEvaluation)

# A* Search Algorithm (for comparison)
def Astar():
    graph = nx.grid_2d_graph(GRIDSIZE, GRIDSIZE)
    for obstacle in obstacles:
        graph.remove_node(obstacle)
    ShortestPath = nx.astar_path(graph, start, destination)
    return ShortestPath

def GA_time_complexity():
    # Time complexity calculation
    # Selection Area complexity: O(PopSIZE * len(Pop))
    # Crossover complexity: O(PopSIZE)
    # Mutation complexity: O(PopSIZE * len(route))
    # Overall complexity: O(GENERATIONS * (PopSIZE^2 * len(Pop) + PopSIZE * len(route)))
    complexity = GENERATIONS * (PopSIZE**2 * (GRIDSIZE**2) + PopSIZE * (GRIDSIZE**2))
    return complexity


# Main function
def main():
    OptimizedGA = GA()
    OptimizedAstar = Astar()

    print("Optimized Route using Genetic Algorithm:", OptimizedGA)
    print("Optimized Route using A* Search Algorithm:", OptimizedAstar)
    print("Time complexity of Genetic Algorithm:", GA_time_complexity())


    # Visualization for comparison
    grid = np.zeros((GRIDSIZE, GRIDSIZE))
    for obstacle in obstacles:
        grid[obstacle] = 1
    grid[start] = 2
    grid[destination] = 3

    plt.imshow(grid, cmap='viridis', origin='lower')
    plt.plot([point[1] for point in OptimizedGA], [point[0] for point in OptimizedGA], 'r-')
    plt.plot([point[1] for point in OptimizedAstar], [point[0] for point in OptimizedAstar], 'b-')
    plt.title('Optimized Route')
    plt.show()

if __name__ == "__main__":
    main()
