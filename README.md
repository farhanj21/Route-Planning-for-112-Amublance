# Path Planning with Genetic Algorithm and A* Search
# Using Q1.ipynb file

This code implements Genetic Algorithm and A* Search, to find the shortest path from a start point to a destination point on a grid-based map. 
The grid represents a map of Lahore, with specified obstacle locations, start point, and destination point.

## Dependencies
- Python 3.11.2
- NumPy 1.26.4
- Matplotlib 3.8.3
- NetworkX  3.2.1

## Installation
1. Make sure you have Python installed and preferably Visual Studio Code or Google Collab.
2. Install the required dependencies using pip: pip install numpy matplotlib networkx
3. Click the double play button "Run All" to execute the code
4. The optimized routes found by both GA and A* Search algorithms will be printed along with a visualization showing the grid map and the optimized routes.

## Parameters
- `GRIDSIZE`: Size of the grid representing the map.
- `PopSIZE`: Size of the population for Genetic Algorithm.
- `MUTATIONRATE`: Probability of mutation for Genetic Algorithm.
- `GENERATIONS`: Number of generations for Genetic Algorithm.
- `obstacles`: List of obstacle locations on the grid.
- `start`: Starting point for path planning.
- `destination`: Destination point for path planning.

## Function Descriptions
- Astar(): Implements the A* search algorithm to find the shortest path from the start point to the destination point on the grid map.
- distance(point1, point2): Calculates the Manhattan distance between two points on the grid.
- IntializePop(size): Initializes the population for Genetic Algorithm with random routes.
- FitnessEvaluation(route)`: Evaluates the fitness of a route based on its distance to the destination point.
- SelectionArea(Pop, k=3): Performs tournament selection to select individuals from the population for the next generation.
- Crossover(parent1, parent2)`: Performs one-point crossover to generate offspring from two parent routes.
- Mutate(route): Applies mutation to a route with a certain probability.
- GA(): Implements the Genetic Algorithm to find the optimized route from the start point to the destination point.
- OptimizedGA = GA(): Invokes the Genetic Algorithm to find the optimized route.
- OptimizedAstar = Astar(): Invokes the A* search algorithm to find the optimized route.

## Algorithm Details
- Genetic Algorithm or GA: Generates a population of routes and evolves them over generations using selection, crossover, and mutation operations to find the optimal route.
- A* Search Algorithm: Utilizes a heuristic search technique to find the optimal path from the start point to the destination point on the grid map.

## Visualization
- The visualization plot shows the grid map with:
1. Obstacles
2. The starting point in BLUE
3. The destination point in Green
4. The optimized routes found by GA in RED and A* Search in BLUE.


