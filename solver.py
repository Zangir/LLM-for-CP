from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
from scipy.spatial import distance_matrix

def get_graph_mat(n=10, size=1):
    """ Throws n nodes uniformly at random on a square, and build a (fully connected) graph.
        Returns the (N, 2) coordinates matrix, and the (N, N) matrix containing pairwise euclidean distances.
    """
    coords = size * np.random.randint(low = 0,high=n*10, size=(n,2))
    dist_mat = distance_matrix(coords, coords)
    dist_mat = [list(dist_mat[i].astype(int)) for i in range(len(dist_mat))]
    
    return coords, dist_mat

def create_data_model(dist_mat, num_vehicles):
    """Stores the data for the problem."""
    data = {}
    data["distance_matrix"] = dist_mat
    data["num_vehicles"] = num_vehicles
    data["depot"] = 0
    return data

def compute_cost_vrp(solution, dist_mat):
    cost = 0
    for i in range(1, len(solution)):
        cost += dist_mat[solution[i-1]][solution[i]]
    return cost

def cost_change(cost_mat, n1, n2, n3, n4):
    return cost_mat[n1][n3] + cost_mat[n2][n4] - cost_mat[n1][n2] - cost_mat[n3][n4]

def two_opt(route, cost_mat):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue
                if cost_change(cost_mat, best[i - 1], best[i], best[j - 1], best[j]) < 0:
                    best[i:j] = best[j - 1:i - 1:-1]
                    improved = True
        route = best
    return best

def return_solution(data, manager, routing, solution):

    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        route = []
        route_distance = 0
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        route.append(manager.IndexToNode(index))

    return route, route_distance

def solve(dist_mat, num_vehicles):
    """Entry point of the program."""
    # Instantiate the data problem.
    data = create_data_model(dist_mat, num_vehicles)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        return return_solution(data, manager, routing, solution)
    else:
        return None, None
    
def get_gap(cost, optimal_cost):
    return max(0, ((cost - optimal_cost)/optimal_cost) * 100)