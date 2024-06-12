from datasets import load_dataset
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from ortools.graph.python import linear_sum_assignment
from ortools.algorithms.python import knapsack_solver
from ortools.linear_solver import pywraplp
import collections
from ortools.sat.python import cp_model
import numpy as np
from scipy.spatial import distance_matrix
import re

##### HUGGINGFACE TASKS #####

def get_huggingface_dataset(task_name):
    dataset = load_dataset(task_name, "main", split="test")
    tasks, answers = [], []
    for data_index in range(len(dataset)):
        inp = dataset[data_index]['question']
        out = dataset[data_index]['answer']
        tasks.append(inp)
        answers.append(out)
    return tasks, answers

def numeric_answer_extractor(text):
    answer = re.findall(r'\d+', text)[-1]
    answer = int(answer)
    return answer

##### GOOGLE-OR TASKS #####

def get_gap(cost, optimal_cost):
    return max(0, ((cost - optimal_cost)/optimal_cost) * 100)

class Assignment():
    def __init__(self, max_interval, num_workers):
        self.max_interval = max_interval
        self.num_workers = num_workers

    def get_variables(self):
        self.costs = np.random.randint(low=0,high=self.max_interval, size=(self.num_workers,self.num_workers))

    def solve(self):
        assignment = linear_sum_assignment.SimpleLinearSumAssignment()
        end_nodes_unraveled, start_nodes_unraveled = np.meshgrid(
            np.arange(self.costs.shape[1]), np.arange(self.costs.shape[0])
        )
        start_nodes = start_nodes_unraveled.ravel()
        end_nodes = end_nodes_unraveled.ravel()
        arc_costs = self.costs.ravel()

        assignment.add_arcs_with_cost(start_nodes, end_nodes, arc_costs)

        solution = []
        cost = 0
        for i in range(0, assignment.num_nodes()):
            solution.append(assignment.right_mate(i))
            cost += assignment.assignment_cost(i)

        return solution, cost

    def compute_cost(self, solution):
        cost = 0
        for i in range(len(solution)):
            cost += self.costs[solution[i]-1]
        return cost
    
    def create_prompt(self):
        inp1 = """You are given a matrix, where each row represents a worker and each column represents a task: """
        inp2 = str(self.costs)
        inp3 = """ Find the assignment of each task to each worker where all tasks are completed and the assignment has the minimum total total cost. Return the asnwer as a Python list of tasks where indces of this list stand for corresponding workers. """
        prompt = inp1 + inp2 + inp3

        return prompt
    
class Knapsack():
    def __init__(self, max_interval, num_items, capacities):
        self.max_interval = max_interval
        self.num_items = num_items
        self.capacities = capacities

    def get_variables(self):
        self.values = np.random.randint(low=1,high=self.max_interval, size=(self.num_items))
        self.weights = np.random.randint(low=0,high=self.max_interval, size=(self.num_items))

    def solve(self):
        solver = knapsack_solver.KnapsackSolver(
            knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
            "KnapsackExample",
        )

        solver.init(self.values, self.weights, self.capacities)
        computed_value = solver.solve()

        solution = []
        cost = 0

        for i in range(len(self.values)):
            if solver.best_solution_contains(i):
                solution.append(i)
                cost += self.weights[0][i]

        return solution, cost

    def compute_cost(self, solution):
        cost = 0
        for i in range(len(solution)):
            cost += self.values[solution[i]-1]
        return cost
    
    def create_prompt(self):
        inp1 = """You are given a first list of item values and a second list of item weights: """
        inp2 = str(self.values) + str(self.weights)
        inp3 = """ Find the a set of items to pack into a container with a maximum weight capacity = """ + str(self.capacities[0])
        inp4 = """ that maximizes total value of packed items. Return the answer as a Python list of item indices. """
        prompt = inp1 + inp2 + inp3 + inp4

        return prompt

class BinPacking():
    def __init__(self, max_interval, num_items, bin_capacity):
        self.max_interval = max_interval
        self.num_items = num_items
        self.bin_capacity = bin_capacity

    def get_variables(self):
        self.weights = np.random.randint(low=0,high=self.max_interval, size=(self.num_items))
        self.data = {}
        self.data["weights"] = self.weights
        self.data["items"] = list(range(len(self.weights)))
        self.data["bins"] = self.data["items"]
        self.data["bin_capacity"] = self.bin_capacity

    def solve(self):
        solver = pywraplp.Solver.CreateSolver("SCIP")
        # Variables
        # x[i, j] = 1 if item i is packed in bin j.
        x = {}
        for i in self.data["items"]:
            for j in self.data["bins"]:
                x[(i, j)] = solver.IntVar(0, 1, "x_%i_%i" % (i, j))

        # y[j] = 1 if bin j is used.
        y = {}
        for j in self.data["bins"]:
            y[j] = solver.IntVar(0, 1, "y[%i]" % j)

        # Constraints
        # Each item must be in exactly one bin.
        for i in self.data["items"]:
            solver.Add(sum(x[i, j] for j in self.data["bins"]) == 1)

        # The amount packed in each bin cannot exceed its capacity.
        for j in self.data["bins"]:
            solver.Add(
                sum(x[(i, j)] * self.data["weights"][i] for i in self.data["items"])
                <= y[j] * self.data["bin_capacity"]
            )

        # Objective: minimize the number of bins used.
        solver.Minimize(solver.Sum([y[j] for j in self.data["bins"]]))
        solution = []
        cost = 0
        for j in self.data["bins"]:
            if y[j].solution_value() == 1:
                bin_items = []
                bin_weight = 0
                for i in self.data["items"]:
                    if x[i, j].solution_value() > 0:
                        bin_items.append(i)
                        bin_weight += self.data["weights"][i]
                if bin_items:
                    cost += 1
                    solution.append(bin_items)

        return solution, cost

    def compute_cost(self, solution):
        cost = len(solution)
        return cost
    
    def create_prompt(self):
        inp1 = """You are given a list of item weights: """
        inp2 = str(self.weights)
        inp3 = """ Find minimum number of bins with a maximum weight capacity = """ + str(self.bin_capacity)
        inp4 = """ that will hold all items given. Return as a Python list of lists, where each row is bin and each column is a list of item indices. """
        prompt = inp1 + inp2 + inp3 + inp4

        return prompt
    
class VRP():
    def __init__(self, max_interval, num_cities, num_vehicles, capacity):
        self.max_interval = max_interval
        self.num_cities = num_cities
        self.num_vehicles = num_vehicles
        self.capacity = capacity

    def get_variables(self):
        """ Throws n nodes uniformly at random on a square, and build a (fully connected) graph.
            Returns the (N, 2) coordinates matrix, and the (N, N) matrix containing pairwise euclidean distances.
        """
        coords = np.random.randint(low=0,high=self.max_interval, size=(self.num_cities,2))
        demands = np.random.randint(low=0,high=self.max_interval, size=(1))
        dist_mat = distance_matrix(coords, coords)
        dist_mat = [list(dist_mat[i].astype(int)) for i in range(len(dist_mat))]
        
        self.variables = {
                        'coords' : coords, 
                        'dist_mat' : dist_mat,
                        'demands' : demands, 
                        'capacity' : self.capacity
                        }

    def create_data_model(self, dist_mat, num_vehicles):
        """Stores the data for the problem."""
        data = {}
        data["distance_matrix"] = dist_mat
        data["num_vehicles"] = num_vehicles
        data["depot"] = 0
        return data

    def compute_cost(self, solution):
        cost = 0
        for i in range(1, len(solution)):
            cost += self.variables['dist_mat'][solution[i-1]][solution[i]]
        return cost

    def return_solution(self, data, manager, routing, solution):

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

    def solve(self):
        """Entry point of the program."""
        dist_mat = self.variables['dist_mat']
        num_vehicles = self.num_vehicles
        # Instantiate the data problem.
        data = self.create_data_model(dist_mat, num_vehicles)

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
            return self.return_solution(data, manager, routing, solution)
        else:
            return None, None
        
    def create_prompt(self):
        coords = self.variables['coords']
        demands = self.variables['demands']
        capacity = self.variables['capacity']
        inp_vrp1 = """You are given a list of customers with coordinates: """
        inp_vrp2 = ""
        for j in range(1, len(coords)):
            inp_vrp2 += '('
            inp_vrp2 += str(j)
            inp_vrp2 += "): (" 
            inp_vrp2 += str(coords[j][0])
            inp_vrp2 += ", "
            inp_vrp2 += str(coords[j][1])
            inp_vrp2 += "); "
        inp_vrp3 = """and a list of customer demands: """
        inp_vrp4 = ""
        for j in range(1, len(coords)):
            inp_vrp4 += '('
            inp_vrp4 += str(j)
            inp_vrp4 += "): " 
            inp_vrp4 += str(demands[j])
            inp_vrp4 += "; "
        inp_vrp4 = inp_vrp4[:-1] + ". "
        inp_vrp5 = "There is a depot (Customer 0) with coordinates (" + str(coords[0][0]) + ", " + str(coords[0][1]) + ") "
        inp_vrp6 = "and a vehicle with a maximum capacity of " + str(capacity) + ". "
        inp_vrp7 = "The goal is to find the route that has the minimum total length and go through all the customers, starting and ending at the depot."
        inp_vrp = inp_vrp1 + inp_vrp2 + inp_vrp3 + inp_vrp4 + inp_vrp5 + inp_vrp6 + inp_vrp7

        return inp_vrp
    
class JSP():
    def __init__(self, max_interval, num_workers):
        self.max_interval = max_interval
        self.num_workers = num_workers

    def get_variables(self):
        self.jobs_data = np.random.randint(low=0,high=self.max_interval, size=(self.num_jobs,self.num_operations,2))

    def solve(self):
        machines_count = 1 + max(task[0] for job in self.jobs_data for task in job)
        all_machines = range(machines_count)
        # Computes horizon dynamically as the sum of all durations.
        horizon = sum(task[1] for job in self.jobs_data for task in job)

        # Create the model.
        model = cp_model.CpModel()

        # Named tuple to store information about created variables.
        task_type = collections.namedtuple("task_type", "start end interval")
        # Named tuple to manipulate solution information.
        assigned_task_type = collections.namedtuple(
            "assigned_task_type", "start job index duration"
        )

        # Creates job intervals and add to the corresponding machine lists.
        all_tasks = {}
        machine_to_intervals = collections.defaultdict(list)

        for job_id, job in enumerate(self.jobs_data):
            for task_id, task in enumerate(job):
                machine, duration = task
                suffix = f"_{job_id}_{task_id}"
                start_var = model.new_int_var(0, horizon, "start" + suffix)
                end_var = model.new_int_var(0, horizon, "end" + suffix)
                interval_var = model.new_interval_var(
                    start_var, duration, end_var, "interval" + suffix
                )
                all_tasks[job_id, task_id] = task_type(
                    start=start_var, end=end_var, interval=interval_var
                )
                machine_to_intervals[machine].append(interval_var)

        # Create and add disjunctive constraints.
        for machine in all_machines:
            model.add_no_overlap(machine_to_intervals[machine])

        # Precedences inside a job.
        for job_id, job in enumerate(self.jobs_data):
            for task_id in range(len(job) - 1):
                model.add(
                    all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].end
                )

        # Makespan objective.
        obj_var = model.new_int_var(0, horizon, "makespan")
        model.add_max_equality(
            obj_var,
            [all_tasks[job_id, len(job) - 1].end for job_id, job in enumerate(self.jobs_data)],
        )
        model.minimize(obj_var)

        # Creates the solver and solve.
        solver = cp_model.CpSolver()
        status = solver.solve(model)

        solution = []
        assigned_jobs = collections.defaultdict(list)
        for job_id, job in enumerate(self.jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                assigned_jobs[machine].append(
                    assigned_task_type(
                        start=solver.value(all_tasks[job_id, task_id].start),
                        job=job_id,
                        index=task_id,
                        duration=task[1],
                    )
                )

        for machine in all_machines:
            # Sort by starting time.
            assigned_jobs[machine].sort()

            for assigned_task in assigned_jobs[machine]:
                solution.append(assigned_task.job)

        cost = solver.objective_value

        return solution, cost
    
    def compute_cost(self, solution):
        cost = 0
        operations = {i : 0 for i in range(len(solution))}
        for i in range(len(solution)):
            last_operation = operations[solution[i]-1]
            cost += self.jobs_data[solution[i]-1][last_operation][1]
            operations[solution[i]-1] += 1
        return cost

    def create_prompt(self):
        inp1 = """You are given a Python array, where first dimension represents jobs, second dimension represents operations, in third dimension there are two numbers, first number is machine id, second number is completion time: """
        inp2 = str(self.jobs_data)
        inp3 = """ Operations in each job can be completed in strict order only. Find the the sequence of operations that completes all jobs and minimizes total completion time. Return final answer as Python list of job indices. """
        prompt = inp1 + inp2 + inp3

        return prompt
    