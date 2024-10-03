import openai
import sys
import json
import numpy as np
import tempfile
import os
import re 

def extract_code(code_text): 
    # Extract code from code block
    pattern = r'```(?:python)?\s*\n(.*?)```'
    code_blocks = re.findall(pattern, code_text, re.DOTALL)
    if code_blocks: 
        return code_blocks[0].strip() 
    else: 
        return None

def call_gpt(prompt, temperature=0):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{'role': 'user', 'content': prompt}],
            n=1,
            stop=None,
            temperature=temperature,
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

def SGE(problem, cities):
    # Step 1: Get a list of heuristics/methods that can solve the problem
    prompt1 = """You are an expert in optimization algorithms. List heuristics or methods that can solve the {problem}. Provide the methods in a JSON array format. Ensure the response is valid JSON, and nothing else. For example:
["Method1", "Method2", "Method3"]
"""

    methods_text = call_gpt(prompt1, temperature=0)
    if methods_text is None:
        print("Failed to get methods from GPT.")
        return None, None

    # Parse methods_text into a list of methods
    try:
        methods = json.loads(methods_text)
        
        assert isinstance(methods, list) and all(isinstance(m, str) for m in methods)
    except Exception as e:
        print(f"Error parsing methods: {e}")
        methods = ["Genetic Algorithm", "Simulated Annealing", "Ant Colony Optimization"]
    methods = methods[:3]
    print("Methods:", methods)
    # Step 2: For each method, get steps required to implement it
    solutions = []
    for method in methods:
        print(f"Processing method: {method}")
        prompt2 = f"""Provide detailed steps to implement the {method} method for solving {problem}. List all the steps in a JSON array of strings. Ensure the response is valid JSON, and nothing else. For example:
["Step1", "Step2", "Step3"]
"""

        steps_text = call_gpt(prompt2, temperature=0)
        if steps_text is None:
            print(f"Failed to get steps for method {method} from GPT.")
            continue

        # Parse steps_text into a list of steps
        try:
            steps = json.loads(steps_text)            
            assert isinstance(steps, list) and all(isinstance(s, str) for s in steps)
            steps_str = '\n'.join(steps)
        except Exception as e:
            print(f"Error parsing steps for method {method}: {e}")
            continue

        print("Steps:", steps)
        # Step 3: Use GPT to implement the steps of each method
        prompt3 = f"""Implement in Python the following steps to solve TSP using the {method} heuristic:
            {steps_str}
            The code should define a function `tsp_solver(cities)` that takes as input a list of city positions (as tuples of coordinates) and outputs a tuple `(tour, distance)`, where `tour` is a list of city indices representing the order of the tour, and `distance` is the total length of the tour.
            Provide only the code inside a Python code block like this:
            ```python
            # Your code here 
            Do not include any explanations or extra text. Only provide the code in the code block. 
            tsp_solver() must enforce a strict time limit of 10 seconds for its execution. Ensure that the code is syntactically correct and uses only standard Python libraries. """
        
        code_text = call_gpt(prompt3, temperature=0)
        print('Code:', code_text)
        if code_text is None:
            print(f"Failed to get code for method {method} from GPT.")
            continue

        # Extract code from code block
        code = extract_code(code_text)
        if not code:
            print(f"Could not extract code for method {method}")
            continue

        # Save code to a temporary file and import it
        with tempfile.NamedTemporaryFile(delete=False, suffix='.py') as temp_module:
            temp_module_name = os.path.basename(temp_module.name).split('.')[0]
            temp_module.write(code.encode('utf-8'))
            temp_module_path = temp_module.name

        try:
            # Import the module
            import importlib.util
            spec = importlib.util.spec_from_file_location(temp_module_name, temp_module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # Check if 'tsp_solver' function exists
            if hasattr(module, 'tsp_solver'):
                tour, distance = module.tsp_solver(cities)
                if tour and distance:
                    solutions.append({'method': method, 'tour': tour, 'distance': distance})
                else:
                    print(f"tsp_solver did not return a valid solution for method {method}")
            else:
                print(f"No tsp_solver function found in the code for method {method}")
            print('Solution:', len(tour), distance)
        except Exception as e:
            print(f"Error executing code for method {method}: {e}")
        finally:
            # Clean up the temporary file
            os.remove(temp_module_path)

    # Step 4: Combine solutions from each method into final solution
    if solutions:
        best_solution = min(solutions, key=lambda x: x['distance'])
        return best_solution['tour'], best_solution['distance']
    else:
        print("No valid solutions were found.")
        return None, None
    
def main():
    # Generate the TSP instance inside the script
    num_cities = 200
    cities = list(map(tuple, np.random.rand(num_cities, 2) * 100))
    # Get OpenAI API key from environment variable or prompt the user
    openai_api_key = os.environ['OPENAI_API_KEY']
    openai.api_key = openai_api_key

    # Solve TSP using GPT
    problem = "Travelling Salesman Problem"
    tour, distance = SGE(problem, cities)
    if tour is not None:
        print("Best tour found:")
        print("Tour:", tour)
        print("Total distance:", distance)
    else:
        print("Failed to find a valid tour.")
        
if __name__ == "__main__": 
    np.random.seed(42)
    main()

