import csv
import os
import time
import pandas as pd
from eightpuzzle import EightPuzzleState, EightPuzzleSearchProblem
from generator import generateRandomFifteenPuzzle, create_and_save_puzzles
from search import aStarSearch
from search import depthFirstSearch, breadthFirstSearch, uniformCostSearch, h3_manhattan_distance

def validate_and_prepare_puzzle(puzzle_state):
    """
    Ensures that the puzzle state is a list of 16 integers.
    Converts it into a valid format for FifteenPuzzleState if necessary.
    """
    # Flatten the list if it is 2D and ensure it's a 16-element list
    if isinstance(puzzle_state, list) and all(isinstance(row, list) for row in puzzle_state):
        # Flatten the 2D array
        puzzle_state = [tile for row in puzzle_state for tile in row]

    # Check if the puzzle state contains exactly 16 elements
    if len(puzzle_state) != 16:
        raise ValueError(f"Invalid puzzle state: {puzzle_state}. It must contain exactly 16 numbers.")

    return puzzle_state

def run_strategic_comparison(puzzles, results_file):
    """
    Run comparisons for A* using different heuristics and write results to a CSV file.
    """
    Strategies = [
        ("DFS", depthFirstSearch),
        ("BFS", breadthFirstSearch),
        ("UCS", uniformCostSearch),
        ("A* with Manhattan Distance", h3_manhattan_distance)
    ]

    # Open the CSV file to write the results
    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["PuzzleID", "Strategy", "Solved", "Solution Depth", "Expanded Nodes", "Max Fringe Size", "Execution Time"]
        )

        # Iterate through the list of puzzles and run the comparisons
        for idx, puzzle_state in enumerate(puzzles, 1):
            print(f"Running comparisons for puzzle {idx}...")

            try:
                # Validate and prepare the puzzle state
                valid_puzzle_state = validate_and_prepare_puzzle(puzzle_state)
                puzzle = EightPuzzleState(valid_puzzle_state)

                # Create the search problem
                problem = EightPuzzleSearchProblem(puzzle)


                i=0
                name="UFS"
                best_heuristic=[0,0,0,0]
                # Track expanded nodes and fringe size locally
                for name, strategy in Strategies:
                    expanded_nodes = 0  # Reset expanded node count
                    max_fringe_size = 0  # Reset max fringe size

                    def track_fringe(fringe):
                        """Utility to update the max fringe size."""
                        nonlocal max_fringe_size
                        max_fringe_size = max(max_fringe_size, fringe.count)

                    def track_expansion():
                        """Utility to increment the expanded node counter."""
                        nonlocal expanded_nodes
                        expanded_nodes += 1

                    start_time = time.perf_counter()

                    if(i%4!=3):
                        solution, max_fringe_size, expanded_nodes = strategy(problem)
                    else:
                        solution = aStarSearch(problem, strategy, track_fringe, track_expansion)

                    end_time = time.perf_counter()


                    solved = bool(solution)
                    depth = len(solution) if solved else "N/A"
                    execution_time = end_time - start_time
                    if not solved:
                        break

                    i=i+1
                    writer.writerow([idx, name, solved, depth, expanded_nodes, max_fringe_size, execution_time])

            except ValueError as e:
                print(f"Error processing puzzle {idx}: {e}")

def analyze_result(results_file):
        """
        Analyze the results from the CSV file to find the heuristic that expanded the least nodes,
        had the least execution time, and the least max fringe size.
        """
        Strategies_scores = {
            "DFS": {"expanded_nodes": 0, "execution_time": 0, "max_fringe": 0, "count": 0},
            "BFS": {"expanded_nodes": 0, "execution_time": 0, "max_fringe": 0, "count": 0},
            "UCS": {"expanded_nodes": 0, "execution_time": 0, "max_fringe": 0, "count": 0},
            "A* with Manhattan Distance": {"expanded_nodes": 0, "execution_time": 0, "max_fringe": 0, "count": 0},
        }

        # Read the CSV results file
        with open(results_file, mode='r') as file:
            reader = csv.DictReader(file)

            # Analyze each row
            for row in reader:
                strategy = row['Strategy']
                expanded_nodes = int(row['Expanded Nodes'])
                execution_time = float(row['Execution Time'])
                max_fringe_size = int(row['Max Fringe Size'])

                # Accumulate scores
                Strategies_scores[strategy]["expanded_nodes"] += expanded_nodes
                Strategies_scores[strategy]["execution_time"] += execution_time
                Strategies_scores[strategy]["max_fringe"] += max_fringe_size
                Strategies_scores[strategy]["count"] += 1

        # Calculate average scores
        for strategy in Strategies_scores:
            if  Strategies_scores[strategy]["count"] > 0:
                Strategies_scores[strategy]["expanded_nodes"] /= Strategies_scores[strategy]["count"]
                Strategies_scores[strategy]["execution_time"] /= Strategies_scores[strategy]["count"]
                Strategies_scores[strategy]["max_fringe"] /= Strategies_scores[strategy]["count"]

        # Determine winners for each metric
        expanded_winner = min(Strategies_scores, key=lambda h: Strategies_scores[h]["expanded_nodes"])
        time_winner = min(Strategies_scores, key=lambda h: Strategies_scores[h]["execution_time"])
        fringe_winner = min(Strategies_scores, key=lambda h: Strategies_scores[h]["max_fringe"])

        # Print final scores and winners
        print("\nFinal Scores (Total Expanded Nodes):")
        for heuristic, metrics in Strategies_scores.items():
            print(f"{heuristic}: {metrics['expanded_nodes']}")

        print(
            f"\nWinner for Least Expanded Nodes: {expanded_winner} with {Strategies_scores[expanded_winner]['expanded_nodes']} total expanded nodes")

        print("\nFinal Scores (Average Execution Time):")
        for heuristic, metrics in Strategies_scores.items():
            print(f"{heuristic}: {metrics['execution_time']:.4f} seconds")

        print(
            f"\nWinner for Least Execution Time: {time_winner} with {Strategies_scores[time_winner]['execution_time']:.4f} seconds")

        print("\nFinal Scores (Average Max Fringe Size):")
        for heuristic, metrics in Strategies_scores.items():
            print(f"{heuristic}: {metrics['max_fringe']}")

        print(
            f"\nWinner for Least Max Fringe Size: {fringe_winner} with {Strategies_scores[fringe_winner]['max_fringe']} size")


if __name__ == "__main__":
    # Example usage, reading scenarios from a CSV
    scenarios_file = "scenarios.csv"
    results_file = "results_task4.csv"

    # If the scenarios file does not exist, we generate it.
    if not os.path.exists(scenarios_file):
        print(f"{scenarios_file} not found. Generating random puzzles...")
        create_and_save_puzzles(scenarios_file, num_puzzles=100, moves=25)

    # Read the puzzles from CSV (assuming they're formatted as arrays)
    puzzles_df = pd.read_csv(scenarios_file)
    puzzles = puzzles_df['State'].apply(eval).tolist()

    # Run the comparison
    run_strategic_comparison(puzzles, results_file)
    analyze_result(results_file)
