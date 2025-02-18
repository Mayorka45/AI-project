import csv
import os
import time
import pandas as pd
import ast  # Import the ast module for safe evaluation
from generator import create_and_save_puzzles
from fifteenpuzzle import FifteenPuzzleState,FifteenPuzzleSearchProblem
from search import aStarSearch
from search import h1_misplaced_tiles, h2_euclidean_distance, h3_manhattan_distance, h4_tiles_out

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

def run_heuristic_comparison(puzzles, results_file):
    """
    Run comparisons for A* using different heuristics and write results to a CSV file.
    """
    heuristics = [
        ("A* with Misplaced Tiles", h1_misplaced_tiles),
        ("A* with Euclidean Distance", h2_euclidean_distance),
        ("A* with Manhattan Distance", h3_manhattan_distance),
        ("A* with Row-Column Misplacements", h4_tiles_out)
    ]

    # Open the CSV file to write the results
    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["PuzzleID", "Heuristic", "Solved", "Solution Depth", "Expanded Nodes", "Max Fringe Size", "Execution Time"]
        )

        # Iterate through the list of puzzles and run the comparisons
        for idx, puzzle_state in enumerate(puzzles, 1):
            print(f"Running comparisons for puzzle {idx}...")

            try:
                # Validate and prepare the puzzle state
                valid_puzzle_state = validate_and_prepare_puzzle(puzzle_state)
                puzzle = FifteenPuzzleState(valid_puzzle_state)

                # Create the search problem
                problem = FifteenPuzzleSearchProblem(puzzle)

                # Track expanded nodes and fringe size locally
                for name, heuristic in heuristics:
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

                    start_time = time.time()
                    solution = aStarSearch(problem, heuristic, track_fringe, track_expansion)
                    end_time = time.time()

                    solved = bool(solution)
                    depth = len(solution) if solved else "N/A"
                    execution_time = end_time - start_time
                    writer.writerow([idx, name, solved, depth, expanded_nodes, max_fringe_size, execution_time])

            except ValueError as e:
                print(f"Error processing puzzle {idx}: {e}")


def analyze_results(results_file):
    """
    Analyze the results from the CSV file to find the heuristic that expanded the least nodes,
    had the least execution time, and the least max fringe size.
    """
    heuristics_scores = {
        "A* with Misplaced Tiles": {"expanded_nodes": 0, "execution_time": 0, "max_fringe": 0, "count": 0},
        "A* with Euclidean Distance": {"expanded_nodes": 0, "execution_time": 0, "max_fringe": 0, "count": 0},
        "A* with Manhattan Distance": {"expanded_nodes": 0, "execution_time": 0, "max_fringe": 0, "count": 0},
        "A* with Row-Column Misplacements": {"expanded_nodes": 0, "execution_time": 0, "max_fringe": 0, "count": 0},
    }

    # Read the CSV results file
    with open(results_file, mode='r') as file:
        reader = csv.DictReader(file)

        # Analyze each row
        for row in reader:
            heuristic = row['Heuristic']
            expanded_nodes = int(row['Expanded Nodes'])
            execution_time = float(row['Execution Time'])
            max_fringe_size = int(row['Max Fringe Size'])

            # Accumulate scores
            heuristics_scores[heuristic]["expanded_nodes"] += expanded_nodes
            heuristics_scores[heuristic]["execution_time"] += execution_time
            heuristics_scores[heuristic]["max_fringe"] += max_fringe_size
            heuristics_scores[heuristic]["count"] += 1

    # Calculate average scores
    for heuristic in heuristics_scores:
        if heuristics_scores[heuristic]["count"] > 0:
            heuristics_scores[heuristic]["expanded_nodes"] /= heuristics_scores[heuristic]["count"]
            heuristics_scores[heuristic]["execution_time"] /= heuristics_scores[heuristic]["count"]
            heuristics_scores[heuristic]["max_fringe"] /= heuristics_scores[heuristic]["count"]

    # Determine winners for each metric
    expanded_winner = min(heuristics_scores, key=lambda h: heuristics_scores[h]["expanded_nodes"])
    time_winner = min(heuristics_scores, key=lambda h: heuristics_scores[h]["execution_time"])
    fringe_winner = min(heuristics_scores, key=lambda h: heuristics_scores[h]["max_fringe"])

    # Print final scores and winners
    print("\nFinal Scores (Total Expanded Nodes):")
    for heuristic, metrics in heuristics_scores.items():
        print(f"{heuristic}: {metrics['expanded_nodes']}")

    print(
        f"\nWinner for Least Expanded Nodes: {expanded_winner} with {heuristics_scores[expanded_winner]['expanded_nodes']} total expanded nodes")

    print("\nFinal Scores (Average Execution Time):")
    for heuristic, metrics in heuristics_scores.items():
        print(f"{heuristic}: {metrics['execution_time']:.4f} seconds")

    print(
        f"\nWinner for Least Execution Time: {time_winner} with {heuristics_scores[time_winner]['execution_time']:.4f} seconds")

    print("\nFinal Scores (Average Max Fringe Size):")
    for heuristic, metrics in heuristics_scores.items():
        print(f"{heuristic}: {metrics['max_fringe']}")

    print(
        f"\nWinner for Least Max Fringe Size: {fringe_winner} with {heuristics_scores[fringe_winner]['max_fringe']} size")


# Include the call to analyze_results at the end of your script
if __name__ == "__main__":
    scenarios_file = "scenarios.csv"
    results_file = "results_task3.csv"

    # If the scenarios file does not exist, we generate it.
    if not os.path.exists(scenarios_file):
        print(f"{scenarios_file} not found. Generating random puzzles...")
        create_and_save_puzzles(scenarios_file,100, 50)

    puzzles_df = pd.read_csv(scenarios_file)
    puzzles = puzzles_df['State'].apply(ast.literal_eval).tolist()  # Use ast.literal_eval for safe evaluation

    run_heuristic_comparison(puzzles, results_file)

    # Analyze the results after running the comparisons
    analyze_results(results_file)