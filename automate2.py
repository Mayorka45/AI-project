import csv
import os
import time
import pandas as pd
from fifteenpuzzle import FifteenPuzzleState, FifteenPuzzleSearchProblem
from generator import generateRandomFifteenPuzzle, create_and_save_puzzles
from search import aStarSearch
from search import depthFirstSearch, breadthFirstSearch, uniformCostSearch, h3_manhattan_distance


def process_puzzle(puzzle_data):
    """Convert and validate puzzle input into a proper 16-element list."""
    if isinstance(puzzle_data, list) and all(isinstance(tier, list) for tier in puzzle_data):
        puzzle_data = [num for tier in puzzle_data for num in tier]

    if len(puzzle_data) != 16:
        raise ValueError(f"Invalid puzzle configuration: {puzzle_data}. Must contain 16 elements.")

    return puzzle_data


def execute_strategy_comparison(puzzle_set, output_file):
    """Execute different search algorithms and record performance metrics."""
    search_plans = [
        ("DFS", depthFirstSearch),
        ("BFS", breadthFirstSearch),
        ("UCS", uniformCostSearch),
        ("A* with Manhattan Distance", h3_manhattan_distance)
    ]

    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            "PuzzleID", "Strategy", "Solved", "Solution Depth",
            "Expanded Nodes", "Max Fringe Size", "Execution Time"
        ])

        for puzzle_num, puzzle_instance in enumerate(puzzle_set, 1):
            print(f"Processing puzzle #{puzzle_num}...")

            try:
                valid_puzzle = process_puzzle(puzzle_instance)
                puzzle_obj = FifteenPuzzleState(valid_puzzle)
                problem_instance = FifteenPuzzleSearchProblem(puzzle_obj)

                method_index = 0
                for strategy_name, search_method in search_plans:
                    nodes_expanded = 0
                    max_fringe = 0

                    def monitor_fringe(fringe):
                        nonlocal max_fringe
                        max_fringe = max(max_fringe, fringe.count)

                    def count_expansion():
                        nonlocal nodes_expanded
                        nodes_expanded += 1

                    start_timer = time.perf_counter()

                    if method_index < 3:  # For DFS, BFS, UCS
                        solution, max_fringe, nodes_expanded = search_method(problem_instance)
                    else:  # For A* with Manhattan
                        solution = aStarSearch(problem_instance, search_method, monitor_fringe, count_expansion)

                    end_timer = time.perf_counter()

                    is_solved = bool(solution)
                    solution_length = len(solution) if is_solved else 0
                    processing_time = end_timer - start_timer

                    if not is_solved:
                        break

                    csv_writer.writerow([
                        puzzle_num,
                        strategy_name,
                        is_solved,
                        solution_length,
                        nodes_expanded,
                        max_fringe,
                        f"{processing_time:.4f}"
                    ])

                    method_index += 1

            except ValueError as error:
                print(f"Error with puzzle {puzzle_num}: {error}")


def assess_performance(results_file):
    """Evaluate and compare performance metrics from recorded results."""
    strategy_metrics = {
        "DFS": {"nodes": 0, "time": 0, "fringe": 0, "runs": 0},
        "BFS": {"nodes": 0, "time": 0, "fringe": 0, "runs": 0},
        "UCS": {"nodes": 0, "time": 0, "fringe": 0, "runs": 0},
        "A* with Manhattan Distance": {"nodes": 0, "time": 0, "fringe": 0, "runs": 0},
    }

    with open(results_file, 'r') as datafile:
        results_reader = csv.DictReader(datafile)

        for record in results_reader:
            current_strategy = record['Strategy']
            strategy_metrics[current_strategy]["nodes"] += int(record['Expanded Nodes'])
            strategy_metrics[current_strategy]["time"] += float(record['Execution Time'])
            strategy_metrics[current_strategy]["fringe"] += int(record['Max Fringe Size'])
            strategy_metrics[current_strategy]["runs"] += 1

    # Calculate averages
    for method in strategy_metrics:
        if strategy_metrics[method]["runs"] > 0:
            for metric in ["nodes", "time", "fringe"]:
                strategy_metrics[method][metric] /= strategy_metrics[method]["runs"]

    # Determine top performers
    node_leader = min(strategy_metrics, key=lambda x: strategy_metrics[x]["nodes"])
    time_leader = min(strategy_metrics, key=lambda x: strategy_metrics[x]["time"])
    fringe_leader = min(strategy_metrics, key=lambda x: strategy_metrics[x]["fringe"])

    print("\nAverage Nodes Expanded:")
    for method, data in strategy_metrics.items():
        print(f"{method}: {data['nodes']:.1f}")

    print(f"\nNodes Expanded Champion: {node_leader} ({strategy_metrics[node_leader]['nodes']:.1f} nodes)")

    print("\nAverage Execution Times:")
    for method, data in strategy_metrics.items():
        print(f"{method}: {data['time']:.4f}s")

    print(f"\nSpeed Champion: {time_leader} ({strategy_metrics[time_leader]['time']:.4f}s)")

    print("\nAverage Maximum Fringe Sizes:")
    for method, data in strategy_metrics.items():
        print(f"{method}: {data['fringe']:.1f}")

    print(f"\nFringe Size Champion: {fringe_leader} ({strategy_metrics[fringe_leader]['fringe']:.1f} units)")


if __name__ == "__main__":
    puzzle_source = "scenarios.csv"
    results_output = "results_task4.csv"

    if not os.path.exists(puzzle_source):
        print("Generating new puzzle scenarios...")
        create_and_save_puzzles(puzzle_source, num_puzzles=100, moves=50)

    puzzle_data = pd.read_csv(puzzle_source)
    puzzle_collection = puzzle_data['State'].apply(eval).tolist()

    execute_strategy_comparison(puzzle_collection, results_output)
    assess_performance(results_output)