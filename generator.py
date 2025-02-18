# =====Start Task 3 & 4 Changes=====
import os
import csv
import random
from fifteenpuzzle import FifteenPuzzleState

def generateRandomFifteenPuzzle(moves=25):

    #from the goal state shuffles the puzzle to generate random puzzles problems to solve

    puzzle = FifteenPuzzleState([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0])  # Solved configuration
    for i in range(moves):
        # Apply a random legal move
        puzzle = puzzle.result(random.choice(puzzle.legalMoves()))
    return puzzle

def create_and_save_puzzles(scenarios_file, num_puzzles, moves=25):
    """
    Creates random 15-puzzles and writes them to a CSV file.
    """
    puzzles = [generateRandomFifteenPuzzle(moves).cells for _ in range(num_puzzles)]

    # Write puzzles to CSV
    with open(scenarios_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['PuzzleID', 'State'])  # Column headers
        for idx, puzzle in enumerate(puzzles):
            writer.writerow([idx + 1, str(puzzle)])  # Write puzzle as string
    print(f"{num_puzzles} puzzles were successfully generated and stored in {scenarios_file}")

# =====End Task 3 & 4 Changes=====
