from preprocessing_md import extract_milestone_keys, count_milestones, sum_impairments

import polars as pl

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


# normalierat över en satt vektor med "unlocked" milestones/impairments
def motorscore_milestones_setvalue() -> float:
    possible_milestones_by_age = {1: 12, 2: 19, 3: 25, 4: 31, 5: 36, 6: 39, 7: 41}
    print("hej")

def motorscore_impairments_setvalue() -> float:
    print("hej")

# normaliserat över alla i den åldersklassen
def motorscore_milestones() -> float:
    print("hej")

def motorscore_impairments() -> float:
    print("hej")

# normaliserat över alla i den åldersklassen och gmfcs nivån
def motorscore_milestones_future() -> float:
    print("hej")

def motorscore_impairments_future() -> float:
    print("hej")

# kombinerat score 
def motorscore_combined(imScore: float, moScore: float) -> float:
    print("hej")
