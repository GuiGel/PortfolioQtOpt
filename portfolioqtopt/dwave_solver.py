from collections import Counter
from enum import Enum, unique

from dimod.sampleset import SampleSet
from dwave.system import LeapHybridSampler

from portfolioqtopt.interpreter import get_selected_funds_indexes
from portfolioqtopt.markovitz_portfolio import Selection
from portfolioqtopt.qubo import Qubo


@unique
class SolverTypes(Enum):
    Clique_Embedding = "Clique_Embedding"
    Find_Embedding = "Find_Embedding"
    hybrid_solver = "hybrid_solver"
    SA = "SA"
    exact = "exact"


def solve_dwave_advantage_cubo(
    qubo: Qubo, solver: SolverTypes, api_token: str
) -> SampleSet:

    # api_token = 'DEV-d9751cb50bc095c993f55b3255f728d5b2793c36'
    _URL = "https://na-west-1.cloud.dwavesys.com/sapi/v2/"

    if solver.value == "hybrid_solver":
        sampler = LeapHybridSampler(token=api_token, endpoint=_URL)
        sampleset: SampleSet = sampler.sample_qubo(qubo.dictionary)
        return sampleset
    else:
        raise ValueError(f"Bad solver. Solver must be hybrid_solver.")


def dimension_reduction(
    selection: Selection,
    runs: int,
    w: int,
    theta1: float,
    theta2: float,
    theta3: float,
    token: str,
    solver: SolverTypes,
) -> Counter[int]:
    c: Counter[int] = Counter()
    for i in range(runs):
        qbits = selection.solve(theta1, theta2, theta3, token, solver)
        indexes = get_selected_funds_indexes(qbits, w)
        if not i:
            c = Counter(indexes)
        else:
            c.update(Counter(indexes))
    return c
