from enum import Enum, unique

from dimod.sampleset import SampleSet
from dwave.system import LeapHybridSampler

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
