from dataclasses import dataclass

from portfolioqtopt.optimization._qbits import *
from portfolioqtopt.optimization._qubo import QuboData
from portfolioqtopt.optimization.utils import Array


@dataclass
class Interpretation:
    investment: Array
    expected_returns: Array
    risk: float
    sharpe_ratio: float
    selected_indexes: Indexes


def get_interpretation(qubo: QuboData, qbits: Qbits) -> Interpretation:
    investments = get_investments(qbits, qubo.w)
    investments_nonzero = get_investments_nonzero(investments)
    expected_returns = get_returns(qbits, qubo.arp)
    selected_indexes = get_selected_funds_indexes(qbits, qubo.w)
    risk = get_risk(investments, qubo.prices)
    sharpe_ratio = get_sharpe_ratio(qbits, qubo.arp, qubo.prices, qubo.w)
    return Interpretation(
        investment=investments_nonzero,
        expected_returns=100.0 * expected_returns.sum(),
        risk=risk,
        sharpe_ratio=sharpe_ratio,
        selected_indexes=selected_indexes,
    )
