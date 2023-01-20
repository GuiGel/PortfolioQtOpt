from dataclasses import dataclass

from portfolioqtopt.optimization.utils import *


@dataclass
class QuboData:
    q: Q  # Coefficient of a QUBO problem
    prices: Array
    m: int
    w: int
    npp: Array
    arp: Array  # Anual returns partition


def get_qubo(
    prices: Array, b: float, w: int, theta1: float, theta2: float, theta3: float
) -> QuboData:
    _, m = prices.shape

    # Compute the granularity partition
    pw = get_partitions_granularity(w)

    # Compute the  partition of the normalized prices
    np = get_normalized_prices(prices, b)
    npp = get_normalized_prices_partition(np, pw)

    # Compute the partition of the anual returns
    anual_returns = get_anual_returns(prices)
    anual_returns_partitions = get_anual_returns_partition(anual_returns, pw)

    # Compute the partitions of the average daily returns.
    # adr = get_average_daily_returns(prices)
    # adrp = get_average_daily_returns_partition(adr, pw)


    adrp = get_average_daily_returns_partition_tecnalia(np, pw)

    # Set qubo values
    qubo_covariance = get_qubo_covariance(npp)
    qubo_returns = get_qubo_returns(adrp)
    partitions_granularity_broadcast = get_partitions_granularity_broadcast(pw, m)
    qubo_linear = get_qubo_prices_linear(partitions_granularity_broadcast, b)
    qubo_quadratic = get_qubo_prices_quadratic(partitions_granularity_broadcast, b)

    # Create qubo.
    qi = -theta1 * qubo_returns - theta2 * qubo_linear  # (p, p).  eq (21a)
    qij = theta2 * qubo_quadratic + theta3 * qubo_covariance  # (p, p). eq (21b)
    qubo_ = typing.cast(Array, qi + qij)
    qubo_matrix = get_upper_triangular(qubo_)
    qubo_dict = get_qubo_dict(qubo_matrix)

    return QuboData(
        q=qubo_dict,
        prices=prices,
        m=m,
        w=w,
        npp=npp,
        arp=anual_returns_partitions,
    )
