from __future__ import annotations

from enum import Enum, unique
from typing import List, Optional, Union

from portfolioqtopt.dwave_solver import DWaveSolver
from portfolioqtopt.PortfolioSelection import PortfolioSelection
from portfolioqtopt.qubo import get_qubo
from portfolioqtopt.reader import read_welzia_stocks_file


@unique
class SolverTypes(Enum):
    Clique_Embedding = "Clique_Embedding"
    Find_Embedding = "Find_Embedding"
    hybrid_solver = "hybrid_solver"
    SA = "SA"
    exact = "exact"


def Portfolio_Calculation(
    api_token: str,  # clave acceso leap
    slices_num: int,
    file_name: str,  # A path
    sheet: str,  # sheet name in excel file
    fondos: int,
    theta1: float = 0.9,
    theta2: float = 0.4,
    theta3: float = 0.1,
    lista_fondos_invertidos: Optional[List[str]] = None,
    solver_type: SolverTypes = SolverTypes.hybrid_solver,
    runs: Optional[int] = None,
):
    """Compute the portfolio.

    theta1, 2, and 3 are the penalty parameters of the Hamiltonian.

    Args:
        api_token (str): _description_
        slices_num (int): Choose the number of slices. Must be >= 0.
        file_name (str): Name of the welzia data file.
        theta1 (float, optional): La variable asociada al retorno.
            Defaults to 0.9.
        theta2 (float, optional): La variable asociada a que se cumpla la
            restricción del budget. Defaults to 0.4.
        theta3 (float, optional): La variable asociada al riesgo.
            Defaults to 0.1.
        lista_fondos_invertidos (Optional[List[str]], optional): Name of some
            specific fund to select. Defaults to None.
        solver_type (SolverTypes, optional): Choose one type of solvers.
            Defaults to SolverTypes.hybrid_solver.
        runs (int, optional): Numero de runs. Defaults to None.

    Returns:
        _type_: _description_
    """

    # Número de fondos (45): el número de fondos que componen el universo.
    # Es decir, los fondos donde se puede llevar a cabo una inversión.
    # En el fichero que os enviamos en este mail, por ejemplo, este número es de 45.

    # Los slices son las proporciones con las que vamos a poder jugar con
    # nuestras acciones
    assert slices_num >= 1

    # Elegimos el tipo de embedding.
    choose_solver = solver_type.value

    # Escogemos el chain strength
    chain_strengths = {0.5: 0.5, 0.75: 0.75, 1: 1, 1.25: 1.25, 2: 2}
    chain_strength = chain_strengths[1]

    if runs is None:
        runs = 10000

    # Elegimos el annealing time
    anneal_time_dict = {1: "1", 5: "5", 100: "100", 250: "250", 500: "500", 999: "999"}
    anneal_time = [500]

    ######### Fijamos el número de días y el numero de fondos que vamos a considerar del dataset completo #########
    days = 8000

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # CARGAMOS LOS DATOS DEL PROBLEMA Y GENERAMOS UN DATAFRAME
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    prices_df = read_welzia_stocks_file(file_path=file_name, sheet_name=sheet)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # AQUI VAMOS A FILTRAR, EN CASO DE SER NECESARIO, EL DATASET
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    if lista_fondos_invertidos is None:
        prices_df = prices_df[:days]
    else:
        prices_df = prices_df[lista_fondos_invertidos]
        # NOTE: All this is redundant
    prices = prices_df.values[:, :fondos]

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # OBTENEMOS LOS VALORES QUE VAN A COMPONER LA MATRIZ QUBO
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    portfolio_selection = PortfolioSelection(theta1, theta2, theta3, prices, slices_num)

    # Generamos la clase QUBO y configuramos la matriz y el diccionario
    # qi son los valores de la diagonal
    # qij son los valores que se colocan por encima de la diagonal
    qubo = get_qubo(portfolio_selection.qi, portfolio_selection.qij)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # RESOLUCIÓN DEL PROBLEMA
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # Configuramos el solver
    dwave_solve = DWaveSolver(
        qubo.matrix,
        qubo.dictionary,
        runs,
        chain_strength,
        anneal_time,
        choose_solver,
        api_token,
    )

    # Resolvemos el problema
    (
        dwave_return,
        dwave_raw_array,
        num_occurrences,
        energies,
    ) = dwave_solve.solve_DWAVE_Advantadge_QUBO()

    return dwave_raw_array, portfolio_selection, new_header, prices


import numpy as np
import numpy.typing as npt


class Portfolio:
    def __init__(
        self,
        api_token: str,  # clave acceso leap
        prices: npt.NDArray[np.float64],
        slices_num: int,
        theta1: float = 0.9,
        theta2: float = 0.4,
        theta3: float = 0.1,
        solver_type: SolverTypes = SolverTypes.hybrid_solver,
        runs: Optional[int] = None,
        anneal_time: Union[List[int], Optional[int]] = None,
        chain_strength_arg: Optional[int] = None,
    ) -> None:
        self.api_token = api_token
        self.prices = prices
        self.slices_num = slices_num
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3
        self.solver_type = solver_type

        # TODO: Argument validation

        assert slices_num > 1

        # Initialize runs
        if runs is None:
            runs = 10000
        assert runs > 0
        self.runs = runs

        # Initialize anneal_time
        if anneal_time is None:
            anneal_time = [500]
        elif isinstance(anneal_time, int):
            anneal_time = [anneal_time]
        assert np.alltrue(np.array(anneal_time) > 0)
        self.anneal_time = anneal_time

        # Initialize chain strength
        if chain_strength_arg is None:
            chain_strength_arg = 3
        assert chain_strength_arg > 0
        self.chain_strength = chain_strength_arg * 0.25

    @classmethod
    def from_welzia(
        cls,
        api_token: str,
        file_name: str,
        sheet: str,
        fondos: int,
        slices_num: int,
        lista_fondos_invertidos: Optional[List[str]] = None,
        days: Optional[int] = None,
        theta1: float = 0.9,
        theta2: float = 0.4,
        theta3: float = 0.1,
        solver_type: SolverTypes = SolverTypes.hybrid_solver,
        runs: Optional[int] = None,
        anneal_time: Union[List[int], Optional[int]] = None,
        chain_strength_arg: Optional[int] = None,
    ) -> Portfolio:
        # We set the number of days and the number of funds we are going to consider
        # from the complete dataset.
        if days is None:
            days = 8000

        # Load Welzia from excel file as a pd.DataFrame
        prices_df = read_welzia_stocks_file(file_path=file_name, sheet_name=sheet)

        # Filter prices_df if needed.
        if lista_fondos_invertidos is None:
            prices_df = prices_df[:days]
        else:
            prices_df = prices_df[lista_fondos_invertidos]
            # NOTE: All this is redundant
        prices = prices_df.values[:, :fondos]
        return Portfolio(
            api_token,
            prices,
            slices_num,
            theta1,
            theta2,
            theta3,
            solver_type,
            runs,
            anneal_time,
            chain_strength_arg,
        )

    def __call__(self):
        # Obtain the values that are going to compose the qubo matrix
        portfolio_selection = PortfolioSelection(
            self.theta1, self.theta2, self.theta3, self.prices, self.slices_num
        )

        # Generamos la clase QUBO y configuramos la matriz y el diccionario
        # qi son los valores de la diagonal
        # qij son los valores que se colocan por encima de la diagonal
        qubo = get_qubo(portfolio_selection.qi, portfolio_selection.qij)

        # Resolution of the problem

        # 1. Configure the solver
        dwave_solve = DWaveSolver(
            qubo.matrix,
            qubo.dictionary,
            self.runs,
            self.chain_strength,
            self.anneal_time,
            self.solver_type.value,
            self.api_token,
        )

        # 2. Solve the problem
        (
            dwave_return,
            dwave_raw_array,
            num_occurrences,
            energies,
        ) = dwave_solve.solve_DWAVE_Advantadge_QUBO()

        return dwave_raw_array, portfolio_selection, new_header, prices
