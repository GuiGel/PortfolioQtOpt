# coding=utf-8

from enum import Enum, unique
from typing import List, Optional

from .dwave_solver import DWaveSolver
from .PortfolioSelection import PortfolioSelection
from .QuboBuilder import QUBO
from .reader import read_welzia_stocks_file


@unique
class SolverTypes(Enum):
    Clique_Embedding = "Clique_Embedding"
    Find_Embedding = "Find_Embedding"
    hybrid_solver = "hybrid_solver"
    SA = "SA"
    exact = "exact"


def Portfolio_Calculation(
    api_token: str,  # clave acceso leap
    num_slices: int,
    file_name: str,  # A path
    sheet: str,  # sheet name in excel file
    fondos: int,
    theta1: float = 0.9,
    theta2: float = 0.4,
    theta3: float = 0.1,
    lista_fondos_invertidos: Optional[List[str]] = None,
    solver_type: SolverTypes = SolverTypes.hybrid_solver,
):
    """Compute the portfolio.

    theta1, 2, and 3 are the penalty parameters of the Hamiltonian.

    Args:
        api_token (str): _description_
        num_slices (int): Choose the number of slices. Must be >= 0.
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

    Returns:
        _type_: _description_
    """

    # Número de fondos (45): el número de fondos que componen el universo.
    # Es decir, los assets donde se puede llevar a cabo una inversión.
    # En el fichero que os enviamos en este mail, por ejemplo, este número es de 45.

    # Los slices son las proporciones con las que vamos a poder jugar con
    # nuestras acciones
    assert num_slices >= 1
    slices_num = num_slices * 2  # NOTE: Is the num_slices alway pair?

    # Elegimos el tipo de embedding.
    choose_solver = solver_type.value

    # Escogemos el chain strength
    chain_strengths = {0.5: 0.5, 0.75: 0.75, 1: 1, 1.25: 1.25, 2: 2}
    chain_strength = chain_strengths[1]

    # Elegimos el numero de runs
    number_runs = {10: 10, 500: 500, 1000: 1000, 2000: 2000, 10000: 10000}
    runs = 10000

    # Elegimos el annealing time
    anneal_time_dict = {1: "1", 5: "5", 100: "100", 250: "250", 500: "500", 999: "999"}
    anneal_time = [500]

    ######### Fijamos el número de días y el numero de fondos que vamos a considerar del dataset completo #########
    days = 8000
    assets = fondos

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # CARGAMOS LOS DATOS DEL PROBLEMA Y GENERAMOS UN DATAFRAME
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    df_price_data = read_welzia_stocks_file(file_path=file_name, sheet_name=sheet)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # AQUI VAMOS A FILTRAR, EN CASO DE SER NECESARIO, EL DATASET
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    if lista_fondos_invertidos is None:
        price_data_df = df_price_data[:days]
    else:
        price_data_df = df_price_data[lista_fondos_invertidos]
    price_data = price_data_df.values[:, :assets]

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # OBTENEMOS LOS VALORES QUE VAN A COMPONER LA MATRIZ QUBO
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    portfolio_selection = PortfolioSelection(
        theta1, theta2, theta3, price_data, slices_num
    )

    # qi son los valores de la diagonal
    qi = portfolio_selection.qi

    # qij son los valores que se colocan por encima de la diagonal
    qij = portfolio_selection.qij

    # Generamos la clase QUBO y configuramos la matriz y el diccionario
    qubo = QUBO(qi, qij)
    qubo_matrix = qubo.qubo
    qubo_dict = qubo.qubo_dict

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # RESOLUCIÓN DEL PROBLEMA
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # Configuramos el solver
    dwave_solve = DWaveSolver(
        qubo_matrix,
        qubo_dict,
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

    return dwave_raw_array, portfolio_selection, new_header, price_data
