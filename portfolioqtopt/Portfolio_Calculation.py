# coding=utf-8


from .dwave_solver import DWaveSolver
from .PortfolioSelection import PortfolioSelection
from .QuboBuilder import QUBO
from .reader import read_welzia_stocks_file


def Portfolio_Calculation(
    API: str,  # clave acceso leap
    slices_arg: int,
    file_name: str,  # A path
    sheet: str,  # sheet name in excel file
    fondos: int,
    theta_one_arg=0.9,
    theta_two_arg=0.4,
    theta_three_arg=0.1,
    lista_fondos_invertidos=[],
):

    # •	Número de fondos (45): el número de fondos que componen el universo. Es decir, los assets donde se puede llevar a cabo una inversión. En el fichero que os enviamos en este mail, por ejemplo, este número es de 45.

    ######### Los slices son las proporciones con las que vamos a poder jugar con nuestras acciones #########
    slice_nums = {1: 1, 2: 2, 4: 4, 5: 5, 6: 6, 8: 8, 10: 10}
    slices = slice_nums[slices_arg]

    ######### Elegimos el tipo de embedding #########
    # TODO Seria interesante poner esto en una enumeration
    solver_types = {
        1: "Clique_Embedding",
        2: "Find_Embedding",
        3: "hybrid_solver",
        4: "SA",
        5: "exact",
    }
    choose_solver = solver_types[3]

    ######### Escogemos el chain strength #########
    chain_strengths = {0.5: 0.5, 0.75: 0.75, 1: 1, 1.25: 1.25, 2: 2}
    chain_strength = chain_strengths[1]

    ######### Elegimos el numero de runs #########
    number_runs = {10: 10, 500: 500, 1000: 1000, 2000: 2000, 10000: 10000}
    runs = 10000

    ######### Elegimos el annealing time #########
    anneal_time_dict = {1: "1", 5: "5", 100: "100", 250: "250", 500: "500", 999: "999"}
    anneal_time = [500]

    ######### Elegimos los parámetros de penalización del hamiltoniano* #########
    theta_one = theta_one_arg  # esta es la variable asociada al retorno
    theta_two = theta_two_arg  # esta es la variable asociada a que se cumpla la restricción del budget
    theta_three = theta_three_arg  # esta es la variable asociada al riesgo

    # print ('weights: (theta_one, theta_two, theta_three)', theta_one, theta_two, theta_three)

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

    if len(lista_fondos_invertidos) == 0:
        price_data_df = df_price_data[:days]
    else:
        price_data_df = df_price_data[lista_fondos_invertidos]
    price_data = price_data_df.values[:, :assets]

    # np.testing.assert_equal(df_price_data.to_numpy(), df.to_numpy())

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # OBTENEMOS LOS VALORES QUE VAN A COMPONER LA MATRIZ QUBO
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    portfolio_selection = PortfolioSelection(
        theta_one, theta_two, theta_three, price_data, slices
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
        qubo_matrix, qubo_dict, runs, chain_strength, anneal_time, choose_solver, API
    )

    # Resolvemos el problema
    (
        dwave_return,
        dwave_raw_array,
        num_occurrences,
        energies,
    ) = dwave_solve.solve_DWAVE_Advantadge_QUBO()

    return dwave_raw_array, portfolio_selection, new_header, price_data
