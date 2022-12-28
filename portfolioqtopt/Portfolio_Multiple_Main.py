# coding=utf-8
import sys
import time

import numpy
import pandas as pd

from portfolioqtopt import portfolio_calculation


def interpretarResultados(
    fondos,
    slices,
    dwave_raw_array,
    portfolio_selection,
    new_header,
    price_data,
    lista_fondos_invertidos=[],
):
    inversiones = []
    inversiones_unicas = []
    cantidades_unicas = []

    for i in range(fondos):
        zeros = dwave_raw_array[0][i * slices : i * slices + slices]
        prices = portfolio_selection.last_prices[0:slices]
        if 1 in zeros:
            result = [num1 * num2 for num1, num2 in zip(zeros, prices)]
            # print(new_header[i + 2], sum(result))
            inversiones.append(sum(result))
            if new_header[i + 2] not in lista_fondos_invertidos:
                lista_fondos_invertidos.append(new_header[i + 2])
            inversiones_unicas.append(new_header[i + 2])
            cantidades_unicas.append(sum(result))
        else:
            inversiones.append(0.0)

    inicial = dwave_raw_array[0] * portfolio_selection.price_data_reversed[0]
    final = (
        dwave_raw_array[0]
        * portfolio_selection.price_data_reversed[
            len(portfolio_selection.price_data_reversed) - 1
        ]
    )
    retornos = final - inicial
    # print("Expected return: ", str(100 * numpy.sum(retornos)))

    desviaciones = 0.0

    portfolio_selection.price_data = portfolio_selection.price_data * 100
    portfolio_selection.price_data_reversed = (
        portfolio_selection.price_data_reversed * 100
    )

    for i in range(len(inversiones)):
        desviaciones = desviaciones + (
            pow(inversiones[i], 2) * pow(numpy.std(price_data[:, i]), 2)
        )

    covarianzas = 0.0

    for i in range(len(inversiones)):
        for j in range(len(inversiones)):
            if i != j:
                covarianzas = covarianzas + (
                    inversiones[i]
                    * inversiones[j]
                    * (
                        numpy.cov(
                            pd.to_numeric(price_data[:, i]),
                            pd.to_numeric(price_data[:, j]),
                        )[0][1]
                    )
                )

    # print("Risk: ", str(numpy.sqrt(desviaciones + covarianzas)))

    # print("----------------------------")

    return (
        lista_fondos_invertidos,
        inversiones_unicas,
        cantidades_unicas,
        str(100 * numpy.sum(retornos)),
        str(numpy.sqrt(desviaciones + covarianzas)),
        (100 * numpy.sum(retornos)) / (numpy.sqrt(desviaciones + covarianzas)),
    )


def main():
    arguments = sys.argv

    repetitions = int(arguments[1])  # 5
    API = arguments[2]  #'DEV-e4fdb27d716a5f9ecfb0da6bea60c20cd7ac6b57'
    slices = int(arguments[3])  # 5
    file_name = arguments[4]  # 'Data\Histórico_carteras_Welzia_2017'
    sheet = arguments[5]  # 'BBG (valores)'
    fondos = int(arguments[6])  # 45

    lista_fondos_invertidos = []

    for i in range(repetitions):

        (
            dwave_raw_array,
            portfolio_selection,
            new_header,
            price_data,
        ) = portfolio_calculation.Portfolio_Calculation(
            API, slices, file_name, sheet, fondos
        )

        lista_fondos_invertidos = interpretarResultados(
            fondos,
            slices,
            dwave_raw_array,
            portfolio_selection,
            new_header,
            price_data,
            lista_fondos_invertidos,
        )[0]

    # print("FINAL OPTIMIZATION with ", str(len(lista_fondos_invertidos)), " assets")

    SHARPE = sys.float_info.min

    for i in range(repetitions):
        (
            dwave_raw_array,
            portfolio_selection,
            new_header,
            price_data,
        ) = portfolio_calculation.Portfolio_Calculation(
            API,
            slices,
            file_name,
            sheet,
            len(lista_fondos_invertidos),
            lista_fondos_invertidos=lista_fondos_invertidos,
        )
        (
            aux_fondos_finales,
            aux_inversiones_unicas,
            aux_inversiones,
            aux_return,
            aux_risk,
            aux_SHARPE,
        ) = interpretarResultados(
            len(lista_fondos_invertidos),
            slices,
            dwave_raw_array,
            portfolio_selection,
            new_header,
            price_data,
        )

        if aux_SHARPE > SHARPE:
            fondos_finales = aux_inversiones_unicas
            inversiones_finales = aux_inversiones
            retorno_final = aux_return
            riesgo_final = aux_risk
            SHARPE = aux_SHARPE

    print("#################  RESULTADOS FINALES  ####################")
    print("Fondos Finales: ", fondos_finales)
    print("Inversiones Finales: ", inversiones_finales)
    print("Retorno Esperado: ", retorno_final)
    print("Riesgo: ", riesgo_final)
    print("SHARPE: ", SHARPE)
    print("###########################################################")

    curr_time = round(time.time() * 1000)
    f = open("Output/results_" + str(curr_time) + ".txt", "w+")
    f.write("#################  RESULTADOS FINALES  ####################" + "\n")
    f.write("Fondos Finales: " + str(fondos_finales) + "\n")
    f.write("Inversiones Finales: " + str(inversiones_finales) + "\n")
    f.write("Retorno Esperado: " + str(retorno_final) + "\n")
    f.write("Riesgo: " + str(riesgo_final) + "\n")
    f.write("SHARPE: " + str(SHARPE) + "\n")
    f.write("###########################################################" + "\n")
    f.close()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
