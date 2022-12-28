# coding=utf-8

from dwave.system import LeapHybridSampler
from dwave.system.samplers import \
    DWaveSampler  # Library to interact with the QPU


class DWaveSolver(object):

    _URL = "https://na-west-1.cloud.dwavesys.com/sapi/v2/"

    def __init__(self, qubo, qubo_dict, runs, chainstrength, anneal_time, solver, API):

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # OBTENEMOS LOS VALORES PARA RESOLVER EL PROBLEMA A TRAVES DE DWAVE
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.chainstrength = chainstrength
        self.annealing_time = anneal_time
        self.numruns = runs
        self.qubo = qubo
        self.qubo_dict = qubo_dict
        self.solver = solver
        # self.sapi_token = 'DEV-d9751cb50bc095c993f55b3255f728d5b2793c36'
        self.sapi_token = API

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # RESOLVEMOS EL PROBLEMA EMPLEANDO DWAVE, LOS PARAMETROS ARRIBA INDICADOS
    # Y EL QUBO CON EL QUE HEMOS INICIALIZADO LA CLASE
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def solve_DWAVE_Advantadge_QUBO(self):

        if self.solver == "hybrid_solver":
            sampler = LeapHybridSampler(token=self.sapi_token, endpoint=self._URL)

        ######### Resolvemos el problema y obtenemos la solución #########
        if self.solver == "hybrid_solver" or self.solver == "exact":
            self.dwave_return = sampler.sample_qubo(self.qubo_dict)

        self.dwave_raw_array = self.dwave_return.record.sample
        self.num_occurrences = self.dwave_return.record.num_occurrences
        self.energies = self.dwave_return.record.energy

        ######### Devolvemos la solucion, ocurrencias y energias #########
        return (
            self.dwave_return,
            self.dwave_raw_array,
            self.num_occurrences,
            self.energies,
        )
