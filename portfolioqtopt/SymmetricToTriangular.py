# coding=utf-8
#                                              Symmetric Matrix to Triangular Matrix
########################################################################################################################
# Lo siguiente clase genera matrices triangulares (más fáciles de incrustar para el D-Wave)
# a partir de matrices simétricas QUBO.
########################################################################################################################
# coding=utf-8


class TriangleGenerator:
    def __init__(self, n, q):
        self.n = n
        self.q = q
        self.i = None
        self.upper_matrix = None
        self.lower_matrix = None

    def upper(self):
        for col in range(0, self.n - 1):
            for row in range(col + 1, self.n):
                self.q[row, col] = 0
        for row in range(0, self.n - 1):
            for col in range(row + 1, self.n):
                self.q[row, col] = 2 * self.q[row, col]
        self.upper_matrix = self.q

    def lower(self):
        for row in range(0, self.n - 1):
            for col in range(row + 1, self.n):
                self.q[row, col] = 0
        for col in range(0, self.n - 1):
            for row in range(col + 1, self.n):
                self.q[row, col] = 2 * self.q[row, col]
        self.lower_matrix = self.q
