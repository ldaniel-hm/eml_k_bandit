import numpy as np

from algorithms.algorithm import Algorithm


class EpsilonGreedy(Algorithm):

    def __init__(self, k: int, epsilon: float = 0.1):
        """
        Inicializa el algoritmo epsilon-greedy.

        :param k: Número de brazos.
        :param epsilon: Probabilidad de exploración (seleccionar un brazo al azar).
        :raises ValueError: Si epsilon no está en [0, 1].
        """
        assert 0 <= epsilon <= 1, "El parámetro epsilon debe estar entre 0 y 1."

        super().__init__(k)
        self.epsilon = epsilon

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política epsilon-greedy.
        :return: índice del brazo seleccionado.
        """

        if np.random.random() < self.epsilon:
            # Selecciona un brazo al azar
            chosen_arm = np.random.choice(self.k)
        else:
            # Selecciona el brazo con la recompensa promedio estimada más alta
            chosen_arm = np.argmax(self.values)

        return chosen_arm




