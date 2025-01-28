# This is a sample Python script.
from typing import List

import numpy as np

from algorithms.algorithm import Algorithm
from algorithms.epsilon_greedy import EpsilonGreedy
from arms.armnormal import ArmNormal
from arms.bandit import Bandit
from plotting.plotting import plot_average_rewards, plot_optimal_selections


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

def run_experiment(bandit: Bandit, algorithms: List[Algorithm], steps: int, runs: int):
    """
    Ejecuta experimentos comparativos entre diferentes algoritmos.

    :param bandit: Instancia de Bandit configurada para el experimento.
    :param algorithms: Lista de instancias de algoritmos a comparar.
    :param steps: Número de pasos de tiempo por ejecución.
    :param runs: Número de ejecuciones independientes.
    :return: Tuple de tres elementos: recompensas promedio, porcentaje de selecciones óptimas, y estadísticas de brazos.
    :rtype: Tuple of (np.ndarray, np.ndarray, list)
    """

    k = bandit.k
    optimal_arm = bandit.optimal_arm

    # Inicializar matrices para recompensas y selecciones óptimas
    rewards = np.zeros((len(algorithms), steps))
    optimal_selections = np.zeros((len(algorithms), steps))


    for run in range(runs):
        # Crear una nueva instancia del bandit para cada ejecución
        current_bandit = Bandit(arms=bandit.arms)

        # Obtener la recompensa esperada óptima
        q_max = current_bandit.get_expected_value(current_bandit.optimal_arm)

        for algo in algorithms:
            algo.reset()

        # Inicializar recompensas acumuladas por algoritmo para esta ejecución
        total_rewards_per_algo = np.zeros(len(algorithms))  # Para análisis por rechazo

        # Inicializar recompensas acumuladas por algoritmo para esta ejecución
        # cumulative_rewards_per_algo = np.zeros(len(algorithms))

        for step in range(steps):
            for idx, algo in enumerate(algorithms):
                chosen_arm = algo.select_arm()
                reward = current_bandit.pull_arm(chosen_arm)
                algo.update(chosen_arm, reward)

                rewards[idx, step] += reward
                total_rewards_per_algo[idx] += reward

                if chosen_arm == optimal_arm:
                    optimal_selections[idx, step] += 1

    # Promediar las recompensas y el regret sobre todas las ejecuciones
    rewards /= runs
    optimal_selections = (optimal_selections / runs) * 100

    return rewards, optimal_selections




def main():
    """
    Main function to set up and execute comparative experiments.
    """

    seed = 42
    np.random.seed(seed)

    k = 10 # Número de brazos
    steps = 1000  # Número de pasos
    runs = 500  # Número de ejecuciones

    bandit = Bandit(arms=ArmNormal.generate_arms(k))  # Bandit(arms=ArmBinomial.generate_arms(k))
    # bandit = Bandit(arms=ArmBernoulli.generate_arms(k))  # Bandit(arms=ArmBinomial.generate_arms(k))
    print(bandit)

    optimal_arm: int = bandit.optimal_arm
    print(f"Optimal arm: {optimal_arm + 1} with expected reward={bandit.get_expected_value(optimal_arm)}")

    algorithms = [EpsilonGreedy(k=k, epsilon=0), EpsilonGreedy(k=k, epsilon=0.01), EpsilonGreedy(k=k, epsilon=0.1)]

    # Ejecutar el experimento y obtener las recompensas promedio y selecciones óptimas
    rewards, optimal_selections = run_experiment(bandit, algorithms, steps, runs)

    # Generar las gráficas utilizando las funciones externas
    plot_average_rewards(steps, rewards, algorithms)
    # plot_optimal_selections(steps, optimal_selections, algorithms)





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
