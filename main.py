import random
import numpy as np
import matplotlib.pyplot as plt

class AntColonyOptimization:
    def __init__(self, num_ants, evaporation_rate, alpha, beta, num_iterations):
        self.num_ants = num_ants  # Кількість мурах (кількість шляхів, що будуть побудовані)
        self.evaporation_rate = evaporation_rate  # Коефіцієнт випаровування феромону
        self.alpha = alpha  # Ваговий коефіцієнт для феромонів
        self.beta = beta  # Ваговий коефіцієнт для відстаней
        self.num_iterations = num_iterations  # Кількість ітерацій алгоритму

    def generate_cities_map(self, num_cities):
        cities_map = np.zeros((num_cities, num_cities))

        # Генерація матриці відстаней між містами
        for i in range(num_cities):
            for j in range(i + 1, num_cities):
                distance = random.randint(10, 100)
                cities_map[i][j] = distance
                cities_map[j][i] = distance

        return cities_map

    def initialize_pheromone_map(self, num_cities):
        return np.ones((num_cities, num_cities))

    def run(self, num_cities):
        cities_map = self.generate_cities_map(num_cities)
        pheromone_map = self.initialize_pheromone_map(num_cities)
        best_path = None
        best_distance = float('inf')
        all_paths = []

        for _ in range(self.num_iterations):
            paths = self.construct_paths(cities_map, pheromone_map)
            self.update_pheromone_map(cities_map, pheromone_map, paths)
            path_lengths = [self.calculate_path_length(cities_map, path) for path in paths]
            min_distance = min(path_lengths)
            min_index = path_lengths.index(min_distance)

            # Збереження найкращого шляху
            if min_distance < best_distance:
                best_distance = min_distance
                best_path = paths[min_index]
            all_paths.extend(paths)

        return best_path, best_distance, all_paths

    def construct_paths(self, cities_map, pheromone_map):
        num_cities = cities_map.shape[0]
        paths = []

        # Побудова шляхів для кожної мурашки
        for ant in range(self.num_ants):
            visited = [False] * num_cities
            path = []
            current_city = random.randint(0, num_cities - 1)
            visited[current_city] = True
            path.append(current_city)

            while len(path) < num_cities:
                next_city = self.select_next_city(cities_map, pheromone_map, current_city, visited)
                visited[next_city] = True
                path.append(next_city)
                current_city = next_city

            paths.append(path)

        return paths

    def select_next_city(self, cities_map, pheromone_map, current_city, visited):
        num_cities = cities_map.shape[0]
        unvisited_probabilities = []
        total = 0.0

        # Обчислення ймовірностей переходу до невідвіданих міст
        for city in range(num_cities):
            if not visited[city]:
                pheromone = pheromone_map[current_city][city]
                visibility = 1.0 / cities_map[current_city][city]
                unvisited_probabilities.append((city, pheromone ** self.alpha * visibility ** self.beta))
                total += pheromone ** self.alpha * visibility ** self.beta

        # Нормалізація ймовірностей та вибір міста для переходу
        probabilities = [prob[1] / total for prob in unvisited_probabilities]
        next_city = np.random.choice([prob[0] for prob in unvisited_probabilities], p=probabilities)

        return next_city

    def update_pheromone_map(self, cities_map, pheromone_map, paths):
        pheromone_map *= self.evaporation_rate

        # Оновлення феромонів на всіх шляхах
        for path in paths:
            path_length = self.calculate_path_length(cities_map, path)
            for i in range(len(path) - 1):
                city_a = path[i]
                city_b = path[i + 1]
                pheromone_map[city_a][city_b] += 1.0 / path_length
                pheromone_map[city_b][city_a] += 1.0 / path_length

    def calculate_path_length(self, cities_map, path):
        path_length = 0

        # Обчислення довжини шляху
        for i in range(len(path) - 1):
            city_a = path[i]
            city_b = path[i + 1]
            path_length += cities_map[city_a][city_b]

        return path_length

    def plot_paths(self, cities_map, paths):
        num_cities = cities_map.shape[0]

        # Візуалізація всіх шляхів
        for path in paths:
            x = [city % 10 for city in path]
            y = [city // 10 for city in path]
            plt.plot(x, y, marker='o')

        plt.xticks(range(10))
        plt.yticks(range(4))
        plt.grid()
        plt.show()

    def plot_best_path(self, cities_map, best_path):
        num_cities = cities_map.shape[0]
        x = [city % 10 for city in best_path]
        y = [city // 10 for city in best_path]

        # Plotting the best path
        plt.plot(x, y, marker='o', color='red')
        plt.xticks(range(10))
        plt.yticks(range(4))
        plt.grid()

        # Adding numbering to each point
        for i, (xi, yi) in enumerate(zip(x, y)):
            plt.text(xi, yi, str(i), color='black', fontsize=8, ha='center', va='center')

        # Highlighting the first and last points with a larger black circle
        plt.plot(x[0], y[0], marker='o', markersize=10, color='green')
        plt.plot(x[-1], y[-1], marker='o', markersize=10, color='green')

        plt.show()


# Параметри алгоритму
num_ants = 10
evaporation_rate = 0.5
alpha = 1
beta = 2
num_iterations = 100

# Створення об'єкту алгоритму та запуск
aco = AntColonyOptimization(num_ants, evaporation_rate, alpha, beta, num_iterations)
best_path, best_distance, all_paths = aco.run(random.randint(25, 35))

print("Найкоротший шлях:", best_path)
print("Довжина найкоротшого шляху:", best_distance)

# Візуалізація всіх маршрутів
aco.plot_paths(aco.generate_cities_map(len(best_path)), all_paths)

# Візуалізація найкращого маршруту
aco.plot_best_path(aco.generate_cities_map(len(best_path)), best_path)
