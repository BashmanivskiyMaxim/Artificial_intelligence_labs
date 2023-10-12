import random
import random
import matplotlib.pyplot as plt
import csv
import numpy as np
from line_profiler import LineProfiler
import psutil

process = psutil.Process()  # Отримати посилання на поточний процес

# Зчитування даних з файлу .csv
def read_distance_matrix(csv_file):
    with open(csv_file, newline="", encoding="mac_cyrillic") as file:
        reader = csv.reader(file)
        rows = list(reader)
    cities = rows[0][1:]  # Перша рядок містить назви міст, відкидаємо перший стовпець
    distance_data = [
        list(map(int, row[1:])) for row in rows[1:]
    ]  # Зчитуємо дані з файла, відкидаємо перший стовпець
    distance_matrix = np.array(distance_data)
    return cities, distance_matrix


csv_file = "Відстань.csv"
csv_file_test1 = "test1.csv"
csv_file_test2 = "test2.csv"
cities, distances = read_distance_matrix(csv_file)
num_cities = len(cities)

# Параметри методу мурашиних колоній
num_ants = 50
max_iterations = 1000
pheromone_evaporation = 0.5
pheromone_deposit = 1.0
alpha = 4.0
beta = 1.0

# Ініціалізація феромонів на шляхах
pheromone = [[1.0] * num_cities for _ in range(num_cities)]

# Основна функція для розв'язання задачі комівояжера методом мурашиних колоній
def solve_tsp():
    best_tour = None
    best_distance = float("inf")

    for iteration in range(max_iterations):
        ant_tours = []

        for ant in range(num_ants):
            tour = construct_tour()
            ant_tours.append(tour)

        update_pheromone(ant_tours)

        for tour in ant_tours:
            distance = tour_distanceCalc(tour)
            if distance < best_distance:
                best_distance = distance
                best_tour = tour

    return best_tour, best_distance


# Функція для конструювання маршруту одного мурахи
def construct_tour():
    tour = []
    start_city = 2 #random.randint(0, num_cities - 1)
    tour.append(start_city)

    while len(tour) < num_cities:
        next_city = select_next_city(tour, pheromone[tour[-1]])
        tour.append(next_city)

    return tour


# Функція для вибору наступного міста для мурахи з урахуванням феромонів і відстаней
def select_next_city(visited, pheromone_values):
    unvisited_cities = [city for city in range(num_cities) if city not in visited]

    probabilities = [
        calculate_probability(visited[-1], city, pheromone_values)
        for city in unvisited_cities
    ]
    selected_city = random.choices(unvisited_cities, probabilities)[0]

    return selected_city


# Функція для розрахунку ймовірностей для вибору наступного міста
def calculate_probability(current_city, next_city, pheromone_values):
    pheromone = pheromone_values[next_city]
    distance = distances[current_city][next_city]
    probability = (pheromone**alpha) * ((1 / distance) ** beta)
    return probability


# Функція для оновлення рівня феромонів на шляхах після кожної ітерації
def update_pheromone(ant_tours):
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                pheromone[i][j] *= 1 - pheromone_evaporation

    for tour in ant_tours:
        tour_distance = tour_distanceCalc(tour)
        for i in range(num_cities - 1):
            city1, city2 = tour[i], tour[i + 1]
            pheromone[city1][city2] += pheromone_deposit / tour_distance


# Функція для обчислення відстані подорожі
def tour_distanceCalc(tour):
    distance = 0
    for i in range(len(tour) - 1):
        city1, city2 = tour[i], tour[i + 1]
        distance += distances[city1][city2]
    return distance


# Візуалізація результатів
def visualize_tsp_solution_with_dots(cities, tour):
    # З'єднання міст у порядку маршруту
    for i in range(len(tour) - 1):
        city1 = tour[i]
        city2 = tour[i + 1]
        x1, y1 = i, cities[city1][0]
        x2, y2 = i + 1, cities[city2][0]
        plt.plot([x1, x2], [y1, y2], "r")

    # Додавання чорних точок на кожну точку маршруту
    for i in range(len(tour)):
        x, y = i, cities[tour[i]][0]
        plt.scatter(x, y, color="black", s=30)

    plt.title("Маршрут комівояжера")
    plt.xlabel("Міста (номери)")
    plt.ylabel("Назви міст")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()


def visualize_tsp_solution(cities, tour):
    plt.plot([cities[i] for i in tour], "o-")
    plt.xlabel("Міста")
    plt.ylabel("Відстань")
    plt.title("Найкращий маршрут задачі комівояжера")
    plt.show()

profiler = LineProfiler()
profiler.add_function(solve_tsp)
profiler.enable()
# Розв'язання задачі комівояжера і виведення результату
best_tour, best_distance = solve_tsp()
best_tour.append(best_tour[0])

print("Найкращий маршрут:", best_tour)
print("Загальна відстань:", best_distance)

# Візуалізуємо найкоротший маршрут
visualize_tsp_solution_with_dots(cities, best_tour)
visualize_tsp_solution(cities, best_tour)



profiler.print_stats()
cpu_usage = process.cpu_percent()
print("Використання CPU:", cpu_usage, "%")
