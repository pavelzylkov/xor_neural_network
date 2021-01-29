import numpy as np

# Создадим класс для нашей нейронной сети
class NuralNetwork():

  def __init__(self):
    # Установим генератор случайных для постоянного распределения
    np.random.seed(1)
    # Объявим веса, т. к. у нас нейронная сеть имеет 3 значения на входе и одно на выходе
    # то поместим наши веса в матрицу [3 x 1]. Значения весов лежат в диапозоне [-1, 1]
    self.syn_weights = 2 * np.random.random((3, 1)) - 1

  # Функция активации - сигмоида. Для нормализации значений от 0 до 1
  def __sigmoid(self, x):
    return 1 / (1 + np.exp(-x))

  # Функция для подсчета значения производной сигмоиды
  def __sigmoid_derivative(self, x):
    return x * (1 - x)

  # Прямое распространение
  def think(self, inputs):
    return self.__sigmoid(np.dot(inputs, self.syn_weights))

  # Функция "обучения"
  def train(self, trainig_inputs, training_outputs, number_of_training_iterations):
    for i in range(number_of_training_iterations):
      output = self.think(trainig_inputs)
      
      # Подсче ошибки, ее коректировка
      error = training_outputs - output
      adjustments = np.dot(trainig_inputs.T, error * self.__sigmoid_derivative(output))
      
      # Обновление весов
      self.syn_weights += adjustments

#if __name__ == "main":
nural_network = NuralNetwork()

print("Случайные веса: ")
print(nural_network.syn_weights)


training_inputs = np.array([[0, 0, 1],
                            [0, 1, 1],
                            [1, 0, 1],
                            [1, 1, 1]])

training_outputs = np.array([[0, 0, 1, 1]]).T

nural_network.train(training_inputs, training_outputs, 20000)

print("Веса после обучения:")
print(nural_network.syn_weights)

print("Тест [1, 0, 0] -> ?: ")
print(nural_network.think(np.array([1, 0, 0])))