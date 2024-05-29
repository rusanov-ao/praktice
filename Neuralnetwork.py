import numpy as np

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return self.activation(total)

    def activation(self, x):
        return 1 / (1 + np.exp(-x))  # Сигмоидная функция активации

    def derivative(self, x):
        return self.activation(x) * (1 - self.activation(x))  # Производная сигмоидной функции

class NeuralBlock:
    def __init__(self, neurons):
        self.neurons = neurons

    def forward(self, inputs):
        outputs = [neuron.forward(inputs) for neuron in self.neurons]
        return outputs

class NeuralNetwork:
    def __init__(self, neural_blocks):
        self.neural_blocks = neural_blocks

    def forward(self, inputs):
        current_outputs = inputs
        for block in self.neural_blocks:
            current_outputs = block.forward(current_outputs)
        return current_outputs

    def compute_loss(self, predicted, actual):
        return np.square(predicted - actual).mean()  # Mean Squared Error loss function

    def update_weights(self, learning_rate):
        for block in self.neural_blocks:
            for neuron in block.neurons:
                neuron.weights -= learning_rate * neuron.derivative(neuron.weights)
                neuron.bias -= learning_rate * neuron.derivative(neuron.bias)

# Создаем нейроны для первого слоя
neuron1 = Neuron(weights=np.array([0.2, 0.3, 0.4]), bias=0.1)
neuron2 = Neuron(weights=np.array([0.5, 0.6, 0.7]), bias=0.2)

# Создаем нейроны для второго слоя
neuron3 = Neuron(weights=np.array([0.1, 0.2]), bias=0.3)
neuron4 = Neuron(weights=np.array([0.3, 0.4]), bias=0.5)

# Создаем нейронные блоки
neural_block1 = NeuralBlock(neurons=[neuron1, neuron2])
neural_block2 = NeuralBlock(neurons=[neuron3, neuron4])

# Создаем нейронную сеть
neural_network = NeuralNetwork(neural_blocks=[neural_block1, neural_block2])

# Пропускаем входные данные через нейронную сеть
inputs = np.array([0.5, 0.6, 0.7])
outputs = neural_network.forward(inputs)

# Вычисляем потери
actual_outputs = np.array([0.1, 0.2])  # Замените на реальные значения
loss = neural_network.compute_loss(outputs, actual_outputs)
print(f"Loss: {loss}")

# Обновляем веса
learning_rate = 0.01
neural_network.update_weights(learning_rate)