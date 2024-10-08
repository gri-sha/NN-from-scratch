import numpy as np


class NeuralNetwork:
    VALID_ACT = {"ReLU", "Sigmoid"} 

    def __init__(self, network_shape: list[int], init_weights_mult=0.01, learning_rate=0.01, act_type="ReLU"):
       
        self.shape = network_shape
        self.init_weigths_mult = init_weights_mult
        self.learning_rate = learning_rate

        if act_type in self.VALID_ACT:
            self.act_type = act_type

        self.layers = [Layer(num_neurons, network=self) for num_neurons in self.shape]
        self.connect()
        self.initialize()

    def connect(self):
        for i in range(1, len(self.layers)):
            self.layers[i].prev_layer = self.layers[i - 1]

    def initialize(self):
        for layer in self.layers:
            layer.initialize()

    def forward(self, input_data):
        # self.layers[0] is not considered since it is the input layer
        for layer in self.layers[1:]:
            input_data = layer.forward(input_data)
        return input_data
    
    @staticmethod
    def cost(output, true_vals):
        return 0.5 * (output - input)**2

    def backward(self, output_error):
        # self.layers[0] is not considered since it is the input layer
        for layer in reversed(self.layers[1:]):
            output_error = layer.backward(output_error)

    def __str__(self):
        result = []

        # Header
        result.append(f"Shape: {self.shape}, Act: {self.act_type}, LR: {self.learning_rate}")
        result.append(f"{'Layer':<5} {'Neurons':<10} {'Weights':<60} {'Biases':<30}")
        result.append("-" * 105)  # Separator line

        # Layer details
        for i, layer in enumerate(self.layers):
            # Format weights
            if layer.weights is None:
                formatted_weights = 'None'
            else:
                weights_flat = layer.weights.flatten()
                if len(weights_flat) > 6:
                    first_three = ', '.join(f"{weight:.4f}" for weight in weights_flat[:3])
                    last_three = ', '.join(f"{weight:.4f}" for weight in weights_flat[-3:])
                    formatted_weights = f"{first_three}, ..., {last_three}"
                else:
                    formatted_weights = ', '.join(f"{weight:.4f}" for weight in weights_flat)

            # Format biases
            if layer.biases is None:
                formatted_biases = 'None'
            else:
                biases_flat = layer.biases.flatten()
                if len(biases_flat) > 4:
                    formatted_biases = ', '.join(f"{bias:.2f}" for bias in biases_flat[:4]) + ", ..."
                else:
                    formatted_biases = ', '.join(f"{bias:.2f}" for bias in biases_flat)

            # Append layer information
            result.append(f"{i + 1:<5} {layer.num_neurons:<10} {formatted_weights:<60} {formatted_biases:<30}")

        return "\n"+"\n".join(result)+"\n"  # Join all lines into a single string



class Layer:
    def __init__(self, num_neurons, network: 'NeuralNetwork'):
        self.num_neurons = num_neurons
        self.network = network

        self.weights = None
        self.biases = None 
        self.prev_layer = None
        self.inputs = None
        self.outputs = None
        self.activations = None

    def initialize(self):
        if self.prev_layer is not None:
            input_size = self.prev_layer.num_neurons
            self.weights = np.random.randn(input_size, self.num_neurons) * self.network.init_weigths_mult # np.random.randn() return vaulues in [-1, 1] interval
            self.biases = np.zeros((1, self.num_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        z = np.dot(self.inputs, self.weights) + self.biases

        if self.network.act_type == "ReLU":
            self.activations = self.relu(z)
        elif self.network.act_type == "Sigmoid":
            self.activations = self.sigmoid(z)
        
        return self.activations

    def backward(self, output_error):
        if self.network.act_type == "ReLU": 
            dz = output_error * self.relu_derivative(self.activations)
        elif self.network.act_type == "Sigmoid":
            dz = output_error * self.sigmoid_derivative(self.activations)

        output_error_prev = np.dot(dz, self.weights.T) # .T <=> transposed

        # Update weights and biases
        self.weights -= self.network.learning_rate * np.dot(self.inputs.T, dz)
        self.biases -= self.network.learning_rate * np.sum(dz, axis=0, keepdims=True)

        return output_error_prev

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(activations):
        return np.where(activations > 0, 1.0, 0.0)
    
    @staticmethod
    def sigmoid(z):
        return 1/(1 + np.exp(-z))
    
    @staticmethod
    def sigmoid_derivative(activations):
        return Layer.sigmoid(activations) * (1 - Layer.sigmoid(activations))


if __name__ == "__main__":
    IN = 5
    OUT = 5

    nn = NeuralNetwork([IN, 7, 6, 7, OUT], act_type="Sigmoid")  # 3 input neurons, 5 hidden neurons, 2 output neurons
    print(nn)

    input_data = np.random.rand(1, IN)
    output = nn.forward(input_data)
    print("Output:", output)


    output_error = np.random.rand(1, OUT)
    print("Error:", output_error)

    nn.backward(output_error)
    print(nn)
