import numpy as np
import activations as act
import losses


class DenseNetwork:
    def __init__(self, layers=[], init_weights_mult=0.01, learning_rate=0.01, name="My Model", loss="mean_squared_error"):     
        self.name = name
        self.layers = layers
        self.shape = [layer.num_neurons for layer in self.layers]
        self.init_weights_mult = init_weights_mult
        self.learning_rate = learning_rate

        if loss in losses.VALID_LOSSES:
            self.loss = loss
        else:
            raise ValueError("Invalid loss function")

        self.connect()
        self.initialize()

    def connect(self):
        self.layers[0].name  = "Input"
        self.layers[0].network = self
        for i in range(1, len(self.layers)):
            self.layers[i].name = f"H_{i}" if i != len(self.layers)-1 else "Output"
            self.layers[i].prev_layer = self.layers[i - 1]
            self.layers[i].network = self

    def initialize(self):
        for layer in self.layers:
            layer.initialize()

    def __str__(self):
        result = []

        result.append(
            f"{self.name}: "
            + " -> ".join(str(layer) for layer in self.layers)
            + f", LR: {self.learning_rate}\n"
        )
        result.append(f"{'Layer':<10} {'Neurons':<10} {'Weights':<60} {'Biases':<30}")
        result.append("-" * 110)

        for i, layer in enumerate(self.layers):
            
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

            if layer.biases is None:
                formatted_biases = 'None'
            else:
                biases_flat = layer.biases.flatten()
                if len(biases_flat) > 4:
                    formatted_biases = ', '.join(f"{bias:.2f}" for bias in biases_flat[:4]) + ", ..."
                else:
                    formatted_biases = ', '.join(f"{bias:.2f}" for bias in biases_flat)

            result.append(f"{layer.name:<10} {layer.num_neurons:<10} {formatted_weights:<60} {formatted_biases:<30}")

        return "\n"+"\n".join(result)+"\n"

    def forward(self, input_data):
        # self.layers[0] is not considered since it is the input layer
        for layer in self.layers[1:]:
            input_data = layer.forward(input_data)
        return input_data

    def backward(self, output_error):
        # self.layers[0] is not considered since it is the input layer
        for layer in reversed(self.layers[1:]):
            output_error = layer.backward(output_error)

    def fit(self, train_data, train_labels, mini_batch_size=10, epochs=8):
        if np.shape(train_data)[0] != np.shape(train_labels)[0]:
            raise ValueError("The number of training samples and labels must match.")
        
        input_size = train_data.shape[1] * train_data.shape[2]

        for _ in range(epochs):
            accumulated_error = np.zeros((1, 10))

            for i in range(len(train_data)):
                input = train_data[i].reshape((1, input_size))
                output = self.forward(input)

                true_val = np.zeros((1, 10))
                true_val[0][train_labels[i]] = 1
            
                error = (output - true_val)
                accumulated_error += error

                if (i + 1) % mini_batch_size == 0:
                    self.backward(accumulated_error)
                    accumulated_error = np.zeros((1, 10))

    def test(self, test_data, test_labels):
        right_ans = 0
        num_samples = len(test_data)

        for i in range(test_data.shape[0]):
            input_data = test_data[i].reshape((1, test_data.shape[1] * test_data.shape[2]))
            output = self.forward(input_data)

            predicted_label = np.argmax(output)
            if predicted_label == test_labels[i]:
                right_ans += 1

        accuracy = right_ans / num_samples if num_samples > 0 else 0.0
        print(f'Correct predictions: {right_ans}/{num_samples}')
        print(f'Accuracy: {accuracy:.3%}')

    def save(self):
        pass


# Dense layer is always fully connected
class DenseLayer:
    def __init__(self, num_neurons, act_type="linear", name="Billy"):
        self.num_neurons = num_neurons
        self.name = name

        if act_type in act.VALID_ACT:
            self.act_type = act_type
            self.act = act.VALID_ACT[self.act_type][0]
            self.act_der = act.VALID_ACT[self.act_type][1]
        else: 
            raise ValueError("Invalid activation type")

        self.network = None
        self.weights = None
        self.biases = None 
        self.prev_layer = None
        self.inputs = None
        self.outputs = None
        self.activations = None

    def __str__(self):
        return f"{self.name}: ({self.num_neurons}, {self.act_type})"

    def initialize(self):
        if self.prev_layer is not None:
            input_size = self.prev_layer.num_neurons
            self.weights = np.random.randn(input_size, self.num_neurons) * self.network.init_weights_mult
            self.biases = np.zeros((1, self.num_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        z = np.dot(self.inputs, self.weights) + self.biases
        self.activations = act.VALID_ACT[self.act_type][0](z)
        return self.activations

    def backward(self, output_error):
        # TODO: case of softmax activation (matrix)
        # TODO: add crossentorpy losses

        if self.network.loss == "mean_squared_error":
            dz = output_error * self.act_der(self.activations)
        else:
            raise ValueError("Invalid loss function.")

        output_error_prev = np.dot(dz, self.weights.T)
        self.weights -= self.network.learning_rate * np.dot(self.inputs.T, dz)
        self.biases -= self.network.learning_rate * np.sum(dz, axis=0, keepdims=True)

        return output_error_prev


if __name__ == "__main__":

    model = DenseNetwork(
        layers=[
            DenseLayer(784, act_type="relu"),
            DenseLayer(128, act_type="relu"),
            DenseLayer(64, act_type="relu"),
            DenseLayer(32, act_type="relu"),
            DenseLayer(10, act_type="sigmoid"),
        ],
        init_weights_mult=0.5,
    )

    print(model)
    input_data = np.random.rand(1, 784)
    output = model.forward(input_data)
    print("Output:", np.round(output, 3))

    output_error = np.random.rand(1, 10)
    print("Error:", np.round(output_error, 3))

    model.backward(output_error)
    print(model)
