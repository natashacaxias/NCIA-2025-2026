import numpy as np

# Função de ativação Heaviside
# Retorna 1 se a entrada for >= 0, caso contrário retorna 0
def heaviside(x):
    return 1 if x >= 0 else 0

# Classe Perceptron
class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=1000):
        """
        Inicializa o perceptron.
        
        Parâmetros:
        learning_rate : float
            Taxa de aprendizado do modelo.
        epochs : int
            Número de iterações sobre o conjunto de dados.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None  # Pesos do perceptron (inicializados durante o fit)
        self.bias = None     # Bias do perceptron (inicializado durante o fit)

    def fit(self, X, y):
        """
        Treina o perceptron usando o algoritmo clássico.
        
        Parâmetros:
        X : np.array, shape (n_samples, n_features)
            Conjunto de dados de entrada.
        y : np.array, shape (n_samples,)
            Rótulos de saída (0 ou 1).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Inicializa pesos como zeros
        self.bias = 0                          # Inicializa bias como zero

        # Loop de treinamento
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                # Calcula saída linear (combinação linear de pesos e entrada)
                linear_output = np.dot(x_i, self.weights) + self.bias
                
                # Aplica a função de ativação
                y_predicted = heaviside(linear_output)

                # Calcula a atualização dos pesos
                update = self.learning_rate * (y[idx] - y_predicted)
                
                # Atualiza pesos e bias
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        """
        Faz previsões para novos dados.
        
        Parâmetros:
        X : np.array, shape (n_features,) ou (n_samples, n_features)
            Dados de entrada.
        
        Retorna:
        np.array ou int : Predição(s) do perceptron (0 ou 1)
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return heaviside(linear_output)

# =========================
# Exemplo de uso
# =========================

# Dados de exemplo (4 amostras com 2 características cada)
X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
y = np.array([0, 0, 1, 1])

# Cria e treina o perceptron
perceptron = Perceptron()
perceptron.fit(X, y)

# Faz previsões
predictions = [perceptron.predict(x) for x in X]
print(predictions)  # Saída esperada: [0, 0, 1, 1]
