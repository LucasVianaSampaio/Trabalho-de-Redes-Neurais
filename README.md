## Questão 1: Regressão Linear
Implemente um modelo de regressão linear. Para isso, utilize um conjunto de dados sintético gerado com a equação:

y = 3x + 5 + ε(1)

1-Gere um conjunto de dados com pelo menos 100 pontos.

```python
np.random.seed(42) 
x = np.random.uniform(-10, 10, 100)
epsilon = np.random.normal(0, 2, 100)
y = 3*x + 5 + epsilon
```

2-Divida os dados em treino (80%) e teste (20%).

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

3-Implemente modelos de regressão linear empregando a solução dos mínimos quadrados:

- A solução de mínimos quadrados (pseudo-inversa):

```python
X_train = np.vstack([x_train, np.ones_like(x_train)]).T # Criando a matriz de design com uma coluna de 1s para o bias
w = np.linalg.pinv(X_train) @ y_train  # Calculando os coeficientes usando a pseudo-inversa
```

- Apresentando a solução de mínimos quadrados

```python
print(f"Solução de Mínimos Quadrados:")
print(f"Coeficiente angular (w1): {w[0]:.4f}")
print(f"Intercepto (w0): {w[1]:.4f}")
```

- A predição nos dados de teste:

```python
X_test = np.vstack([x_test, np.ones_like(x_test)]).T  # Criando a matriz de design para os dados de teste
y_pred = X_test @ w  # Fazendo previsões com o modelo de mínimos quadrados
```
- Cálculo do erro quadrático médio(MSE) para mínimos quadrados:

```python
mse = np.mean((y_test - y_pred)**2)
print(f'MSE (Mínimos Quadrados): {mse:.4f}')
```

4-Implementação do modelo de regressão linear empregando uma rede neural com uma camada treinada via gradiente descendente utilizando MSE-
Loss (Erro Quadrático Médio) e otimizador SGD:

- Convertendo os dados de entrada para tensores do PyTorch:

```python
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).view(-1, 1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
```

- Definição da rede neural para regressão linear:

```python
class LinearRegressionNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # Camada linear com 1 entrada e 1 saída
    
    def forward(self, x):
        return self.linear(x)  # Aplicação da camada linear
```

- Inicializando o modelo, função de perda e otimizador:

```python
model = LinearRegressionNN()
criterion = nn.MSELoss()  # Função de perda: Erro Quadrático Médio (MSE)
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Otimizador: Gradiente Descendente Estocástico (SGD)
```

- Treinamento da rede neural

```python
num_epochs = 1000  # Número de épocas de treinamento
for epoch in range(num_epochs):
    outputs = model(x_train_tensor)  # Passagem para frente (forward pass)
    loss = criterion(outputs, y_train_tensor)  # Cálculo da perda
    optimizer.zero_grad()  # Zerando os gradientes acumulados
    loss.backward()  # Retropropagação para calcular gradientes
    optimizer.step()  # Atualização dos pesos
    
    if (epoch+1) % 100 == 0:  # Exibindo a perda a cada 100 épocas
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```

- Apresentando a solução da rede neural

```python
print(f"\nSolução da Rede Neural:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.data.numpy()}")
```

- Avaliação do modelo nos dados de teste:

```python
model.eval()  # Mudando para modo de avaliação
with torch.no_grad():  # Desativando o cálculo de gradientes para eficiência
    y_pred_nn = model(x_test_tensor).numpy()  # Fazendo previsões
```

- Cálculo do MSE para a rede neural:

```python
mse_nn = np.mean((y_test - y_pred_nn.flatten())**2)
print(f'MSE (Rede Neural): {mse_nn:.4f}')
```

- Visualização dos resultados:

```python
plt.scatter(x_train, y_train, label='Treino', color='blue', alpha=0.6)  # Pontos de treino
plt.scatter(x_test, y_test, label='Teste', color='red', alpha=0.6)  # Pontos de teste
plt.plot(x_test, y_pred, label=f'Mínimos Quadrados (MSE: {mse:.2f})', color='black', linewidth=2)  # Linha da regressão por mínimos quadrados
plt.plot(x_test, y_pred_nn, label=f'Rede Neural (MSE: {mse_nn:.2f})', color='green', linewidth=2, linestyle='--')  # Linha da regressão pela rede neural
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparação entre Mínimos Quadrados e Rede Neural')
plt.show()
```

[por a imagem dos resultados]

O gráfico mostra a comparação entre os modelos de regressão linear obtidos com a solução de mínimos quadrados e a rede neural.