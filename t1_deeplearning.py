# entrada:        784 neurônios  → um neurônio para cada pixel da imagem 28x28
# hidden Layer 1: 128 neurônios  → inicialização He + ativação ReLU (handout 06 -> he feita p relu)
# hidden Layer 2: 64 neurônios  → inicialização He + ativação ReLU
# saída:           10 neurônios  → inicialização Xavier + ativação Softmax
# bias iniciam em zero -> handout 06 
# custo: Categorical Cross-Entropy
# otimizador: SGD mini-batch 64
# taxa de aprend.: 0,05
# épocas: 30
# passos aplicação: dados -> inicialização -> forward pass -> custo -> backward pass -> 
# atualização pesos -> treinamento -> avaliação

import numpy as np #matrizes
import matplotlib.pyplot as plt #graficos 
import struct

# MNIST tem 60.000 exemplos de treino e 10.000 de teste.
# cada exemplo é uma imagem 28x28 pixels de dígitos (0 a 9).

def carregar_imagens(caminho): #carrega dataset MNIST
    with open(caminho, 'rb') as f:
        # leitura do cabeçalho do arquivo, pixels e transforma em matriz (num_imagens, 784)
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        imagens = np.frombuffer(f.read(), dtype=np.uint8)
        imagens = imagens.reshape(num, rows * cols) #cada linha é uma img, cada pixel uma coluna
    return imagens

def carregar_labels(caminho): #vetor respostas

    with open(caminho, 'rb') as f:
        # cabeçalho do arquivo
        magic, num = struct.unpack('>II', f.read(8))
        # labels
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# carrega os dados de teste e treino (imgs e results)
X_treino = carregar_imagens('mnist/train-images.idx3-ubyte')
y_treino = carregar_labels('mnist/train-labels.idx1-ubyte')
X_teste  = carregar_imagens('mnist/t10k-images.idx3-ubyte')
y_teste  = carregar_labels('mnist/t10k-labels.idx1-ubyte')

# teste imagens dataset
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    imagem = X_treino[i].reshape(28, 28)
    ax.imshow(imagem, cmap='gray')
    ax.set_title(f'Label: {y_treino[i]}')
    ax.axis('off')
plt.suptitle('Primeiras imagens do MNIST')
plt.tight_layout()
plt.show()

# normalização — pixels de [0, 255] para [0, 1]
X_treino = X_treino / 255.0
X_teste  = X_teste  / 255.0

# one-hot encoding dos labels
# transforma o número do dígito em vetor de 10 posições para representar classes de saída (num)
def one_hot(y, num_classes=10):
    n = len(y)
    one_hot_matrix = np.zeros((n, num_classes)) #matriz de zeros
    one_hot_matrix[np.arange(n), y] = 1 
    return one_hot_matrix

y_treino_oh = one_hot(y_treino)
y_teste_oh  = one_hot(y_teste)

def inicializar_parametros(h1=128, h2=64):

    #zerar bias
    #camadas escondidas relu usa he var(w)=2/n  (compensa o fato de relu zerar metade dos neuronios, começa com o dobro)
    #camada de saida softmax usa xavier var(w)=1/n (simétrica)
    #desvio padrão sqrt(var)

    W1 = np.random.randn(784, h1) * np.sqrt(2 / 784)   # He
    b1 = np.zeros((1, h1))

    W2 = np.random.randn(h1, h2) * np.sqrt(2 / h1)     # He
    b2 = np.zeros((1, h2))

    W3 = np.random.randn(h2, 10) * np.sqrt(1 / h2)     # Xavier
    b3 = np.zeros((1, 10))

    return W1, b1, W2, b2, W3, b3

def relu(z):
     # handout 2, eq. 5: ReLU(z) = max(0, z)
     return np.maximum(0, z)

# def softmax(z):
#     exp_z = np.exp(z)
#     return exp_z / np.sum(exp_z)

def softmax(z): #somatorio igual a 1.0
    # handout 2, eq. 6: softmax(z) = e^z / soma(e^z)
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# forward_pass z = W*x+b

def forward_pass(X, W1, b1, W2, b2, W3, b3):
    # @ p/ multiplicação de matrizes  + bias
    # camada 1 - ReLU
    z1 = X @ W1 + b1
    a1 = relu(z1) #ativação camada 1

    # camada 2 — ReLU
    z2 = a1 @ W2 + b2
    a2 = relu(z2) #ativição camada 2

    # camada saída — Softmax
    z3 = a2 @ W3 + b3
    a3 = softmax(z3) #predição/ativação camada 3 

    return z1, a1, z2, a2, z3, a3

# inicializa os parâmetros com seed 42
np.random.seed(42) # seed fixo pra comparar resultados
W1, b1, W2, b2, W3, b3 = inicializar_parametros()

# # batch pequeno de 5 imagens para testar
# X_batch = X_treino[:5]
# y_batch = y_treino_oh[:5]

#  # forward pass
# z1, a1, z2, a2, z3, a3 = forward_pass(X_batch, W1, b1, W2, b2, W3, b3)

# softmax soma 1 por linha
# print("\nSoma das probabilidades:")
# print(a3.sum(axis=1))              

# teste se a predição faz sentido - sem treinamento
# print("\nPredições (a3):")
# print(a3)                          

# verifica labels reais
# print("\nLabels:")
# print(y_treino[:5])     

def calcular_custo(y_pred, y_true):
    N = y_true.shape[0]
         #clipping 
         #y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    custo = -np.sum(y_true * np.log(y_pred)) / N
    return custo         

# custo = calcular_custo(a3, y_batch)
# print(f'Custo teste: {custo:.4f}')
# esperado: próximo de 2.302 chutando perto de 10% p cada classe

#backward pass -> calcula delta, gtadiente pesos, gradiente bias
# handout 3, eq. 27: erro na camada de saída (fica delta[L] = ^y - y porque usa CCE com softmax eq. 12)
# handout 3, eq. 29: propagação do erro - distribuição pesos
# # delta[l] = Wtransposta * erro anterior * (produto notarial) derivada da relu
# handout 3, eq. 30,31: gradiente dos pesos
# handout 3, eq. 32: gradiente dos bias 
# erro no bias é igual ao erro da ativação do neurônio dL/db = delta 

def relu_derivada(z):
    # 1 onde z era positivo, 0 onde era negativo
    return (z >= 0).astype(float)

def backward_pass(X, y_true, z1, a1, z2, a2, a3, W2, W3):
    N = X.shape[0]  # tamanho do batch

    # camada de saída
    delta3 = a3 - y_true
    dW3 = (a2.T @ delta3) / N
    db3 = np.sum(delta3, axis=0, keepdims=True) / N

    # camada 2
    delta2 = (delta3 @ W3.T) * relu_derivada(z2)
    dW2 = (a1.T @ delta2) / N
    db2 = np.sum(delta2, axis=0, keepdims=True) / N

    # camada 1
    delta1 = (delta2 @ W2.T) * relu_derivada(z1)
    dW1 = (X.T @ delta1) / N
    db1 = np.sum(delta1, axis=0, keepdims=True) / N

    return dW1, db1, dW2, db2, dW3, db3

def atualizar_pesos(W1, b1, W2, b2, W3, b3,
                    dW1, db1, dW2, db2, dW3, db3,
                    learning_rate=0.05):

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    W3 = W3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3

    return W1, b1, W2, b2, W3, b3

# # interação completa p/ teste
# X_batch = X_treino[:64]
# y_batch = y_treino_oh[:64]
# z1, a1, z2, a2, z3, a3 = forward_pass(X_batch, W1, b1, W2, b2, W3, b3)
# custo_antes = calcular_custo(a3, y_batch)
# print(f'Custo antes: {custo_antes:.4f}')
# dW1, db1, dW2, db2, dW3, db3 = backward_pass(X_batch, y_batch, z1, a1, z2, a2, a3, W2, W3)
# W1, b1, W2, b2, W3, b3 = atualizar_pesos(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3)
# z1, a1, z2, a2, z3, a3 = forward_pass(X_batch, W1, b1, W2, b2, W3, b3)
# custo_depois = calcular_custo(a3, y_batch)
# print(f'Custo depois: {custo_depois:.4f}')

# if custo_depois < custo_antes:
#     print('Custo diminuiu')
# else:
#     print('Custo não diminuiu!')

############### TREINAMENTO 

# handout 6, seção 4.1: SGD - gradiente descente estocastico mini-batch de 64, 30 épocas, alpha = 0,05

def treinar(X_treino, y_treino_oh, X_teste, y_teste_oh,
            W1, b1, W2, b2, W3, b3,
            epochs=30, batch_size=64, learning_rate=0.05):

    # histórico para os gráficos
    historico_custo_treino = []
    historico_custo_teste  = []
    historico_acuracia     = []

    for epoca in range(epochs):

        # embaralha os dados a cada época
        indices = np.random.permutation(X_treino.shape[0])
        X_embaralhado = X_treino[indices]
        y_embaralhado = y_treino_oh[indices]

        custo_epoca = 0
        num_batches = 0

        # loop dos mini-batches
        for i in range(0, X_treino.shape[0], batch_size):
            # pega o batch atual
            X_batch = X_embaralhado[i:i+batch_size]
            y_batch = y_embaralhado[i:i+batch_size]

            # forward pass
            z1, a1, z2, a2, z3, a3 = forward_pass(X_batch, W1, b1, W2, b2, W3, b3)

            # custo
            custo = calcular_custo(a3, y_batch)
            custo_epoca += custo
            num_batches += 1

            # backward pass
            dW1, db1, dW2, db2, dW3, db3 = backward_pass(
                X_batch, y_batch, z1, a1, z2, a2, a3, W2, W3)

            # atualiza os pesos
            W1, b1, W2, b2, W3, b3 = atualizar_pesos(
                W1, b1, W2, b2, W3, b3,
                dW1, db1, dW2, db2, dW3, db3,
                learning_rate)

        # custo médio da época
        custo_medio = custo_epoca / num_batches
        historico_custo_treino.append(custo_medio)

        # avalia no dataset de teste
        _, _, _, _, _, a3_teste = forward_pass(X_teste, W1, b1, W2, b2, W3, b3)
        custo_teste = calcular_custo(a3_teste, y_teste_oh)
        historico_custo_teste.append(custo_teste)

        # acurácia no teste
        predicoes = np.argmax(a3_teste, axis=1)
        acuracia  = np.mean(predicoes == y_teste)
        historico_acuracia.append(acuracia)

        # imprime progresso
        print(f'Época {epoca+1:2d}/{epochs} | '
              f'Custo treino: {custo_medio:.4f} | '
              f'Custo teste: {custo_teste:.4f} | '
              f'Acurácia: {acuracia*100:.2f}%')

    return W1, b1, W2, b2, W3, b3, historico_custo_treino, historico_custo_teste, historico_acuracia

W1, b1, W2, b2, W3, b3, custo_treino, custo_teste, acuracia = treinar(
    X_treino, y_treino_oh,
    X_teste, y_teste_oh,
    W1, b1, W2, b2, W3, b3
)

def avaliar_modelo(X_teste, y_teste, y_teste_oh, W1, b1, W2, b2, W3, b3):

    # forward pass no teste completo
    _, _, _, _, _, a3_teste = forward_pass(X_teste, W1, b1, W2, b2, W3, b3)

    # predições finais
    y_pred = np.argmax(a3_teste, axis=1)  # classe com maior probabilidade

    N  = len(y_teste)       # total de exemplos
    K  = 10                 # número de classes

    #matriz de confusao
    matriz_confusao = np.zeros((K, K), dtype=int)
    for real, pred in zip(y_teste, y_pred):
        matriz_confusao[real][pred] += 1

    #acuracia geral
    acuracia = np.sum(y_pred == y_teste) / N

    #precisão
    precisao = np.zeros(K)
    #recall
    recall   = np.zeros(K)
    #f1 score por classe
    f1       = np.zeros(K)

    for k in range(K):
        TP = matriz_confusao[k, k] #verdadeiro pos
        FP = np.sum(matriz_confusao[:, k]) - TP  #falso pos
        FN = np.sum(matriz_confusao[k, :]) - TP  #falso neg

        precisao[k] = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall[k]   = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1[k]       = 2 * (precisao[k] * recall[k]) / (precisao[k] + recall[k]) \
                      if (precisao[k] + recall[k]) > 0 else 0

    # handout 04, eq. 7: average class accuracy (ACA)
    aca = np.mean(recall)

    #perplexidade
    # handout 04, eq. 25: exp(CCE)
    custo_teste = calcular_custo(a3_teste, y_teste_oh)
    perplexidade = np.exp(custo_teste)

    #resultados:
    print('\nAVALIAÇÃO DO MODELO:')

    print(f'\nAcurácia geral:          {acuracia*100:.2f}%')
    print(f'Average Class Accuracy:  {aca*100:.2f}%')
    print(f'Perplexidade:            {perplexidade:.4f}')

    print('\nMétrica por classe:')
    print(f'{"Classe":>8} {"Precisão":>10} {"Recall":>10} {"F1":>10}')
    print('-' * 42)
    for k in range(K):
        print(f'{k:>8} {precisao[k]*100:>9.2f}% {recall[k]*100:>9.2f}% {f1[k]*100:>9.2f}%')

    print('-' * 42)
    print(f'{"Média":>8} {np.mean(precisao)*100:>9.2f}% '
          f'{np.mean(recall)*100:>9.2f}% '
          f'{np.mean(f1)*100:>9.2f}%')

    return matriz_confusao, acuracia, aca, precisao, recall, f1, perplexidade

# chama a avaliação
matriz_confusao, acuracia_final, aca, precisao, recall, f1, perplexidade = avaliar_modelo(
    X_teste, y_teste, y_teste_oh, W1, b1, W2, b2, W3, b3)

def plotar_graficos(historico_custo_treino, historico_custo_teste,
                    historico_acuracia, matriz_confusao,
                    precisao, recall, f1):

    epocas = range(1, len(historico_custo_treino) + 1)

    #gráficos 
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig1.suptitle('Curvas de Aprendizado', fontsize=14)

    # gráfico 1 — custo treino vs teste por época
    ax1.plot(epocas, historico_custo_treino, label='Treino', color='blue')
    ax1.plot(epocas, historico_custo_teste,  label='Teste',  color='red')
    ax1.set_title('Custo por Época')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Custo (CCE)')
    ax1.legend()
    ax1.grid(True)

    # gráfico 2 — acurácia por época
    ax2.plot(epocas, [a*100 for a in historico_acuracia], color='green')
    ax2.set_title('Acurácia por Época')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Acurácia (%)')
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

    fig2, (ax3, ax4, ax5) = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle('Avaliação Final do Modelo', fontsize=14)

    # gráfico 3 — matriz de confusão
    im = ax3.imshow(matriz_confusao, cmap='Blues')
    ax3.set_title('Matriz de Confusão')
    ax3.set_xlabel('Predição')
    ax3.set_ylabel('Real')
    ax3.set_xticks(range(10))
    ax3.set_yticks(range(10))
    plt.colorbar(im, ax=ax3)

    # adiciona os valores dentro da matriz
    for i in range(10):
        for j in range(10):
            cor = 'white' if matriz_confusao[i,j] > 500 else 'black'
            ax3.text(j, i, str(matriz_confusao[i,j]),
                    ha='center', va='center',
                    color=cor, fontsize=7)

    # gráfico 4 — F1 por classe
    classes = [str(i) for i in range(10)]
    ax4.bar(classes, f1*100, color='purple')
    ax4.set_title('F1-Score por Classe')
    ax4.set_xlabel('Classe (Dígito)')
    ax4.set_ylabel('F1-Score (%)')
    ax4.set_ylim([95, 100])
    ax4.grid(True, axis='y')

    # adiciona valores nas barras
    for i, v in enumerate(f1*100):
        ax4.text(i, v + 0.05, f'{v:.1f}%',
                ha='center', va='bottom', fontsize=8)

    # gráfico 5 — precisão e recall por classe
    x = range(10)
    largura = 0.35
    ax5.bar([i - largura/2 for i in x], precisao*100,
            largura, label='Precisão', color='blue', alpha=0.7)
    ax5.bar([i + largura/2 for i in x], recall*100,
            largura, label='Recall', color='red', alpha=0.7)
    ax5.set_title('Precisão e Recall por Classe')
    ax5.set_xlabel('Classe (Dígito)')
    ax5.set_ylabel('%')
    ax5.set_xticks(list(x))
    ax5.set_xticklabels(classes)
    ax5.set_ylim([95, 100])
    ax5.legend()
    ax5.grid(True, axis='y')

    plt.tight_layout()
    plt.show()

# chama os gráficos
plotar_graficos(custo_treino, custo_teste, acuracia,  
                matriz_confusao, precisao, recall, f1)