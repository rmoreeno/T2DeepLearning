# =============================================================================
# lstm 2 camadas — dataset penn tree bank (ptb)
# sgd com decaimento de learning rate + dropout manual + early stopping
# =============================================================================
# materiais p/ consultas:
#   https://github.com/CaptainE/RNN-LSTM-in-numpy/blob/master/RNN_LSTM_from_scratch.ipynb
#   https://christinakouridi.github.io/posts/implement-lstm/
#   https://nimasarang.com/blog/2024-06-15-lstm-from-scratch/
# =============================================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
import time

device = torch.device('cpu')

# hiperparâmetros

EMBED_DIM   = 128   # dimensão do vetor de embedding: cada palavra → vetor de 128 números
HIDDEN_SIZE = 128   # dimensão do estado oculto da lstm
SEQ_LEN     = 35    # processa 35 tokens por vez
BATCH_SIZE  = 32    # número de sequências processadas em paralelo
LR          = 2.0   # learning rate inicial do sgd
LR_DECAY    = 0.8   # fator multiplicativo: a partir de lr_start, lr ← lr × 0.8 por época
LR_START    = 7     # época a partir da qual começa o decaimento do lr
GRAD_CLIP   = 5.0   # norma máxima dos gradientes (limiar do gradient clipping)
DROPOUT_P   = 0.35  # 35% dos neurônios são desligados aleatoriamente a cada passo
EPOCHS      = 30    # limite máximo de épocas mas aqui usa early stopping
PATIENCE    = 4     # quantas épocas sem melhoria na validação antes do early stopping parar

# seed fixa para reprodutibilidade
torch.manual_seed(42)
np.random.seed(42)

# 1. carregamento do dataset penn treebank (ptb)
#   tokenização por palavra: cada palavra é um token.
#
#  o ptb (penn treebank) é um corpus de notícias em inglês pré-processado:
#   palavras raras → substituídas por <unk> (unknown) para limitar o vocabulário
#   números (1, 3.5, 1992) → substituídos por N para reduzir variação
#   fim de cada linha → marcado com <eos>
#   o vocabulário tem ~10.000 palavras

def carregar_ptb(caminho):
    # lê o arquivo como texto único, substitui cada quebra de linha por " <eos> "
    # e depois separa em lista de palavras pelo espaço
    with open(caminho, 'r', encoding='utf-8') as f:
        return f.read().replace('\n', ' <eos> ').split()

palavras_treino = carregar_ptb('penntreebank/ptbdataset/ptb.train.txt')
palavras_valid  = carregar_ptb('penntreebank/ptbdataset/ptb.valid.txt')
palavras_teste  = carregar_ptb('penntreebank/ptbdataset/ptb.test.txt')

# vocabulário: lista ordenada de todas as palavras do conjunto de treino
vocab      = sorted(set(palavras_treino))
vocab_size = len(vocab)
print(f'vocabulário: {vocab_size} palavras únicas')
print(f'tokens — treino: {len(palavras_treino):,} | valid: {len(palavras_valid):,} | teste: {len(palavras_teste):,}')

# dicionários de conversão bidirecional: palavra ↔ índice inteiro
# a rede não processa texto — só números. "the" pode ser o índice 8432.
# ex: palavra2idx = {"a": 0, "aer": 1, ..., "the": 8432, ...}
palavra2idx = {p: i for i, p in enumerate(vocab)}
idx2palavra = vocab                               # lista indexada diretamente
unk_idx     = palavra2idx.get('<unk>', 0)         # índice do token desconhecido

def tokens_para_ids(tokens):
    # converte lista de strings em array numpy de inteiros
    # palavras ausentes do vocabulário de treino → unk_idx
    return np.array([palavra2idx.get(t, unk_idx) for t in tokens], dtype=np.int64)

ids_treino = tokens_para_ids(palavras_treino)
ids_valid  = tokens_para_ids(palavras_valid)
ids_teste  = tokens_para_ids(palavras_teste)

# 2. inicialização dos parâmetros do modelo

def t(arr):
    # função auxiliar: numpy array → tensor pytorch com gradiente habilitado
    return torch.tensor(arr, dtype=torch.float32, device=device, requires_grad=True)

def inicializar_parametros():
    D = EMBED_DIM
    H = HIDDEN_SIZE
    V = vocab_size

    # --- tabela de embedding: (vocab_size × embed_dim) ---
    E  = t(np.random.uniform(-0.1, 0.1, (V, D)))

    # --- camada lstm 1 com gates fundidos ---
    # em vez de 4 matrizes separadas (Wf, Wi, Wg, Wo), usamos duas matrizes grandes:
    #   W1 (D, 4H): pesos que recebem o embedding x_t
    #   U1 (H, 4H): pesos que recebem o estado oculto h_{t-1}
    # os 4 gates são calculados de uma vez: gates = x @ W1 + h @ U1 + b1
    # e então fatiados: gates[:, 0:H]=forget, H:2H=input, 2H:3H=candidato, 3H:4H=output
    # vantagem: 2 matmuls por passo em vez de 8 (4x menos operações)
    sc1 = np.sqrt(1.0 / (D + H))
    W1  = t(np.random.randn(D, 4 * H) * sc1)
    U1  = t(np.random.randn(H, 4 * H) * sc1)
    # bias: forget gate inicia em 1 (mantém memória fluindo no início), demais em 0
    b1  = t(np.concatenate([np.ones((1, H)), np.zeros((1, 3 * H))], axis=1))

    # --- camada lstm 2 com gates fundidos ---
    sc2 = np.sqrt(1.0 / (2 * H))
    W2  = t(np.random.randn(H, 4 * H) * sc2)
    U2  = t(np.random.randn(H, 4 * H) * sc2)
    b2  = t(np.concatenate([np.ones((1, H)), np.zeros((1, 3 * H))], axis=1))

    # --- bias de saída (weight tying: usa E como matriz de projeção) ---
    by  = t(np.zeros((1, V)))

    return dict(E=E, W1=W1, U1=U1, b1=b1, W2=W2, U2=U2, b2=b2, by=by)

params = inicializar_parametros()
n_params = sum(p.numel() for p in params.values())
print(f'total de parâmetros: {n_params:,}')

# 3. dropout manual com máscara de bernoulli

def dropout(x, p=DROPOUT_P, training=True):
    # durante avaliação ou se p=0, não aplica nada
    if not training or p == 0.0:
        return x
    # gera máscara binária de bernoulli: 1 = sobrevive, 0 = zerado
    mascara = (torch.rand_like(x) > p).float()
    # multiplica pela máscara e escala os sobreviventes (inverted dropout)
    return x * mascara / (1.0 - p)

# 4. célula lstm 

def lstm_cell(x, h, c, W, U, b):
    H = HIDDEN_SIZE
    # calcula os 4 gates de uma vez: 2 matmuls em vez de 8
    # gates[B, 0:H]=forget | H:2H=input | 2H:3H=candidato | 3H:4H=output
    gates = x @ W + h @ U + b          # (B, 4H)

    f = torch.sigmoid(gates[:, 0*H : 1*H])   # forget gate
    i = torch.sigmoid(gates[:, 1*H : 2*H])   # input gate
    g = torch.tanh   (gates[:, 2*H : 3*H])   # candidato
    o = torch.sigmoid(gates[:, 3*H : 4*H])   # output gate

    c_novo = f * c + i * g
    h_novo = o * torch.tanh(c_novo)

    return h_novo, c_novo

# 5. forward pass da lstm de 2 camadas com dropout

def lstm_2camadas_forward(input_ids, h1, c1, h2, c2, params, training=True):
    E        = params['E']
    W1, U1, b1 = params['W1'], params['U1'], params['b1']
    W2, U2, b2 = params['W2'], params['U2'], params['b2']

    T = input_ids.shape[1]
    saidas = []

    for passo in range(T):
        # busca o embedding da palavra atual: indexação direta na linha da matriz E
        x = E[input_ids[:, passo]]              # (B, embed_dim)

        # dropout 1: regulariza o embedding de entrada antes da lstm1
        x = dropout(x, training=training)

        # processa pela primeira camada lstm (gates fundidos)
        h1, c1 = lstm_cell(x, h1, c1, W1, U1, b1)

        # dropout 2: regulariza a comunicação entre lstm1 e lstm2
        h1_drop = dropout(h1, training=training)

        # processa pela segunda camada lstm
        h2, c2 = lstm_cell(h1_drop, h2, c2, W2, U2, b2)

        # dropout 3: regulariza a saída antes da projeção para o vocabulário
        h2_drop = dropout(h2, training=training)
        saidas.append(h2_drop)

    return saidas, h1, c1, h2, c2

# 6. forward completo: lstm → projeção → cross-entropy loss

def forward_completo(inp_np, tgt_np, h1, c1, h2, c2, params, training=True):
    inp = torch.tensor(inp_np, dtype=torch.long, device=device)  # (B, T) — índices de entrada
    tgt = torch.tensor(tgt_np, dtype=torch.long, device=device)  # (B, T) — índices alvo

    # roda a lstm de 2 camadas por T passos de tempo
    saidas, h1_n, c1_n, h2_n, c2_n = lstm_2camadas_forward(
        inp, h1, c1, h2, c2, params, training=training)

    # empilha saídas de todos os passos: lista de T tensores (B,H) → (T*B, H)
    H_all = torch.stack(saidas, dim=0).reshape(-1, HIDDEN_SIZE)

    # projeção via weight tying: logits = H @ E^T + by  →  (T*B, vocab_size)
    logits = H_all @ params['E'].T + params['by']

    # organiza os alvos para corresponder às linhas de H_all
    targets_flat = tgt.T.reshape(-1)

    # log_softmax: log das probabilidades normalizadas
    log_probs = torch.log_softmax(logits, dim=1)
    loss      = -log_probs[
        torch.arange(len(targets_flat), device=device), targets_flat].mean()

    return loss, h1_n, c1_n, h2_n, c2_n

# 7. gradient clipping — controle da explosão de gradientes

def clip_gradientes(param_list, max_norm=GRAD_CLIP):
    # calcula a norma l2 global: sqrt da soma dos quadrados de todos os gradientes
    norma = torch.sqrt(sum(p.grad.data.norm() ** 2
                           for p in param_list if p.grad is not None))
    if norma > max_norm:
        fator = max_norm / norma  # fator de escala para não ultrapassar max_norm
        for p in param_list:
            if p.grad is not None:
                p.grad.data.mul_(fator)  # escala o gradiente in-place

# 8. cálculo da perplexidade — métrica padrão de modelos de linguagem

def calcular_perplexidade(ids, params):
    B, T  = BATCH_SIZE, SEQ_LEN
    n     = (len(ids) // B) * B   # descarta tokens extras para dividir em B sequências iguais
    dados = ids[:n].reshape(B, -1)
    n_passos = (dados.shape[1] - 1) // T  # número de janelas de tamanho T

    total = 0.0
    # estado inicial zerado — lstm começa sem memória
    h1 = torch.zeros(B, HIDDEN_SIZE, device=device)
    c1 = torch.zeros(B, HIDDEN_SIZE, device=device)
    h2 = torch.zeros(B, HIDDEN_SIZE, device=device)
    c2 = torch.zeros(B, HIDDEN_SIZE, device=device)

    with torch.no_grad():   # desativa o autograd para economizar memória e tempo
        for p in range(n_passos):
            inp = dados[:, p * T:(p + 1) * T]
            tgt = dados[:, p * T + 1:(p + 1) * T + 1]
            # training=False: dropout desativado — usamos a rede completa na avaliação
            loss, h1, c1, h2, c2 = forward_completo(
                inp, tgt, h1, c1, h2, c2, params, training=False)
            total += loss.item()

    # perplexidade = exp(cross-entropy média sobre todas as janelas)
    return np.exp(total / n_passos)

# 9. loop de treinamento com sgd + decaimento de lr + early stopping

def treinar(ids_treino, ids_valid, params):
    B, T       = BATCH_SIZE, SEQ_LEN
    lr         = LR
    param_list = list(params.values())

    # reorganiza corpus em matri
    n        = (len(ids_treino) // B) * B
    dados    = ids_treino[:n].reshape(B, -1)
    n_passos = (dados.shape[1] - 1) // T   # número de janelas de SEQ_LEN no corpus

    hist_custo  = []   # custo de treino por época
    hist_ppl_t  = []   # perplexidade de treino por época
    hist_ppl_v  = []   # perplexidade de validação por época
    epoca_parada = None  # registra onde ocorreu o early stopping

    # variáveis do early stopping
    melhor_ppl_v    = float('inf')
    espera          = 0
    melhores_params = None

    for epoca in range(1, EPOCHS + 1):

        # decaimento do learning rate após LR_START épocas
        if epoca > LR_START:
            lr *= LR_DECAY

        total_custo = 0.0
        inicio_epoca = time.time()

        # reinicia o estado oculto no início de cada época
        h1 = torch.zeros(B, HIDDEN_SIZE, device=device)
        c1 = torch.zeros(B, HIDDEN_SIZE, device=device)
        h2 = torch.zeros(B, HIDDEN_SIZE, device=device)
        c2 = torch.zeros(B, HIDDEN_SIZE, device=device)

        for p in range(n_passos):
            # inp: palavras de entrada  [w_t, ..., w_{t+T-1}]
            # tgt: palavras alvo        [w_{t+1}, ..., w_{t+T}]
            # cada posição de inp deve prever a posição correspondente de tgt
            inp = dados[:, p * T:(p + 1) * T]
            tgt = dados[:, p * T + 1:(p + 1) * T + 1]

            # zera gradientes do passo anterior
            for param in param_list:
                if param.grad is not None:
                    param.grad.zero_()

            # detach(): corta o grafo de computação entre janelas de tbptt
            # os estados h e c são passados adiante (memória preservada entre janelas)
            # mas os gradientes não retornam além desta janela (truncated bptt)
            h1 = h1.detach(); c1 = c1.detach()
            h2 = h2.detach(); c2 = c2.detach()

            # forward com dropout ativo
            loss, h1, c1, h2, c2 = forward_completo(
                inp, tgt, h1, c1, h2, c2, params, training=True)

            # backward: pytorch computa ∂loss/∂param para todos os parâmetros
            # percorre o grafo de computação de trás para frente (regra da cadeia)
            loss.backward()

            # gradient clipping: evita explosão de gradientes
            clip_gradientes(param_list)

            # atualização sgd manual: w ← w - lr × ∂loss/∂w
            with torch.no_grad():
                for param in param_list:
                    if param.grad is not None:
                        param.data -= lr * param.grad.data

            total_custo += loss.item()

        # métricas da época
        custo_medio = total_custo / n_passos
        ppl_treino  = np.exp(custo_medio)
        ppl_valid   = calcular_perplexidade(ids_valid, params)

        hist_custo.append(custo_medio)
        hist_ppl_t.append(ppl_treino)
        hist_ppl_v.append(ppl_valid)

        tempo = time.time() - inicio_epoca
        print(f'época {epoca:2d}/{EPOCHS} | {tempo:.0f}s | lr={lr:.5f} | '
              f'custo={custo_medio:.4f} | '
              f'ppl treino={ppl_treino:.1f} | '
              f'ppl valid={ppl_valid:.1f}')

        # lógica do early stopping
        if ppl_valid < melhor_ppl_v:
            # validação melhorou — salva os melhores pesos e reinicia o contador
            melhor_ppl_v    = ppl_valid
            espera          = 0
            melhores_params = {k: v.detach().clone() for k, v in params.items()}
        else:
            espera += 1
            if espera >= PATIENCE:
                epoca_parada = epoca
                print(f'\nearly stopping na época {epoca}:')
                print(f'  validação não melhorou por {PATIENCE} épocas consecutivas.')
                # restaura os pesos da melhor época encontrada
                for k in params:
                    params[k].data.copy_(melhores_params[k])
                print(f'  pesos restaurados (melhor ppl valid: {melhor_ppl_v:.1f})')
                break

    return params, hist_custo, hist_ppl_t, hist_ppl_v, epoca_parada

# executa o treinamento e captura o histórico e a época de parada
params, hist_custo, hist_ppl_t, hist_ppl_v, epoca_parada = treinar(
    ids_treino, ids_valid, params)

# 10. avaliação final no conjunto de teste

# o conjunto de teste é usado uma vez:

ppl_teste = calcular_perplexidade(ids_teste, params)
print(f'\nperplexidade final — teste: {ppl_teste:.2f}')

# 11. gráficos de avaliação

def plotar_graficos(hist_custo, hist_ppl_t, hist_ppl_v, ppl_teste, epoca_parada):
    epocas = list(range(1, len(hist_custo) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'lstm 2 camadas + dropout (p={DROPOUT_P}) — penn treebank | sgd + early stopping',
        fontsize=12)

    # --- painel esquerdo: curva de custo de treino ---
    axes[0].plot(epocas, hist_custo, 'bo-', linewidth=2, markersize=5)
    axes[0].set_title('custo de treinamento por época')
    axes[0].set_xlabel('época')
    axes[0].set_ylabel('cross-entropy (média)')
    axes[0].grid(True, alpha=0.3)

    # marca onde o early stopping ocorreu, se houver
    if epoca_parada is not None:
        axes[0].axvline(epoca_parada, color='purple', linestyle=':', linewidth=1.5,
                        label=f'early stopping (época {epoca_parada})')
        axes[0].legend()

    # --- painel direito: perplexidade treino × validação × teste ---
    axes[1].plot(epocas, hist_ppl_t, 'bo-', linewidth=2, markersize=5, label='treino')
    axes[1].plot(epocas, hist_ppl_v, 'rs-', linewidth=2, markersize=5, label='validação')
    axes[1].axhline(ppl_teste, color='green', linestyle='--', linewidth=2,
                    label=f'teste final: {ppl_teste:.1f}')

    if epoca_parada is not None:
        axes[1].axvline(epoca_parada, color='purple', linestyle=':', linewidth=1.5,
                        label=f'early stopping (época {epoca_parada})')

    axes[1].set_title('perplexidade por época (treino vs validação)')
    axes[1].set_xlabel('época')
    axes[1].set_ylabel('perplexidade (menor = melhor)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('resultados_lstm_2camadas.png', dpi=150, bbox_inches='tight')
    print('gráfico salvo em resultados_lstm_2camadas.png')
    plt.close()

plotar_graficos(hist_custo, hist_ppl_t, hist_ppl_v, ppl_teste, epoca_parada)

# 12. geração de texto — análise qualitativa do modelo
#
# algoritmo de geração (amostragem autoregressiva):
#   1. inicializa a lstm com estado zerado e alimenta com a palavra-semente
#   2. obtém logits (pontuações) para cada palavra do vocabulário
#   3. converte em probabilidades via softmax
#   4. amostra a próxima palavra das probabilidades 
#   5. a palavra amostrada vira a entrada do próximo passo → repete

def gerar_texto(params, palavra_inicial, n_palavras=50):
    # estado inicial zerado — começa uma nova "frase" sem contexto anterior
    h1 = torch.zeros(1, HIDDEN_SIZE, device=device)
    c1 = torch.zeros(1, HIDDEN_SIZE, device=device)
    h2 = torch.zeros(1, HIDDEN_SIZE, device=device)
    c2 = torch.zeros(1, HIDDEN_SIZE, device=device)

    idx    = palavra2idx.get(palavra_inicial, unk_idx)
    gerado = [palavra_inicial]

    with torch.no_grad():
        for _ in range(n_palavras):
            inp = torch.tensor([[idx]], dtype=torch.long, device=device)  # (1, 1)

            # roda a lstm sem dropout (avaliação pura)
            saidas, h1, c1, h2, c2 = lstm_2camadas_forward(
                inp, h1, c1, h2, c2, params, training=False)

            # weight tying: usa E para converter h_t em logits sobre o vocabulário
            logits = saidas[-1] @ params['E'].T + params['by']  # (1, vocab_size)

            # converte em probabilidades e garante soma exata = 1 (normaliza ruído numérico)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            probs = np.abs(probs) / np.abs(probs).sum()

            # amostra a próxima palavra — multinomial: cada palavra com sua probabilidade
            idx   = np.random.choice(vocab_size, p=probs)
            gerado.append(idx2palavra[idx])

    return ' '.join(gerado)

print('\n─── geração de texto ───')
print(gerar_texto(params, 'the', n_palavras=50))