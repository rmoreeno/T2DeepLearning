# =============================================================================
# modelo de linguagem com lstm — dataset penn treebank (ptb)
# otimizador: sgd com decaimento de learning rate
# =============================================================================
# referência: https://github.com/CaptainE/RNN-LSTM-in-numpy/blob/master/RNN_LSTM_from_scratch.ipynb
# =============================================================================

import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cpu')

# hiperparâmetros

EMBED_DIM      = 128  # dimensão do vetor de embedding de cada palavra
HIDDEN_SIZE    = 128  # dimensão do estado oculto da lstm (igual ao embed para weight tying)
SEQ_LEN        = 35   # comprimento da janela de tbptt
BATCH_SIZE     = 32   # sequências processadas em paralelo
EPOCHS         = 15   # número de épocas de treinamento
LR             = 1.0  # learning rate inicial do sgd
LR_DECAY       = 0.8  # fator multiplicativo do lr após lr_decay_start
LR_DECAY_START = 8    # época a partir da qual o lr começa a decair
GRAD_CLIP      = 5.0  # norma máxima dos gradientes (gradient clipping)

torch.manual_seed(42)
np.random.seed(42)

# 1. carregamento do dataset penn treebank (ptb)
#   o ptb é um corpus de notícias em inglês pré-processado:
#   palavras raras → <unk>, números → N, fim de linha → <eos>
#   o vocabulário tem ~10.000 palavras únicas

def carregar_ptb(caminho):
    # lê o arquivo, substitui quebras de linha por <eos> e separa em lista de palavras
    with open(caminho, 'r', encoding='utf-8') as f:
        return f.read().replace('\n', ' <eos> ').split()

palavras_treino = carregar_ptb('penntreebank/ptbdataset/ptb.train.txt')
palavras_valid  = carregar_ptb('penntreebank/ptbdataset/ptb.valid.txt')
palavras_teste  = carregar_ptb('penntreebank/ptbdataset/ptb.test.txt')

# vocabulário: lista ordenada de todas as palavras do conjunto de treino
vocab      = sorted(set(palavras_treino))
vocab_size = len(vocab)
print(f'vocabulário : {vocab_size} palavras únicas')
print(f'tokens      : treino={len(palavras_treino):,} | valid={len(palavras_valid):,} | teste={len(palavras_teste):,}')

# dicionários de conversão bidirecional: palavra ↔ índice inteiro
# a rede não processa texto — só números. "the" pode ser o índice 8432.
palavra2idx = {p: i for i, p in enumerate(vocab)}
idx2palavra  = vocab
unk_idx      = palavra2idx.get('<unk>', 0)

def tokens_para_ids(tokens):
    # converte lista de strings em array numpy de inteiros
    return np.array([palavra2idx.get(t, unk_idx) for t in tokens], dtype=np.int64)

ids_treino = tokens_para_ids(palavras_treino)
ids_valid  = tokens_para_ids(palavras_valid)
ids_teste  = tokens_para_ids(palavras_teste)

# 2. inicialização dos parâmetros do modelo
#
# embedding: tabela de consulta (vocab_size × embed_dim) — cada linha é o vetor de uma palavra
# weight tying: a projeção de saída reutiliza a transposta do embedding, sem matriz wy separada
# xavier: inicializa com escala sqrt(1/fan_in) para manter variância estável no início
# requires_grad=True: habilita o autograd — pytorch calcula ∂loss/∂param automaticamente

def t(arr):
    # numpy array → tensor pytorch com gradiente habilitado
    return torch.tensor(arr, dtype=torch.float32, device=device, requires_grad=True)

def inicializar_parametros():
    D = EMBED_DIM
    H = HIDDEN_SIZE
    V = vocab_size
    C = D + H  # tamanho da entrada concatenada [h_{t-1}; x_t]

    E = t(np.random.uniform(-0.1, 0.1, (V, D)))  # tabela de embedding

    sc = np.sqrt(1.0 / C)
    Wf = t(np.random.randn(H, C) * sc)  # forget gate — o que esquecer da memória
    Wi = t(np.random.randn(H, C) * sc)  # input gate  — o que aceitar de novo
    Wg = t(np.random.randn(H, C) * sc)  # candidato   — nova informação proposta
    Wo = t(np.random.randn(H, C) * sc)  # output gate — quanto da memória revelar

    # bias da forget gate começa em 1: mantém a memória fluindo no início do treino
    bf = t(np.ones((1, H)))
    bi = t(np.zeros((1, H)))
    bg = t(np.zeros((1, H)))
    bo = t(np.zeros((1, H)))

    by = t(np.zeros((1, V)))  # bias de saída (weight tying: sem Wy separada)

    return dict(E=E, Wf=Wf, Wi=Wi, Wg=Wg, Wo=Wo,
                bf=bf, bi=bi, bg=bg, bo=bo, by=by)

params = inicializar_parametros()

# 3. forward pass da lstm (implementada manualmente)
#
# para cada passo de tempo t, a lstm executa:
#   z_t = [h_{t-1}; x_t]              → concatena estado anterior + embedding atual
#   f_t = sigmoid(Wf @ z_t + bf)      → forget gate: quanto esquecer de c_{t-1}
#   i_t = sigmoid(Wi @ z_t + bi)      → input gate:  quanto aceitar do candidato
#   g_t = tanh(Wg @ z_t + bg)         → candidato:   nova informação proposta
#   o_t = sigmoid(Wo @ z_t + bo)      → output gate: quanto de c_t revelar
#   c_t = f_t * c_{t-1} + i_t * g_t  → atualiza a memória de longo prazo
#   h_t = o_t * tanh(c_t)             → estado oculto: saída do passo t

def lstm_forward(input_ids, h, c, params):
    E, Wf, Wi, Wg, Wo = params['E'], params['Wf'], params['Wi'], params['Wg'], params['Wo']
    bf, bi, bg, bo     = params['bf'], params['bi'], params['bg'], params['bo']

    _, T = input_ids.shape
    hiddens = []  # guarda h_t de cada passo de tempo

    for step in range(T):
        x_t = E[input_ids[:, step]]          # busca o embedding da palavra atual
        z_t = torch.cat([h, x_t], dim=1)    # concatena com o estado anterior

        f_t = torch.sigmoid(z_t @ Wf.T + bf)  # forget gate
        i_t = torch.sigmoid(z_t @ Wi.T + bi)  # input gate
        g_t = torch.tanh   (z_t @ Wg.T + bg)  # candidato
        o_t = torch.sigmoid(z_t @ Wo.T + bo)  # output gate

        c = f_t * c + i_t * g_t   # atualiza a memória
        h = o_t * torch.tanh(c)   # calcula a saída

        hiddens.append(h)

    return hiddens, h, c

# 4. forward completo: lstm → projeção → cross-entropy loss

def forward(inp_np, tgt_np, h, c, params):
    inp = torch.tensor(inp_np, dtype=torch.long, device=device)
    tgt = torch.tensor(tgt_np, dtype=torch.long, device=device)

    hiddens, h_n, c_n = lstm_forward(inp, h, c, params)

    # empilha os estados ocultos de todos os passos: (T*B, H)
    H_all = torch.stack(hiddens, dim=0).reshape(-1, HIDDEN_SIZE)

    # projeção via weight tying: logits = H @ E^T + by  →  (T*B, vocab_size)
    logits = H_all @ params['E'].T + params['by']

    targets_flat = tgt.T.reshape(-1)

    # cross-entropy: -log(probabilidade da palavra correta), média sobre todas as posições
    log_probs = torch.log_softmax(logits, dim=1)
    loss      = -log_probs[torch.arange(len(targets_flat), device=device), targets_flat].mean()

    return loss, h_n, c_n

# 5. gradient clipping — evita explosão de gradientes em rnns
#   calcula a norma l2 global e escala todos os gradientes se passar de grad_clip

def clip_gradientes(param_list, max_norm=GRAD_CLIP):
    norma = torch.sqrt(sum(p.grad.data.norm() ** 2
                           for p in param_list if p.grad is not None))
    if norma > max_norm:
        fator = max_norm / norma
        for p in param_list:
            if p.grad is not None:
                p.grad.data.mul_(fator)

# 6. perplexidade — métrica padrão de modelos de linguagem
#   ppl = exp(cross-entropy média)
#   interpretação: ppl=k → modelo se comporta como se tivesse k palavras igualmente prováveis
#   referência no ptb: aleatório~10.000 | lstm simples~200 | lstm bem treinada~130-160

def calcular_perplexidade(ids, params):
    B, T  = BATCH_SIZE, SEQ_LEN
    n     = (len(ids) // B) * B
    dados = ids[:n].reshape(B, -1)
    n_passos = (dados.shape[1] - 1) // T

    total = 0.0
    h = torch.zeros(B, HIDDEN_SIZE, device=device)
    c = torch.zeros(B, HIDDEN_SIZE, device=device)

    with torch.no_grad():  # desativa o autograd — só leitura, sem atualização
        for p in range(n_passos):
            inp = dados[:, p * T:(p + 1) * T]
            tgt = dados[:, p * T + 1:(p + 1) * T + 1]
            loss, h, c = forward(inp, tgt, h, c, params)
            total += loss.item()

    return np.exp(total / n_passos)

# 7. loop de treinamento com sgd + decaimento de lr
#
# tbptt: corpus reorganizado em B sequências paralelas; processa T tokens por vez
# e passa h/c entre janelas, mas corta o grafo com detach() (sem gradientes entre janelas)
# sgd: w ← w - lr × ∂loss/∂w
# decaimento: lr ← lr × 0.8 a cada época após lr_decay_start

def treinar(ids_treino, ids_valid, params):
    B, T       = BATCH_SIZE, SEQ_LEN
    lr         = LR
    param_list = list(params.values())

    n        = (len(ids_treino) // B) * B
    dados    = ids_treino[:n].reshape(B, -1)
    n_passos = (dados.shape[1] - 1) // T

    hist_custo, hist_ppl_t, hist_ppl_v = [], [], []

    for epoca in range(1, EPOCHS + 1):

        # decaimento do lr após a época de corte
        if epoca > LR_DECAY_START:
            lr *= LR_DECAY

        total_custo = 0.0
        h = torch.zeros(B, HIDDEN_SIZE, device=device)
        c = torch.zeros(B, HIDDEN_SIZE, device=device)

        for p in range(n_passos):
            inp = dados[:, p * T:(p + 1) * T]
            tgt = dados[:, p * T + 1:(p + 1) * T + 1]

            # zera gradientes acumulados do passo anterior
            for param in param_list:
                if param.grad is not None:
                    param.grad.zero_()

            # detach(): corta o grafo entre janelas — memória passa, gradientes não
            h = h.detach()
            c = c.detach()

            loss, h, c = forward(inp, tgt, h, c, params)
            loss.backward()

            clip_gradientes(param_list)

            # atualização sgd manual: w ← w - lr × grad
            with torch.no_grad():
                for param in param_list:
                    if param.grad is not None:
                        param.data -= lr * param.grad.data

            total_custo += loss.item()

        custo_medio = total_custo / n_passos
        ppl_treino  = np.exp(custo_medio)
        ppl_valid   = calcular_perplexidade(ids_valid, params)

        hist_custo.append(custo_medio)
        hist_ppl_t.append(ppl_treino)
        hist_ppl_v.append(ppl_valid)

        print(f'época {epoca:2d}/{EPOCHS} | lr={lr:.5f} | '
              f'custo={custo_medio:.4f} | '
              f'ppl treino={ppl_treino:.1f} | '
              f'ppl valid={ppl_valid:.1f}')

    return params, hist_custo, hist_ppl_t, hist_ppl_v


params, hist_custo, hist_ppl_t, hist_ppl_v = treinar(ids_treino, ids_valid, params)

# 8. avaliação final no conjunto de teste
# o teste é usado uma única vez ao final — representa dados que o modelo nunca viu

ppl_teste = calcular_perplexidade(ids_teste, params)
print(f'\nperplexidade final — teste: {ppl_teste:.2f}')
print('referência: aleatório~10.000 | lstm simples~200 | lstm bem treinada~130-160')

# 9. gráficos de avaliação

def plotar_graficos(hist_custo, hist_ppl_t, hist_ppl_v, ppl_teste):
    epocas = range(1, len(hist_custo) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('lstm — modelo de linguagem (penn treebank)', fontsize=14)

    axes[0].plot(epocas, hist_custo, 'bo-', linewidth=2, markersize=5)
    axes[0].set_title('custo de treinamento por época')
    axes[0].set_xlabel('época')
    axes[0].set_ylabel('cross-entropy (média)')
    axes[0].grid(True)

    axes[1].plot(epocas, hist_ppl_t, 'bo-', linewidth=2, markersize=5, label='treino')
    axes[1].plot(epocas, hist_ppl_v, 'rs-', linewidth=2, markersize=5, label='validação')
    axes[1].axhline(ppl_teste, color='green', linestyle='--', linewidth=1.5,
                    label=f'teste final: {ppl_teste:.1f}')
    axes[1].set_title('perplexidade por época')
    axes[1].set_xlabel('época')
    axes[1].set_ylabel('perplexidade')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('resultados_lstm.png', dpi=150, bbox_inches='tight')
    print('gráfico salvo em resultados_lstm.png')
    plt.close()

plotar_graficos(hist_custo, hist_ppl_t, hist_ppl_v, ppl_teste)

# 10. geração de texto — análise qualitativa do modelo

def gerar_texto(params, palavra_inicial, n_palavras=40):
    h   = torch.zeros(1, HIDDEN_SIZE, device=device)
    c   = torch.zeros(1, HIDDEN_SIZE, device=device)
    idx = palavra2idx.get(palavra_inicial, unk_idx)
    gerado = [palavra_inicial]

    with torch.no_grad():
        for _ in range(n_palavras):
            inp = torch.tensor([[idx]], dtype=torch.long, device=device)
            hiddens, h, c = lstm_forward(inp, h, c, params)

            logits = hiddens[-1] @ params['E'].T + params['by']
            probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            idx    = np.random.choice(vocab_size, p=probs)
            gerado.append(idx2palavra[idx])

    return ' '.join(gerado)

print('\n─── geração de texto ───')
print(gerar_texto(params, 'the', n_palavras=40))
