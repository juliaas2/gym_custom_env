# Relatório – Coverage Path Planning com PPO

## Métrica principal: cobertura do ambiente

## 1. Objetivo

O objetivo deste projeto foi treinar um agente de aprendizado por reforço capaz de resolver o problema de **Coverage Path Planning (CPP)** em ambientes GridWorld com obstáculos, utilizando a biblioteca Gymnasium.

O foco principal foi maximizar a **cobertura do ambiente**, isto é, o percentual de células livres visitadas pelo agente durante um episódio. Essa foi a métrica central usada para acompanhar o treinamento, comparar modelos e escolher o resultado final.

---

## 2. Estratégia escolhida

A estratégia adotada foi o uso do algoritmo **PPO (Proximal Policy Optimization)**.

A escolha do PPO se justifica por:

* ser um algoritmo estável e amplamente utilizado em aprendizado por reforço;
* funcionar bem com espaços de observação estruturados;
* ser adequado para problemas de controle sequencial, como navegação;
* permitir avaliar o desempenho do agente pela **cobertura média** obtida ao longo dos episódios.

Além disso, foi utilizado **curriculum learning**, no qual o agente é treinado progressivamente em ambientes mais difíceis para melhorar a cobertura em cenários maiores.

### Estrutura do ambiente

O ambiente CPP foi baseado em GridWorld com as seguintes características:

* grid com obstáculos;
* observação composta por:

  * posição do agente normalizada;
  * percentual de cobertura;
  * visão local em uma matriz 5x5.

### Função de recompensa

A reward foi projetada para incentivar diretamente a cobertura:

| Ação              | Reward |
| ----------------- | ------ |
| Nova célula       | +1.0   |
| Revisita          | -0.3   |
| Colisão           | -0.5   |
| Passo             | -0.1   |
| Cobertura total   | +10.0  |
| Falha por timeout | -5.0   |

Essa estrutura força o agente a:

* explorar novas áreas e aumentar a cobertura;
* evitar loops e revisitas que reduzem a eficiência;
* minimizar passos desnecessários;
* buscar cobertura alta em menos tempo.

---

## 3. Processo de treinamento

O treinamento foi realizado em múltiplas etapas, sempre com foco em maximizar a cobertura média do ambiente.

### Treinamento base

Inicialmente, o agente foi treinado em ambientes menores, como o 5x5.

### Escalonamento com curriculum

Depois, o treinamento foi expandido para ambientes maiores:

* 10x10;
* 20x20.

Esse escalonamento ajudou o agente a aprender primeiro padrões simples de cobertura e, em seguida, se adaptar a espaços mais complexos.

### Treino agressivo

O treinamento final foi feito com um script automatizado de múltiplos rounds para tentar elevar a cobertura média:

* 10 rounds;
* 10 milhões de timesteps por round;
* total de até **100 milhões de timesteps**.

A cada round:

1. O melhor modelo disponível era carregado.
2. Um novo treinamento era executado.
3. O modelo resultante era testado.
4. A **cobertura média** era verificada.

O processo continuava até:

* atingir 90% de cobertura média no ambiente 20x20;
* ou completar os 10 rounds.

### Limitação computacional

O treinamento completo não pôde ser estendido além disso porque o computador não suportou continuar rodando por mais tempo com esse custo computacional.

Por esse motivo, o **último modelo obtido foi escolhido como modelo final**, mesmo sem atingir 90% de cobertura média.

---

## 4. Resultados

O melhor modelo encontrado apresentou os seguintes valores de **cobertura média**:

* **5x5 (3 obstáculos):** 95.68%
* **10x10 (12 obstáculos):** 80.74%
* **20x20 (48 obstáculos):** 66.18%
* **Cobertura média global:** 80.87%

Esses números foram o critério principal para selecionar o modelo final.

---

## 5. Análise dos resultados

O agente conseguiu aprender uma política eficiente de exploração, especialmente em ambientes menores, o que se reflete diretamente na cobertura média obtida.

### Pontos positivos

* PPO convergiu de forma estável;
* o reward shaping funcionou bem para aumentar a cobertura;
* o curriculum learning ajudou na adaptação e no ganho de cobertura média.

### Limitações

* dificuldade de generalização para grids grandes, o que reduziu a cobertura em mapas maiores;
* alta dependência de tempo de treino para subir a cobertura média;
* limitação de hardware impediu mais iterações e pode ter limitado o desempenho final.

### Possíveis melhorias

* treinar por mais tempo com hardware mais forte;
* ajustar hiperparâmetros do PPO;
* testar RecurrentPPO;
* melhorar a função de recompensa para reduzir revisitas e aumentar ainda mais a cobertura.

---

## 6. Conclusão

O uso de PPO combinado com curriculum learning foi eficaz para aumentar a cobertura no problema de CPP, principalmente em ambientes menores.

Entretanto, a generalização para ambientes maiores ainda é um desafio, principalmente devido ao aumento da complexidade, ao crescimento do espaço de busca e às limitações computacionais que impediram continuar treinando até coberturas mais altas.

O modelo final representa o melhor resultado possível dentro das restrições de tempo e hardware disponíveis, sendo escolhido por apresentar a melhor cobertura média obtida no processo.

---

## 7. Como executar

### Treinar

```bash
python train_grid_world_cpp.py train 5 3 200 500000
```

### Testar

```bash
python train_grid_world_cpp.py test 5 3 200 --model-path data/NOME_DO_MODELO.zip
```

### Rodar treino agressivo

Executar o script PowerShell fornecido para múltiplos rounds de treino e avaliação da cobertura.

---

## 8. Observação final

O critério principal utilizado foi **cobertura média**, e não cobertura total.

Isso reflete melhor o desempenho do agente em cenários reais, onde atingir 100% de cobertura nem sempre é viável.

Em especial, neste ambiente, a **geração de obstáculos é aleatória**, o que pode criar configurações onde:

* algumas regiões ficam de difícil acesso;
* trajetórias ótimas exigem exploração muito longa;
* ou até situações praticamente impossíveis de cobrir completamente dentro do limite de passos.

Por esse motivo, a cobertura final de 100% nem sempre é atingível, e a **cobertura média** se torna uma métrica mais robusta e realista para avaliar o desempenho do agente.
