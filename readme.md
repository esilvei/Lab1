# Relatório Técnico: Reconhecimento Facial para Fechadura Biométrica (Artefato 1)

## 1. Visão Geral
Este documento detalha o desenvolvimento do sistema de inteligência artificial para o projeto de uma fechadura biométrica utilizando a placa FPGA DE2-115. O foco desta etapa foi a criação de um pipeline robusto para aquisição de dados, treinamento de uma **Tiny-CNN** e a exportação dos parâmetros quantizados para o hardware.

## 2. Desenvolvimento do Dataset

O dataset foi projetado para um problema de classificação binária: **Classe 1 (Autorizados)** e **Classe 0 (Desconhecidos)**.

### Classe 1: Autorizados (Equipe)
* **Origem dos Dados**: As imagens foram extraídas de vídeos em formato `.mp4` ou pastas de fotos contendo os integrantes da equipe.
* **Processamento**: Foi utilizado o algoritmo *Haar Cascade* para a detecção facial automática, garantindo que apenas a região do rosto fosse recortada.
* **Sanitização**: Os nomes dos arquivos e pastas foram normalizados (remoção de acentos e caracteres especiais) para evitar inconsistências no carregamento.
* **Aumento de Dados (Augmentation)**: Para atingir a meta de **400 imagens por integrante**, aplicou-se técnicas de aumento de dados, incluindo espelhamento horizontal (*flip*) e variações controladas de brilho e contraste.

### Classe 0: Desconhecidos (Unknowns)
Para garantir que a fechadura não permita o acesso de pessoas estranhas à equipe, a Classe 0 foi composta de forma diversificada:
* **Dataset Externo**: Utilizou-se o **UCF Selfie-dataset**, de onde foram extraídas 7.000 selfies de pessoas diversas para treinar a capacidade de negação do modelo.
* **Imagens de Fundo (Background)**: Foram geradas **300 imagens sintéticas** representando paredes e ruído visual com diferentes gradientes de iluminação. Isso ensina a rede que a ausência de um rosto claro também deve resultar em "Acesso Negado".
* **Consistência**: Todas as imagens desta classe passaram pelo mesmo processo de detecção facial e recorte $32 \times 32$ aplicado à Classe 1.

## 3. Processo de Treinamento

O treinamento foi realizado em Python utilizando a biblioteca TensorFlow/Keras, estruturado para otimização automática através de busca de hiperparâmetros.

* **Busca de Hiperparâmetros**: Utilizou-se o **Keras Tuner** (algoritmo Hyperband) para definir a melhor configuração da rede, testando variações de neurônios na camada densa (entre 16 e 64) e taxas de aprendizado (*learning rate*) entre $10^{-2}$ e $10^{-4}$.
* **Balanceamento**: Devido à disparidade numérica entre as classes (75% desconhecidos e 25% autorizados), utilizou-se a técnica de `class_weight` (pesos balanceados) durante o ajuste do modelo, garantindo que a rede desse a devida importância aos membros da equipe.
* **Monitoramento**: O treinamento contou com o *callback* `EarlyStopping`, que interrompe o processo caso a perda de validação pare de diminuir, prevenindo o *overfitting*.
* **Rastreamento**: Todo o progresso e métricas de cada experimento foram registrados no **MLflow** para auditoria e comparação de performance.

## 4. Quantização e Conversão para .MIF

Para que o modelo matemático funcione na FPGA, foi necessário converter os pesos de ponto flutuante (*float32*) para **Ponto Fixo (Q1.7)**.

* **Lógica de Quantização**: Cada peso foi multiplicado por $2^7$ (128) e arredondado para o inteiro mais próximo, limitando o valor ao intervalo de 8 bits ($-128$ a $127$).
* **Formato de Saída**: O pipeline gera automaticamente arquivos `.mif` (*Memory Initialization File*) individuais para cada camada do modelo.
* **Arquivos Gerados**:
    * `conv2d_hardware_weights.mif`: Contém os filtros da convolução (36 valores para 4 filtros $3 \times 3$).
    * `conv2d_hardware_biases.mif`: Contém os 4 valores de bias da primeira camada.
    * `dense_weights.mif` e `dense_biases.mif`: Pesos e biases das camadas totalmente conectadas.
* **Configuração de Hardware**: Os arquivos são gerados com `WIDTH = 8` e `ADDRESS_RADIX = HEX`, prontos para serem carregados em blocos de memória ROM (M9K) no Intel Quartus Prime.

## 5. Estrutura do Projeto 

O projeto segue princípios de engenharia de software para garantir modularidade:
* **Config**: Centralização de caminhos e parâmetros globais.
* **Preprocessor**: Encapsulamento da lógica de visão computacional e detecção.
* **Engine**: Gerenciamento do ciclo de vida de treinamento e tuning.
* **Evaluator**: Cálculo de métricas e geração de matriz de confusão.
* **Exporter**: Conversão de modelos Keras para formatos de hardware.

---
**Disciplina:** ENGG52 - LABORATÓRIO INTEGRADO I-A (UFBA)  
**Docentes:** Wagner Oliveira e Edmar de Souza