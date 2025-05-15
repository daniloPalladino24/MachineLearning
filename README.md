# Progetto Gatto-Topo-Formaggio: Reinforcement Learning in Ambienti a Griglia

## Introduzione

Questo progetto implementa un ambiente di simulazione chiamato "Gatto-Topo-Formaggio" utilizzando Python, PyTorch e Gymnasium per dimostrare l'applicazione del reinforcement learning in un contesto di navigazione di labirinti. Il progetto è ispirato agli esempi di creazione di ambienti personalizzati in Gymnasium, come illustrato nella documentazione ufficiale: [Environment Creation Tutorial](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/).

Nel gioco, un topo (l'agente controllato tramite reinforcement learning) deve navigare attraverso un labirinto per raggiungere un pezzo di formaggio, evitando di essere catturato da un gatto che si muove casualmente nell'ambiente. L'implementazione utilizza tecniche di Q-learning per permettere all'agente topo di apprendere il comportamento ottimale in diversi tipi di labirinti.

## Metodologia

### Struttura dell'Ambiente

Il progetto implementa tre varianti dell'ambiente:

1. **Griglia vuota 5x5**: Un ambiente semplice senza ostacoli.
2. **Griglia 5x5 con muri sui bordi**: Un ambiente con barriere che definiscono "stanze" e corridoi.
3. **Griglia 10x10 con ostacoli**: Un ambiente più complesso con vari oggetti che occupano alcune celle.

Ogni ambiente è implementato come una sottoclasse di `gymnasium.Env`, seguendo le linee guida ufficiali di Gymnasium. L'implementazione include:

- Definizione degli spazi di osservazione e azione
- Sistema di ricompense che incentiva l'avvicinamento al formaggio ed evitare il gatto
- Logica per la terminazione degli episodi (raggiungimento del formaggio, cattura da parte del gatto, o superamento del limite di passi)
- Rendering grafico dell'ambiente per la visualizzazione

### Algoritmo di Apprendimento

Per l'apprendimento, utilizziamo il Q-learning, un algoritmo di reinforcement learning model-free che permette all'agente di apprendere una politica ottimale interagendo con l'ambiente. Il processo include:

- Creazione di una Q-table per memorizzare i valori di qualità delle coppie stato-azione
- Utilizzo di una strategia ε-greedy per bilanciare esplorazione e sfruttamento
- Aggiornamento dei valori Q in base alle ricompense ricevute e alle stime di ricompense future
- Decadimento graduale del parametro epsilon per ridurre l'esplorazione nel tempo

## Strumenti Utilizzati

- **Python**: Linguaggio di programmazione principale
- **PyTorch**: Framework per l'implementazione di algoritmi di deep learning
- **Gymnasium**: Libreria per la creazione di ambienti di reinforcement learning
- **NumPy**: Libreria per il calcolo numerico
- **Pygame**: Libreria per la visualizzazione grafica dell'ambiente
- **Matplotlib**: Libreria per la visualizzazione dei dati e dei risultati dell'apprendimento

## Implementazione

### Struttura del Progetto

```
gatto-topo-formaggio/
├── 5x5_vuoto/             # Implementazione della griglia vuota 5x5
│   ├── cat_mouse_cheese_env.py
│   ├── graphics.py
│   └── main.py
├── 5x5_bordi/             # Implementazione della griglia 5x5 con muri
│   ├── cat_mouse_cheese_env.py
│   ├── graphics.py
│   └── main.py
├── 10x10_ostacoli/        # Implementazione della griglia 10x10 con ostacoli
│   ├── cat_mouse_cheese_env.py
│   ├── graphics.py
│   └── main.py
├── img/                   # Directory per le immagini utilizzate nella visualizzazione
│   ├── cat.png
│   ├── mouse.png
│   └── cheese.png
└── README.md
```

### Dettagli Implementativi Chiave

#### Ambiente (`cat_mouse_cheese_env.py`)

La classe `CatMouseCheeseEnv` definisce l'ambiente di gioco:
- Lo stato è rappresentato dalle posizioni del topo, del gatto e del formaggio
- Le azioni possibili sono 4: su, giù, sinistra, destra
- Il sistema di ricompense incentiva l'avvicinamento al formaggio e l'allontanamento dal gatto
- Penalizza comportamenti non ottimali come rimanere fermi o visitare più volte la stessa posizione

#### Visualizzazione (`graphics.py`)

Il modulo `graphics.py` gestisce la visualizzazione dell'ambiente:
- Inizializza la finestra di gioco con Pygame
- Disegna la griglia, i muri/ostacoli e le entità (topo, gatto, formaggio)
- Fornisce funzioni per aggiornare la visualizzazione ad ogni passo

#### Allenamento e Test (`main.py`)

Il modulo `main.py` contiene:
- La funzione `train_q_learning()` per addestrare l'agente
- Funzioni per salvare e caricare la Q-table
- La funzione `test_q_learning()` per visualizzare il comportamento dell'agente addestrato
- La funzione `calculate_accuracy()` per valutare le prestazioni dell'agente

## Risultati

### Analisi delle Prestazioni

L'algoritmo di Q-learning è stato in grado di apprendere strategie efficaci in tutti e tre gli ambienti. Dopo un milione di episodi di addestramento:

- **Griglia vuota 5x5**: L'agente ha raggiunto un'accuratezza di oltre il 90%
- **Griglia 5x5 con muri**: L'accuratezza è stata circa dell'85%
- **Griglia 10x10 con ostacoli**: L'accuratezza è stata circa del 75%

Le curve di apprendimento mostrano un miglioramento costante della ricompensa totale nel corso degli episodi, con una stabilizzazione verso la fine dell'addestramento.

### Confronto tra le Griglie

- La **griglia vuota 5x5** è stata la più semplice da apprendere, dato che l'agente ha meno vincoli di movimento
- La **griglia 5x5 con muri** ha richiesto all'agente di apprendere a navigare in corridoi e attraverso "stanze", aumentando la complessità
- La **griglia 10x10 con ostacoli** è stata la più impegnativa a causa delle dimensioni maggiori e degli ostacoli, richiedendo strategie più complesse

### Comportamenti Appresi

L'agente ha dimostrato di apprendere comportamenti intelligenti, tra cui:
- Prendere strade più lunghe ma sicure per evitare il gatto
- Sfruttare gli ostacoli come "scudi" per proteggersi dal gatto
- Attendere nei punti sicuri quando il gatto blocca temporaneamente il percorso diretto verso il formaggio

## Conclusioni

Questo progetto dimostra l'efficacia del Q-learning nel risolvere problemi di navigazione in ambienti discreti con elementi dinamici. L'agente topo è stato in grado di apprendere strategie ottimali per raggiungere il formaggio evitando il gatto in diverse configurazioni di labirinto.

La complessità dell'ambiente influisce significativamente sul tempo di apprendimento e sulle prestazioni finali dell'agente. Gli ambienti più complessi richiedono più episodi di addestramento ma permettono l'emergere di comportamenti più sofisticati.

I risultati suggeriscono che il reinforcement learning è una tecnica promettente per addestrare agenti in grado di navigare in ambienti complessi e dinamici, con potenziali applicazioni in robotica, sistemi di navigazione autonoma e videogiochi.

## Possibili Estensioni

- Implementazione di tecniche di Deep Q-Learning per gestire spazi di stato più complessi
- Aggiunta di più gatti con comportamenti diversi (ad esempio, uno che insegue attivamente il topo)
- Introduzione di ricompense dinamiche (ad esempio, formaggio che appare in posizioni diverse)
- Sperimentazione con altri algoritmi di reinforcement learning come SARSA o Actor-Critic

## Crediti

Questo progetto è stato sviluppato come parte di un corso universitario sul reinforcement learning. L'implementazione si basa sulle linee guida di Gymnasium e sull'esempio fornito nella documentazione ufficiale.
