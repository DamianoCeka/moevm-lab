# MoEVM Lab — panoramica italiana

MoEVM Lab è un progetto di ricerca per capire se gli expert di un modello MoE enorme possano essere gestiti come una gerarchia di memoria:

```text
VRAM: expert necessari subito
RAM: expert probabili o recenti
NVMe: archivio completo dei pesi
```

La versione 0.2 è un laboratorio eseguibile, non un runtime di Kimi K3. Genera
routing trace sintetici, cattura routing da un piccolo MoE reale e confronta
una cache LRU normale con un prefetch predittivo. Misura:

- hit-rate VRAM e RAM;
- byte NVMe→RAM e RAM→VRAM;
- tempo bloccante di demand e prefetch;
- precisione del predictor;
- throughput stimato dal modello temporale configurato.

Il demo incluso ottiene un vantaggio sintetico, ma mostra anche più traffico PCIe. Questo è utile perché impedisce di presentare come innovazione un algoritmo che sposta soltanto il costo da una metrica all'altra.

## Stato v0.2

La milestone M1 è completata su `allenai/OLMoE-1B-7B-0924`: 10 trace reali,
438 token e 56.064 accessi agli esperti, con revisione del modello e hash dei
trace registrati. Il routing reale mostra il 44,09% di sovrapposizione temporale
media. Il replay simulato è quasi neutro (1,0022×) e aumenta il traffico
RAM→VRAM del 6,17%: è un risultato negativo utile, perché indica che il predictor
va migliorato prima di costruire il runtime.

Le decisioni del router sono misurate sul modello reale; speedup, stall e
traffico restano simulati. Il prossimo obiettivo è calibrare questi valori con
microbenchmark di RAM pinned/pageable, PCIe e NVMe.

## Avvio rapido

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
```

## Obiettivo successivo

La prossima milestone seria non è K3: è misurare RAM, PCIe e NVMe sulla
workstation e calibrare il replay dei trace reali. Solo dopo si passa a pinned
memory, stream CUDA, NVMe asincrono e checkpoint reali.
