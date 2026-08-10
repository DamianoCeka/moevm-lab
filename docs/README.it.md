# MoEVM Lab — panoramica italiana

MoEVM Lab è un progetto di ricerca per capire se gli expert di un modello MoE enorme possano essere gestiti come una gerarchia di memoria:

```text
VRAM: expert necessari subito
RAM: expert probabili o recenti
NVMe: archivio completo dei pesi
```

La release 0.2 è un laboratorio eseguibile, non un runtime di Kimi K3. Genera
routing trace sintetici, cattura routing da un piccolo MoE reale e confronta
una cache LRU normale con un prefetch predittivo. Lo sviluppo successivo non
ancora rilasciato aggiunge calibrazione hardware e un primo runtime paginato
sincrono per il piccolo OLMoE. Il laboratorio misura:

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

Le decisioni del router sono misurate sul modello reale; lo speedup M1, gli
stall e il traffico del replay restano simulati.

## Sviluppo M2/M3 non rilasciato

Il nuovo P310 M.2 misura `6,04 GB/s + 299 us` nel fit delle letture casuali da
12 MiB, contro `1,84 GB/s + 465 us` del vecchio P2. Il trasferimento pinned
RAM→VRAM misura `13,00 GB/s + 22,3 us`. Con questo profilo il predictor attuale
ottiene `0,9791x` sui dieci trace reali: circa il 2,09% più lento e con il 4,91%
di traffico RAM→VRAM in più. Quindi l'SSD migliore è utile per il paging, ma non
rende automaticamente buono il prefetch.

È stato anche eseguito il primo smoke completo del runtime paginato sul modello
OLMoE reale. Con 32 slot per layer, i due token generati coincidono con il
riferimento CPU-offload. La memoria GPU allocata di picco scende da 8,770 a
6,899 GiB (−21,34%). Il passaggio con cache expert vuota è più lento (`0,677x`),
mentre la ripetizione che conserva la cache arriva a `1,372x`. È un segnale di
fattibilità, non un claim generale: c'è un solo prompt, due token generati, un
solo intervallo decode e la cache del sistema operativo non è controllata.

## Avvio rapido

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
```

## Obiettivo successivo

La prossima mossa seria non è K3: è ripetere il benchmark su più prompt e decode
più lunghi, separare meglio cache expert e cache del sistema operativo, poi
sovrapporre letture, copie H2D e calcolo con I/O asincrono e stream CUDA.
