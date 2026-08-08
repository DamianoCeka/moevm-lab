# MoEVM Lab — panoramica italiana

MoEVM Lab è un progetto di ricerca per capire se gli expert di un modello MoE enorme possano essere gestiti come una gerarchia di memoria:

```text
VRAM: expert necessari subito
RAM: expert probabili o recenti
NVMe: archivio completo dei pesi
```

La versione 0.1 è già eseguibile, ma è un **simulatore**, non un runtime di Kimi K3. Genera routing trace riproducibili, confronta una cache LRU normale con un prefetch predittivo e misura:

- hit-rate VRAM e RAM;
- byte NVMe→RAM e RAM→VRAM;
- tempo bloccante di demand e prefetch;
- precisione del predictor;
- throughput stimato dal modello temporale configurato.

Il demo incluso ottiene un vantaggio sintetico, ma mostra anche più traffico PCIe. Questo è utile perché impedisce di presentare come innovazione un algoritmo che sposta soltanto il costo da una metrica all'altra.

## Avvio rapido

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
```

## Obiettivo successivo

La prossima milestone seria non è K3: è catturare il routing di un MoE piccolo, riprodurlo e confrontare le previsioni del simulatore con trasferimenti misurati su hardware reale. Solo dopo si passa a pinned memory, stream CUDA, NVMe asincrono e checkpoint reali.
