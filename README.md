# parallel-kmeans
parallel solution for kmeans problem

# N.B.
Nell'implementazione cuda non si può scegliere un numero troppo grande di K (numero di cluster), consuma tutta la shared memory.
*soluzione: fare il tiling.

problema con il false sharing nelle funzioni per resettare gli array, schedule(static,16) va bene solo se la memoria è allineata (ma non lo è).
