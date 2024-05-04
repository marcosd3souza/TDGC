# Topological Deep Graph Clustering
- step 1: apply topological data analysis (TDA) to generate meaningful graph 
  - step 1.1: using tmap lib: https://tmap.readthedocs.io/en/latest/basic.html
- step 2: apply Louvain to detect communities in tmap graph
- step 3: apply GCN to optimize KL-divergence
  - step 3.1: calculate the divergence between cluster probabilities from Normal/random and communities probabilities by tmap-graph + Louvain 
