![Workflow](https://github.com/marcosd3souza/TDGC/blob/main/TDGC.png)

# Topological Deep Graph Clustering
- step 1: apply topological data analysis (TDA) to extract meaningful graph by Mapper
  - step 1.1: using tmap lib: https://tmap.readthedocs.io/en/latest/basic.html
- step 1.1: apply spectral clustering on TDA graph
- step 2: generate new graph by inverse mapping
- step 3: generate knowledge graph by using the new graph and labels
- step 3: apply GCN
  - step 3.1: add contrastive term to extract meaningful embeddings
