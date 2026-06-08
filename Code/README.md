# CAPE KeyDesign Reproduction Notes

This repository contains data-prep scripts and experiment runners for the KeyDesign task.

## What is in the repo

- `CAPE_KeyDesign.py`: builds TextCNN + GCN embeddings per project and writes a `emb.csv` embedding file for each project.
- `data_process/`: scripts to derive tokens, nodes, edges, and labels from raw Java sources, network graphs, and metric CSVs.
- `RQ1/downstream_task/`: scripts for downstream classification and evaluation.

## Data preparation pipeline (expected order)

The scripts imply the following pipeline. Each step writes files under `dataset_keyDesign_FGCS/<project>/`.

1. `data_process/extract_Compared_Class.py`
   - Inputs: `datasets_0528/FGCS/<project>_DM+NM+INM.csv`
   - Output: `compared_Class.txt`
2. `data_process/ast.py`
   - Inputs: raw Java source roots (see `data_process/path.txt`), `compared_Class.txt`
   - Outputs: `repeated_tokens.txt`, `repeated_tokens_cross.txt`
3. `data_process/deduplication-token.py`
   - Inputs: `repeated_tokens*.txt`, `compared_Class.txt`
   - Outputs: `tokens.txt`, `tokens_cross.txt`
4. `data_process/node_txt.py`
   - Input: `tokens.txt`
   - Output: `nodes.txt`
5. `data_process/original_edges.py`
   - Input: `dataset/软件网络图/AllSoftNets_FGCS/CCN_SoftNet_FGCS_<project>.net`
   - Output: `original_edges_weight.txt`
6. `data_process/edges.py` and `data_process/edges_weight.py`
   - Inputs: `nodes.txt`, `original_edges_weight.txt`
   - Outputs: `edges.txt`, `edges_weight.txt`
7. `data_process/token_map.py`
   - Input: `tokens.txt`
   - Outputs: `vocab_emb_dict_30.txt`, `tokens_map.txt`
8. `data_process/embedding.py`
   - Input: `vocab_emb_dict_30.txt`
   - Output: `vocab_emb_dict_30.csv` (word2vec vectors)
9. `data_process/Process-Binary.py`
   - Input: `datasets_0528/FGCS/<project>_DM+NM+INM.csv`
   - Output: `Process-Binary.csv`


## Experiments and outputs

### Embedding generation

- Script: `CAPE_KeyDesign.py`
- Reads (per project, from `dataset_keyDesign_Sora/<project>/`):
  - `tokens_map.txt` – tokenized source code sequences
  - `Process-Binary.csv` – node attributes and KeyDesign labels
  - `vocab_emb_dict_30.csv` – pre-trained word embedding matrix
  - `edges_weight.txt` – weighted directed edges between nodes (class dependencies)
- Output: `hyq_emb_AN_Sora_non-dw.csv` – 32‑dimensional hybrid embeddings for all nodes in the project

### Downstream classification 

- Script: `RQ1/downstream_task/cross_boot_v1.py`
- Inputs:
  - `dataset_keyDesign_FGCS/<project>/Process-Binary.csv`
  - CGCN embeddings like `CGCN_emb_FGCS_directed_ins.csv`
  - `configs/cross_project_demo.txt`
- Output: CSV metrics under `cross_results/source_results_demo-v1/FGCS/`

## Experimental protocol: train/test isolation in cross‑validation

In each 2‑fold cross‑validation round within a project:
- Nodes are split into `train_nodes` and `test_nodes` (no overlap).
- All edges incident to `test_nodes` are removed to form `G_train`.
- Word2Vec is trained only on walks over `G_train`.
- GCN is trained only on `G_train` using `FullBatchNodeGenerator`.
- CNN is trained only on `train_nodes` token sequences.
- The fusion network is trained only on `train_nodes`.
- After training, weights are frozen; full graph (`G`) is used for inference to obtain test‑node embeddings.

## Mapping between paper components and source files

| Paper component | Source file / function |
|----------------|------------------------|
| Data preprocessing | `data_process/*.py` |
| Semantic encoder | `CAPE_KeyDesign.py` - `TextCNN_model()` |
| Structural encoder | `CAPE_KeyDesign.py` - `_compute_initial_node_attributes()`, GCN |
| Fusion & joint training | `CAPE_KeyDesign.py` - `mix()` |
| Embedding generation | `CAPE_KeyDesign.py` main loop |
| Downstream RF | `RQ1/downstream_task/cross_boot_v1.py` |



