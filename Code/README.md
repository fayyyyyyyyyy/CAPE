# CAPE KeyDesign Reproduction Notes

This repository contains data-prep scripts and experiment runners for the KeyDesign task.

## What is in the repo

- `CAPE_KeyDesign.py`: provides shared TextCNN and graph-processing functions used by the foldwise representation-learning pipeline.
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

### Within-project evaluation

- Entry point: `RQ1/downstream_task/keyDesign_within.py`
- Foldwise representation learning: `foldwise_experiment/representation_foldwise.py`
- Repeated two-fold evaluation and downstream Random Forest: `foldwise_experiment/run_within_project.py`
- In each of the 100 repetitions, a new stratified 2-fold split is generated.
- For every fold, the representation models are newly trained using the training fold, and the same train/test indices are used for the downstream Random Forest evaluation.
- Required files for each project: `tokens_map.txt`, `Process-Binary.csv`, and `edges_weight.txt`.




