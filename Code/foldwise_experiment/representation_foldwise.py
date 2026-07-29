from dataclasses import dataclass
from pathlib import Path
import os
import random
import sys
import types

import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec
from stellargraph import StellarGraph
from stellargraph.layer import GCN
from stellargraph.mapper import FullBatchNodeGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


CODE_ROOT = Path(__file__).resolve().parents[1]
RQ1_ROOT = CODE_ROOT / "RQ1"

rq1_utils = types.ModuleType("utils")
rq1_utils.__path__ = [str(RQ1_ROOT / "utils")]
sys.modules["utils"] = rq1_utils
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import CAPE_KeyDesign as cape_keydesign


CAPE_SOURCE = (CODE_ROOT / "CAPE_KeyDesign.py").resolve()
if Path(cape_keydesign.__file__).resolve() != CAPE_SOURCE:
    raise ImportError(
        "Unexpected CAPE_KeyDesign module: {}".format(
            cape_keydesign.__file__
        )
    )

DEFAULT_DATASET_ROOT = CODE_ROOT / "dataset_keyDesign_Sora"


@dataclass
class FoldFeatures:
    """保存一个fold中提取的特征 """

    train_idx: np.ndarray
    test_idx: np.ndarray
    train_features: np.ndarray
    test_features: np.ndarray


def set_all_seeds(seed):

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def _project_dir(project, dataset_root=None):
    root = Path(dataset_root) if dataset_root is not None else DEFAULT_DATASET_ROOT
    return root / project


def load_token_sequences(project, dataset_root=None):
    """读取token序列"""

    token_path = _project_dir(project, dataset_root) / "tokens_map.txt"
    sequences = []
    with token_path.open("r", encoding="utf-8") as token_file:
        for line_number, line in enumerate(token_file, start=1):
            if "\t" not in line:
                raise ValueError(
                    "Malformed tokens_map.txt line {} for {}".format(line_number, project)
                )
            raw_tokens = line.split("\t", 1)[1].strip().split()
            sequences.append([int(token) for token in raw_tokens])
    return sequences


def build_fold_semantic_inputs(
    project,
    train_idx,
    test_idx,
    seed,
    dataset_root=None,
    vector_size=30,
    window=5,
    epochs=20,
):

    sequences = load_token_sequences(project, dataset_root)
    train_sequences = [sequences[index] for index in train_idx]
    train_sentences = [
        [str(token) for token in sequence]
        for sequence in train_sequences
        if sequence
    ]
    if not train_sentences:
        raise ValueError("The training fold contains no semantic tokens for {}".format(project))

    word2vec = Word2Vec(
        sentences=train_sentences,
        vector_size=vector_size,
        window=window,
        min_count=1,
        sg=1,
        workers=1,
        seed=seed,
        epochs=epochs,
    )

    max_token_id = max(
        (max(sequence) for sequence in sequences if sequence),
        default=0,
    )
    embedding_matrix = np.zeros((max_token_id + 1, vector_size), dtype=np.float32)
    for token in word2vec.wv.index_to_key:
        token_id = int(token)
        if 0 <= token_id < embedding_matrix.shape[0]:
            embedding_matrix[token_id] = word2vec.wv[token]

    sequence_length = max((len(sequence) for sequence in train_sequences), default=1)
    all_inputs = pad_sequences(
        sequences,
        maxlen=sequence_length,
        padding="post",
        truncating="post",
        dtype="int32",
    )
    return (
        all_inputs[train_idx],
        all_inputs[test_idx],
        all_inputs,
        embedding_matrix,
        sequence_length,
    )


def load_graph_data(project, dataset_root=None):

    root = Path(dataset_root) if dataset_root is not None else DEFAULT_DATASET_ROOT
    cape_keydesign.baseURL = str(root)
    edgelist, node_data, _, node_ids, _ = cape_keydesign.load_GCN_data(project)
    edgelist = edgelist[["source", "target", "weight"]].copy()
    labels = node_data["KeyDesign"].astype(int).to_numpy()
    node_subjects = pd.Series(labels, index=node_ids, dtype="int64")

    if len(set(node_ids)) != len(node_ids):
        raise ValueError("Node identifiers must be unique for {}".format(project))
    graph_nodes = set(edgelist["source"]).union(set(edgelist["target"]))
    unknown_nodes = graph_nodes.difference(set(node_ids))
    if unknown_nodes:
        preview = sorted(unknown_nodes, key=str)[:10]
        raise ValueError(
            "Edge endpoints do not match node identifiers for {}: {}".format(
                project, preview
            )
        )
    return edgelist, node_data, node_subjects, node_ids


def _compute_initial_node_attributes(
    edgelist,
    node_data,
    node_ids,
    train_idx,
    train_node_set,
    seed,
    directed=False,
):

    cape_node_data = node_data.copy()
    fold_id_column = cape_node_data.columns[0]
    cape_node_data[fold_id_column] = node_ids
    return cape_keydesign._compute_initial_node_attributes(
        edgelist=edgelist,
        node_data=cape_node_data,
        node_ids=node_ids,
        id_col=fold_id_column,
        directed=directed,
        train_node_set=train_node_set,
        walk_params={
            "p": 0.25,
            "q": 2,
            "n": 10,
            "length": 80,
            "seed": seed,
            "weighted": True,
        },
        w2v_params={
            "vector_size": 128,
            "window": 10,
            "min_count": 1,
            "sg": 1,
            "workers": 1,
            "epochs": 20,
            "seed": seed,
        },
    )


def _fuse(semantic_output, structural_output, variant, name_suffix=""):
    normalized = variant.lower()
    if normalized in {"cape", "cape-an", "hybrid"}:
        return tf.keras.layers.Add(name="feature_fusion{}".format(name_suffix))(
            [0.5 * semantic_output, 0.5 * structural_output]
        )
    if normalized in {"cape-a", "semantic"}:
        return tf.keras.layers.Add(name="semantic_only{}".format(name_suffix))(
            [semantic_output, 0.0 * structural_output]
        )
    if normalized in {"cape-n", "structural"}:
        return tf.keras.layers.Add(name="structural_only{}".format(name_suffix))(
            [0.0 * semantic_output, structural_output]
        )
    raise ValueError("Unknown CAPE variant: {}".format(variant))


def _build_cape_models(
    project,
    embedding_matrix,
    sequence_length,
    generator,
    variant,
    name_suffix,
):
    semantic_input = Input(
        batch_shape=(1, None, sequence_length),
        dtype="int32",
        name="cnn_input{}".format(name_suffix),
    )
    semantic_encoder = cape_keydesign.TextCNN_model(
        semantic_input, embedding_matrix, project
    )
    gcn = GCN(
        layer_sizes=[32, 32],
        activations=["relu", "relu"],
        generator=generator,
        dropout=0.0,
    )
    structural_input, structural_output = gcn.in_out_tensors()
    fused = _fuse(
        semantic_encoder.output,
        structural_output,
        variant,
        name_suffix=name_suffix,
    )
    representation = Dense(
        32,
        activation="relu",
        name="cape_representation{}".format(name_suffix),
    )(fused)
    prediction = Dense(
        2,
        activation="softmax",
        name="prediction{}".format(name_suffix),
    )(representation)
    model = Model(
        inputs=[semantic_input, structural_input],
        outputs=prediction,
        name="cape_fold_model{}".format(name_suffix),
    )
    representation_model = Model(
        inputs=[semantic_input, structural_input],
        outputs=representation,
        name="cape_feature_extractor{}".format(name_suffix),
    )
    return model, representation_model


def _train_fusion_and_extract(
    project,
    train_inputs,
    all_inputs,
    embedding_matrix,
    sequence_length,
    train_generator,
    full_generator,
    train_flow,
    full_flow,
    variant,
    epochs,
    patience,
):
    model, representation_model = _build_cape_models(
        project,
        embedding_matrix,
        sequence_length,
        train_generator,
        variant,
        "_train",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        ReduceLROnPlateau(
            monitor="loss",
            patience=patience,
            factor=0.5,
            min_lr=0.00001,
        ),
        EarlyStopping(
            monitor="loss",
            patience=patience,
            restore_best_weights=True,
        ),
    ]
    model.fit(
        x=[np.expand_dims(train_inputs, axis=0), train_flow.inputs],
        y=train_flow.targets,
        epochs=epochs,
        verbose=2,
        callbacks=callbacks,
    )

    # 训练特征只在训练子图上计算。
    train_extracted = representation_model.predict(
        [np.expand_dims(train_inputs, axis=0), train_flow.inputs],
        verbose=0,
    )

    # 完整图与训练子图使用相同的网络结构，并复制已训练的权重。
    full_model, full_representation_model = _build_cape_models(
        project,
        embedding_matrix,
        sequence_length,
        full_generator,
        variant,
        "_full",
    )
    full_model.set_weights(model.get_weights())
    full_extracted = full_representation_model.predict(
        [np.expand_dims(all_inputs, axis=0), full_flow.inputs],
        verbose=0,
    )
    return train_extracted.squeeze(0), full_extracted.squeeze(0)


def generate_fold_features(
    project,
    train_idx,
    test_idx,
    seed,
    dataset_root=None,
    variant="cape",
    epochs=100,
    patience=50,
):

    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError("Both train_idx and test_idx must be non-empty")
    if np.any(train_idx < 0) or np.any(test_idx < 0):
        raise IndexError("Fold indices must be non-negative")
    if np.unique(train_idx).size != train_idx.size:
        raise ValueError("Training indices contain duplicates")
    if np.unique(test_idx).size != test_idx.size:
        raise ValueError("Test indices contain duplicates")
    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError("Training and test indices overlap")

    tf.keras.backend.clear_session()
    set_all_seeds(seed)
    root = Path(dataset_root) if dataset_root is not None else DEFAULT_DATASET_ROOT
    cape_keydesign.baseURL = str(root)

    edgelist, node_data, node_subjects, node_ids = load_graph_data(
        project, root
    )
    if len(load_token_sequences(project, dataset_root)) != len(node_subjects):
        raise ValueError("Token rows and Process-Binary rows differ for {}".format(project))
    if np.max(np.concatenate([train_idx, test_idx])) >= len(node_subjects):
        raise IndexError("Fold index is outside the project data")
    if np.unique(np.concatenate([train_idx, test_idx])).size != len(node_subjects):
        raise ValueError("Training and test indices must partition all project nodes")

    _, _, _, embedding_matrix, _ = (
        build_fold_semantic_inputs(
            project,
            train_idx,
            test_idx,
            seed,
            dataset_root=root,
        )
    )
    train_inputs, _, _, _, cape_all_data = cape_keydesign.load_CNN_data(
        project, train_idx, test_idx
    )
    train_inputs = np.asarray(train_inputs, dtype=np.int32)
    all_inputs = np.asarray(cape_all_data[0], dtype=np.int32)
    sequence_length = all_inputs.shape[1]
    train_node_set = set(node_subjects.index[train_idx])
    initial_attributes = _compute_initial_node_attributes(
        edgelist,
        node_data,
        node_ids,
        train_idx,
        train_node_set,
        seed,
        directed=False,
    )
    node_features = pd.DataFrame(initial_attributes, index=node_ids)

    full_graph = StellarGraph(
        node_features,
        edgelist,
        edge_weight_column="weight",
    )
    train_edges = edgelist[
        edgelist["source"].isin(train_node_set)
        & edgelist["target"].isin(train_node_set)
    ]
    ordered_train_nodes = [node_id for node_id in node_ids if node_id in train_node_set]
    train_graph = StellarGraph(
        node_features.loc[ordered_train_nodes],
        train_edges,
        edge_weight_column="weight",
    )

    train_generator = FullBatchNodeGenerator(train_graph, method="gcn")
    full_generator = FullBatchNodeGenerator(full_graph, method="gcn")
    labels = node_subjects.to_numpy(dtype=int)
    train_targets = to_categorical(labels[train_idx], num_classes=2)
    train_flow = train_generator.flow(
        node_subjects.index[train_idx], train_targets
    )
    # 推理阶段只输入图数据，不使用测试集标签。
    full_flow = full_generator.flow(node_subjects.index)

    train_features, full_features = _train_fusion_and_extract(
        project,
        train_inputs,
        all_inputs,
        embedding_matrix,
        sequence_length,
        train_generator,
        full_generator,
        train_flow,
        full_flow,
        variant,
        epochs,
        patience,
    )
    test_features = full_features[test_idx]
    if train_features.shape[0] != train_idx.size:
        raise AssertionError("Training feature rows do not match training indices")
    if test_features.shape[0] != test_idx.size:
        raise AssertionError("Test feature rows do not match test indices")
    if not np.isfinite(train_features).all() or not np.isfinite(test_features).all():
        raise ValueError("Non-finite values were found in extracted fold features")

    return FoldFeatures(
        train_idx=train_idx.copy(),
        test_idx=test_idx.copy(),
        train_features=train_features,
        test_features=test_features,
    )


if __name__ == "__main__":
    print()
