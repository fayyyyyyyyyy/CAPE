import numpy as np
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, concatenate, Flatten, Dropout, Dense
from tensorflow.keras.models import Model
from sklearn import metrics
from tensorflow.keras.utils import to_categorical
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from stellargraph.layer import GCN
import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from utils.MyLabelBinarizer import *
from utils.MyEvaluate import *
from IPython.display import display, HTML
import tensorflow as tf
from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec
from sklearn import preprocessing
import os

# Set random seed
seed = 123
tf.random.set_seed(seed)

######################################################################################################################
# baseURL = "./downstream_task/data/"
baseURL = "./dataset_keyDesign_Sora"
######################################################################################################################

def TextCNN_model(main_input, embedding_matrix, project):
    main_input_ = tf.squeeze(main_input, axis=0)
    # 创建一个嵌入层，用于将整数序列转换为密集向量的嵌入。
    embedding = Embedding(input_dim=embedding_matrix.shape[0],
                         output_dim=embedding_matrix.shape[1],
                         input_length=get_avg_len(project),
                         weights=[embedding_matrix],  
                         trainable=False)   # 表示嵌入层的权重在训练过程中不可更新
    # 将输入数据传递给嵌入层以获取嵌入表示
    embed = embedding(main_input_)
    # padding='same' 表示使用填充保持输出和输入具有相同的长度。
    cnn = Conv1D(filters=10, kernel_size=5, padding='same', strides=1, activation='relu')(embed)
    cnn = MaxPooling1D(pool_size=int(cnn.shape[1]))(cnn)
    # 创建具有32个神经元的全连接层Dense
    # 这个全连接层的作用是将池化层提取的特征进行线性组合和变换，输出一个32维的特征表示
    hidden = Dense(32, activation='relu')(cnn)
    flat = Flatten()(hidden)
    # drop = Dropout(0.1)(flat) 
    output = tf.expand_dims(flat, axis=0)
    # 首先定义好网络，再将网络的输入和输出部分作为参数定义一个Model类对象。
    model = Model(inputs=main_input, outputs=output)
    return model

def load_embedding_matrix(path):
    # header=0 表示第一行是列名
    embed_matrix_file = pd.read_csv(path, header=0, index_col=False)
    # iloc 方法选择所有行和从第二列开始到最后一列的所有列，然后将其转换为 NumPy 数组。
    embed_matrix = np.array(embed_matrix_file.iloc[:, 1:])
    return embed_matrix

def get_avg_len(project):
    tokens_integer = []
    len_list = []
    tokens_integer_file = open(baseURL + "/" + project + "/tokens_map.txt", 'r')
    lines = tokens_integer_file.readlines()
    for each_line in lines:
        integer = each_line[each_line.index('\t') + 1:].strip('\n')
        integer_list = integer.split(' ')
        tokens_integer.append(integer_list[:-1])
        len_list.append(len(integer_list[:-1]))
    max_len = max(len_list)
    return max_len

def sequence(tokens_seq,project):
    # pad_sequences 方法：一个常见的序列处理工具，可以用来处理文本或者其他序列数据，确保它们具有相同的长度。(后向填充或后向截断)
    return pad_sequences(tokens_seq, maxlen=get_avg_len(project),padding='post',truncating='post')

def load_CNN_data(project, train_index, test_index):
# 1、读取 tokens_map 文件并处理数据：
    tokens_map = []
    tokens_map_file = open(baseURL +  "/" + project + "/tokens_map.txt", 'r')
    
    #  逐行读取文件内容
    lines = tokens_map_file.readlines()
    for each_line in lines:
        # 获取每一行中以制表符分隔的部分（获取制表符后的整数序列字符串，最后去除可能存在的末尾换行符）
        integer = each_line[each_line.index('\t') + 1:].strip('\n')
        # 将整数序列拆分为单独的整数，并存储在 integer_list 中。
        integer_list = integer.split(' ')
        # 将处理后的整数序列添加到 tokens_map 列表中，但是排除了最后一个元素。
        tokens_map.append(integer_list[:-1])

# 2、处理原始数据并进行序列填充和标签编码：
    # 定义了一个处理函数，它对数据进行序列填充和标签编码。返回一个元组
    process = lambda data,label: (sequence(data, project), to_categorical(label))
    origin_data = pd.read_csv(baseURL + "/" + project + "/Process-Binary.csv",header=0,index_col=False)
    # 将tokens_map(data)和origin_data['bug'](label) 进行处理(使用 process 函数)
    Alldata = process(tokens_map, origin_data['KeyDesign'])

# 3、准备训练和测试数据集：
    # 获取处理后的数据作为特征 X。
    X = Alldata[0]
    # 对处理后的标签进行独热编码，并将其作为目标标签 y。
    y = np.argmax(Alldata[1], axis=1)

# 4、划分训练集和测试集：
    # 将特征数据划分为训练集和测试集。
    X_train, X_test = X[train_index], X[test_index]
    # 将目标标签划分为训练集和测试集。
    y_train, y_test = y[train_index], y[test_index]
# 5、返回训练和测试数据集以及处理后的完整数据：
    return X_train, y_train, X_test, y_test, Alldata

def _compute_initial_node_attributes(edgelist, node_data, node_ids, id_col=None, directed=False, train_node_set=None, walk_params=None, w2v_params=None):
    cls = sg.StellarDiGraph if directed else sg.StellarGraph
    edgelist = edgelist[["source", "target", "weight"]].copy()

    # ---------- 1. 在训练子图上训练 Word2Vec ----------
    if train_node_set is not None:
        train_mask = edgelist["source"].isin(train_node_set) & edgelist["target"].isin(train_node_set)
        edgelist_train = edgelist[train_mask].copy()
    else:
        edgelist_train = edgelist.copy()
    G_train = cls(edges=edgelist_train, edge_weight_column="weight")

    walk_params = walk_params or {}
    p = walk_params.get("p", 0.25)
    q = walk_params.get("q", 1)
    n = walk_params.get("n", 10)
    length = walk_params.get("length", 80)
    seed = walk_params.get("seed", 42)
    weighted = walk_params.get("weighted", True)

    rw = BiasedRandomWalk(G_train, p=p, q=q, n=n, length=length, seed=seed, weighted=weighted)
    walks = rw.run(nodes=list(G_train.nodes()))
    str_walks = [[str(n) for n in walk] for walk in walks]

    w2v_params = w2v_params or {}
    w2v_defaults = dict(vector_size=128, window=10, min_count=1, sg=1, workers=4, epochs=20)
    w2v_defaults.update(w2v_params)
    model = Word2Vec(str_walks, **w2v_defaults)

    # ---------- 2. 构建全图邻接表 ----------
    full_adj = {}
    for _, row in edgelist.iterrows():
        s = row["source"]
        t = row["target"]
        full_adj.setdefault(s, []).append(t)
        full_adj.setdefault(t, []).append(s)

    vec_size = model.wv.vector_size

    node_to_vec = {}
    for n in node_ids:
        key = str(n)
        if key in model.wv:
            node_to_vec[n] = model.wv[key].copy()
        else:
            node_to_vec[n] = np.zeros(vec_size, dtype=float)

    num_iters = 3
    for _ in range(num_iters):
        new_vecs = {}
        for n in node_ids:
            if n in train_node_set:
                continue
            neighbors = full_adj.get(n, [])
            if len(neighbors) == 0:
                new_vecs[n] = np.zeros(vec_size, dtype=float)
            else:
                neigh_vectors = [node_to_vec[nb] for nb in neighbors if nb in node_to_vec]
                if len(neigh_vectors) == 0:
                    new_vecs[n] = np.zeros(vec_size, dtype=float)
                else:
                    new_vecs[n] = np.mean(neigh_vectors, axis=0)
        for n, vec in new_vecs.items():
            node_to_vec[n] = vec

    # 按 node_ids 顺序收集最终向量
    node_embeddings = np.array([node_to_vec[n] for n in node_ids])

    features = node_data.iloc[:, 3:-1]
    if train_node_set is not None and id_col is not None:
        features_train = node_data[node_data[id_col].isin(list(train_node_set))].iloc[:, 3:-1]
    else:
        features_train = features
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(features_train)
    features_std = scaler.transform(features)

    if features_std.shape[0] > node_embeddings.shape[0]:
        num_rows_to_pad = features_std.shape[0] - node_embeddings.shape[0]
        node_embeddings_padded = np.pad(node_embeddings, ((0, num_rows_to_pad), (0, 0)), mode='constant')
        initial_node_attributes = np.concatenate([features_std, node_embeddings_padded], axis=1)
    else:
        num_rows_to_pad = node_embeddings.shape[0] - features_std.shape[0]
        features_std_padded = np.pad(features_std, ((0, num_rows_to_pad), (0, 0)), mode='constant')
        initial_node_attributes = np.concatenate([features_std_padded, node_embeddings], axis=1)

    return initial_node_attributes


def load_GCN_data(project):
    base_dir = os.path.join(baseURL, project)
    edges_path = os.path.join(base_dir, "edges_weight.txt")
    process_binary_path = os.path.join(base_dir, "Process-Binary.csv")

    edgelist = pd.read_csv(
        edges_path,
        sep="\t",
        header=None,
        names=["source", "target", "weight"],
        usecols=[0, 1, 2],
    )
    node_data = pd.read_csv(process_binary_path)
    node_data.apply(pd.to_numeric, errors='ignore')

    subjects_num = node_data['KeyDesign']
    label_list = subjects_num.to_list()
    labels = ['KeyDesign' if x == 1 else 'nonKeyDesign' for x in label_list]

    id_col = None
    for col in node_data.columns:
        if col.lower() in ["id", "node_id", "nodeid"]:
            id_col = col
            break
    if id_col is not None:
        node_ids = list(node_data[id_col])
    else:
        node_ids = list(node_data.index)

    node_subjects = pd.Series(labels, index=node_ids, dtype='str')
    return edgelist, node_data, node_subjects, node_ids, id_col

def mix(inp1, inp2, r1, r2, Alldata, X_train, X_test, gen_full, train_gen, test_gen, project, fold):
    alpha = 0.5
    concat = tf.keras.layers.Add()([(1 - alpha) * r1, alpha * r2])

    dense1 = Dense(32, activation='relu')(concat)
    prediction = Dense(2, activation='softmax', name="pred_Layer")(dense1)

    merged_model = Model(inputs=[inp1, inp2], outputs=prediction)

    merged_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=['binary_crossentropy'],
                         metrics=['accuracy'])

    learning_rate_reduction = ReduceLROnPlateau(monitor='loss',
                                                patience=50,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)

    early_stopping = EarlyStopping(monitor='loss', patience=50)

    train_in = np.expand_dims(X_train, axis=0)
    data_in = np.expand_dims(Alldata[0], axis=0)

    history = merged_model.fit(x=[train_in, train_gen.inputs], y=train_gen.targets,
                               epochs=500, verbose=2,
                               callbacks=[learning_rate_reduction])

    embedding_model = tf.keras.Model(inputs=[inp1, inp2], outputs=dense1)

    emb = embedding_model.predict([data_in, gen_full.inputs])
    X_fold_features = emb.squeeze(0)

    return X_fold_features


if __name__ == '__main__':
    projects = ['ant_main', 'argouml', 'gwtportlets', 'javaclient', 'jedit', 'jgap',
            'jhotdraw', 'jmeter_core', 'JPMC', 'log4j', 'Mars', 'Maze', 
            'neuroph','PDFBox', 'tomcat', 'wro4j', 'Xerces', 'xuml']

    seed = 123
    tf.random.set_seed(seed)

    for current_project in projects:
        print(f"\n==================== Start Project: {current_project} ====================")

        edgelist, node_data, node_subjects, node_ids, id_col = load_GCN_data(current_project)
        embedding_matrix = load_embedding_matrix(
            baseURL + "/" + current_project + "/" + "vocab_emb_dict_30.csv")

        final_project_embeddings = np.zeros((len(node_subjects), 32))

        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)

        for fold, (train_idx, test_idx) in enumerate(skf.split(node_subjects, node_subjects)):
            print(f"\n>>> Running Fold {fold + 1}/2 for {current_project}...")

            train_nodes_set = set(node_subjects.index[train_idx])
            test_nodes_set = set(node_subjects.index[test_idx])

            initial_node_attributes = _compute_initial_node_attributes(
                edgelist,
                node_data,
                node_ids,
                id_col=id_col,
                directed=False,
                train_node_set=train_nodes_set,
            )
            node_features = pd.DataFrame(initial_node_attributes, index=node_ids)

            G = sg.StellarGraph(node_features, edgelist)
            filtered_edges = edgelist[edgelist['source'].isin(train_nodes_set) & edgelist['target'].isin(train_nodes_set)]
            G_train = sg.StellarGraph(node_features, filtered_edges)

            generator_train = FullBatchNodeGenerator(G_train, method="gcn")
            generator_full = FullBatchNodeGenerator(G, method="gcn")

            target_encoding = MyLabelBinarizer()
            node_targets = target_encoding.fit_transform(node_subjects)

            train_subjects_fold = node_subjects.iloc[train_idx]
            test_subjects_fold = node_subjects.iloc[test_idx]
            train_targets_fold = target_encoding.transform(train_subjects_fold)
            test_targets_fold = target_encoding.transform(test_subjects_fold)

            train_gen = generator_train.flow(train_subjects_fold.index, train_targets_fold)
            test_gen = generator_full.flow(test_subjects_fold.index, test_targets_fold)
            gen_full = generator_full.flow(node_subjects.index, node_targets)

            X_train, y_train, X_test, y_test, Alldata = load_CNN_data(current_project, train_idx, test_idx)

            inpCNN = Input(batch_shape=(1, None, get_avg_len(current_project)), dtype='float64', name='cnn_input')
            modelCNN = TextCNN_model(inpCNN, embedding_matrix, current_project)
            r1 = modelCNN.output

            gcn = GCN(layer_sizes=[32, 32], activations=["relu", "relu"], generator=generator_train, dropout=0.0)
            x_inp_train, x_out_train = gcn.in_out_tensors()

            fold_features = mix(inpCNN, x_inp_train, r1, x_out_train, Alldata,
                                X_train, X_test, gen_full, train_gen, test_gen, current_project, fold)

            final_project_embeddings[test_idx] = fold_features[test_idx]

        df = pd.DataFrame(final_project_embeddings, columns=[('emb_' + str(i)) for i in range(final_project_embeddings.shape[1])])
        df.to_csv(baseURL + "/" + current_project + "/" + "hyq_emb_AN_Sora_non-dw.csv", index=False)
        print(f"Project {current_project} embeddings saved to hyq_emb_AN_Sora_non-dw.csv")
