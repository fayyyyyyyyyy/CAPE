import pandas as pd
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
filepath = '../'
dataset_keyDesign_root = 'dataset_keyDesign_FGCS/'
def process_vocab_emb_dict_file(project_name):
    input_file = filepath + dataset_keyDesign_root + project_name + '/vocab_emb_dict_30.txt'
    output_file = filepath + dataset_keyDesign_root + project_name + '/vocab_emb_dict_30.csv'

    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    words = [line.split()[1] for line in lines]

    class EpochLogger(CallbackAny2Vec):
        def __init__(self):
            self.epoch = 0

        def on_epoch_end(self, model):
            self.epoch += 1
            print("Epoch #{} end".format(self.epoch))

    epoch_logger = EpochLogger()

    model = Word2Vec(sentences=[words], vector_size=30, window=5, min_count=1, workers=4, sg=1, callbacks=[epoch_logger])

    word_embeddings = []
    for word in words:
        embedding = model.wv[word].tolist()
        word_embeddings.append([word] + embedding)

    columns = ['name'] + [f'vector_{i}' for i in range(30)]
    df = pd.DataFrame(word_embeddings, columns=columns)
    df.to_csv(output_file, index=False)

    print(f"\n10 OK for {project_name}!")
    print(f"结果保存到: {output_file}")

project_names = ['ant_main', 'argouml', 'gwtportlets', 'javaclient', 'jedit', 'jgap',
                 'jhotdraw', 'jmeter_core', 'JPMC', 'log4j', 'Mars', 'Maze', 'neuroph',
                 'PDFBox', 'tomcat', 'wro4j', 'Xerces', 'xuml']

for project_name in project_names:
    process_vocab_emb_dict_file(project_name)
