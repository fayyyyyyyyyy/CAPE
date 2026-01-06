import pandas as pd

project_names = ['ant_main', 'argouml', 'gwtportlets', 'javaclient', 'jedit', 'jgap',
                 'jhotdraw', 'jmeter_core', 'JPMC', 'log4j', 'Mars', 'Maze', 'neuroph',
                 'PDFBox', 'tomcat', 'wro4j', 'Xerces', 'xuml']
filepath = '../'
dataset_keyDesign_root = 'dataset_keyDesign_FGCS/'
for project_name in project_names:
    vocab_emb_dict_30_file = filepath + dataset_keyDesign_root + project_name + '/vocab_emb_dict_30.txt'
    vocab_emb_dict_30_file_csv = filepath + dataset_keyDesign_root + project_name + '/vocab_emb_dict_30.csv'

    data = []
    with open(vocab_emb_dict_30_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_name = ' '.join(parts[1:])  # 类名可能包含空格，需要合并
            data.append([class_name])

    df = pd.DataFrame(data, columns=['name'])

    for i in range(30):
        df[f'vector_{i}'] = None

    df.to_csv(vocab_emb_dict_30_file_csv, index=False)

    print(f"\n08 OK for {project_name}!")
    print(f"结果保存到: {vocab_emb_dict_30_file_csv}")
