import os
project_names = ['ant_main', 'argouml', 'gwtportlets', 'javaclient', 'jedit', 'jgap',
                 'jhotdraw', 'jmeter_core', 'JPMC', 'log4j', 'Mars', 'Maze', 'neuroph',
                 'PDFBox', 'tomcat', 'wro4j', 'Xerces', 'xuml']
filepath = '../'
dataset_keyDesign_root = 'dataset_keyDesign_FGCS/'
for project_name in project_names:
    tokens_file = filepath + dataset_keyDesign_root + project_name + '/tokens.txt'
    vocab_emb_dict_30_file = filepath + dataset_keyDesign_root + project_name + '/vocab_emb_dict_30.txt'
    tokens_map_file = filepath + dataset_keyDesign_root + project_name + '/tokens_map.txt'

    output_folder = os.path.dirname(vocab_emb_dict_30_file)
    os.makedirs(output_folder, exist_ok=True)

    # 统计节点属性出现次数并排序
    word_count = {}
    with open(tokens_file, 'r') as file:
        for line in file:
            parts = line.split('\t')
            if len(parts) > 1:
                second_part = parts[1].strip().split(',')
                for word in second_part:
                    if word in word_count:
                        word_count[word] += 1
                    else:
                        word_count[word] = 1

    sorted_word_count = {k: v for k, v in sorted(word_count.items(), key=lambda item: item[1], reverse=True)}

    i = 0
    with open(vocab_emb_dict_30_file, 'w') as edges_file:
        for word, count in sorted_word_count.items():
            if count > 2:
                edges_file.write(f"{i}\t{word}\n")
                i = i + 1

    nodes_dict = {}
    with open(vocab_emb_dict_30_file, 'r') as nodes_file:
        for line in nodes_file:
            line = line.strip().split('\t')
            nodes_dict[line[1]] = int(line[0])

    with open(tokens_file, 'r') as ant_file, open(tokens_map_file, 'w') as tokens_map:
        i = 0
        for line in ant_file:
            flag, tokens = line.strip().split('\t')
            token = tokens.strip().split(',')
            tokens_map.write(f"{i}\t")

            for t in token:
                if t in nodes_dict:
                    tokens_map.write(f"{nodes_dict[t]} ")
            i = i + 1
            tokens_map.write(f"\n")

    print(f"\n06 OK for {project_name}!")
    print(f"结果保存到: {vocab_emb_dict_30_file}")
    print(f"结果保存到: {tokens_map_file}")
