import os

project_names = ['ant_main', 'argouml', 'gwtportlets', 'javaclient', 'jedit', 'jgap',
                 'jhotdraw', 'jmeter_core', 'JPMC', 'log4j', 'Mars', 'Maze', 'neuroph',
                 'PDFBox', 'tomcat', 'wro4j', 'Xerces', 'xuml']
filepath = '../'
dataset_keyDesign_root = 'dataset_keyDesign_FGCS/'
for project_name in project_names:
    file_path = filepath + dataset_keyDesign_root + project_name + '/nodes.txt'
    map_file = filepath + dataset_keyDesign_root + project_name + '/original_edges_weight.txt'
    output_file = filepath + dataset_keyDesign_root + project_name + '/edges_weight.txt'

    output_folder = os.path.dirname(output_file)
    os.makedirs(output_folder, exist_ok=True)

    nodes_dict = {}
    with open(file_path, 'r') as nodes_file:
        for line in nodes_file:
            line = line.strip().split('\t')
            nodes_dict[line[1]] = int(line[0])  # 将类名作为键，数字作为值

    with open(map_file, 'r') as ant_file, open(output_file, 'w') as edges_file:
        for line in ant_file:
            class1, class2, weight = line.strip().split('\t')
            if class1 in nodes_dict and class2 in nodes_dict:
                class1_id = nodes_dict[class1]
                class2_id = nodes_dict[class2]
                edges_file.write(f"{class1_id}\t{class2_id}\t{weight}\n")
                print(f"{class1_id}\t{class2_id}\t{weight}")

    print(f"\n05 OK for {project_name}!")
    print(f"结果保存到: {output_file}")
