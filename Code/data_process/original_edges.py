import os

project_names = ['ant_main', 'argouml', 'gwtportlets', 'javaclient', 'jedit', 'jgap',
                 'jhotdraw', 'jmeter_core', 'JPMC', 'log4j', 'Mars', 'Maze', 'neuroph',
                 'PDFBox', 'tomcat', 'wro4j', 'Xerces', 'xuml']

filepath = '../'
dataset_keyDesign_root = 'dataset_keyDesign_FGCS/'

for project_name in project_names:
    input_file = f'../dataset/软件网络图/AllSoftNets_FGCS/CCN_SoftNet_FGCS_{project_name}.net'
    output_file = filepath + dataset_keyDesign_root + project_name + '/original_edges_weight.txt'

    output_folder = os.path.dirname(output_file)
    os.makedirs(output_folder, exist_ok=True)

    vertices = {}
    edges = []

    with open(input_file, 'r') as file:
        lines = file.readlines()

    in_vertices = False
    in_arcs = False

    for line in lines:
        line = line.strip()

        if line.startswith('*Vertices'):
            in_vertices = True
            in_arcs = False
            continue
        elif line.startswith('*Arcs'):
            in_arcs = True
            in_vertices = False
            continue

        if in_vertices:
            index, rest = line.split(' ', 1)
            vertices[index] = rest.strip('"')
        elif in_arcs:
            src, dest, weight = line.split()
            src_node = vertices.get(src, None)
            dest_node = vertices.get(dest, None)
            if src_node is not None and dest_node is not None:
                edges.append((src_node, dest_node, weight))

    with open(output_file, 'w') as file:
        for edge in edges:
            file.write(f"{edge[0]}\t{edge[1]}\t{edge[2]}\n")

    print(f"\n04 OK for {project_name}!")
    print(f"结果保存到: {output_file}")
