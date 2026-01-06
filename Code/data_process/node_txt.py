project_names = ['ant_main','argouml', 'gwtportlets', 'javaclient', 'jedit', 'jgap',
                 'jhotdraw', 'jmeter_core', 'JPMC', 'log4j', 'Mars', 'Maze', 'neuroph',
                 'PDFBox', 'tomcat', 'wro4j', 'Xerces', 'xuml']

filepath = '../'
dataset_keyDesign_root = 'dataset_keyDesign_FGCS/'

for project_name in project_names:
    file_path = filepath + dataset_keyDesign_root + project_name + '/tokens.txt'
    output_file_path = filepath + dataset_keyDesign_root + project_name + '/nodes.txt'

    with open(file_path, 'r') as file, open(output_file_path, 'w') as output_file:
        for line_number, line in enumerate(file):
            parts = line.strip().split('\t', 1)  # 使用制表符（tab）分割行的第一个部分
            first_part = parts[0]

            output_line = f"{line_number}\t{first_part}\n"
            output_file.write(output_line)

    print(f"\n03-OK for {project_name}!")
    print(f"结果保存到: {output_file_path}")
