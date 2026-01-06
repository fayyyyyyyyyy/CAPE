import csv
import os

project_names = ['ant_main', 'argouml', 'gwtportlets', 'javaclient', 'jedit', 'jgap',
                 'jhotdraw', 'jmeter_core', 'JPMC', 'log4j', 'Mars', 'Maze', 'neuroph',
                 'PDFBox', 'tomcat', 'wro4j', 'Xerces', 'xuml']

filepath = '../'
dataset_keyDesign_root = 'dataset_keyDesign_FGCS/'

for project_name in project_names:
    csv_data = f'../datasets_0528/FGCS/{project_name}_DM+NM+INM.csv'
    output_root = filepath + dataset_keyDesign_root + project_name + '/compared_Class.txt'

    output_folder = os.path.dirname(output_root)
    os.makedirs(output_folder, exist_ok=True)

    names = []
    with open(csv_data, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过列名
        for row in reader:
            names.append(row[0])

    names.sort()

    with open(output_root, 'w') as samew:
        for name in names:
            samew.write(name)
            samew.write("\n")

    print(f"\n 01 OK for {project_name}!")
    print(f"结果保存到: {output_root}")
