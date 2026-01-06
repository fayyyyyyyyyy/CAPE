import csv
import os

def remove_columns(input_csv_path, output_csv_path, columns_to_remove):
    with open(input_csv_path, 'r', newline='') as infile:
        reader = csv.reader(infile)
        data = list(reader)

    header = data[0]

    for column in columns_to_remove:
        if column.strip() not in header:
            print(f"Warning: Column '{column}' not found in the CSV file.")

    new_header = [col for col in header if col.strip() not in columns_to_remove]

    columns_to_remove_indexes = [header.index(col.strip()) for col in columns_to_remove]

    new_data = [new_header] + [[row[i] for i in range(len(row)) if i not in columns_to_remove_indexes] for row in data[1:]]

    with open(output_csv_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(new_data)

def read_classes_from_file(file_path):
    classes = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                classes.append(parts[1])  # 获取第二列作为类名
    print(classes)
    return classes

def find_class_positions(class_names, csv_file):
    positions = []
    first_column_data=[]
    if not os.path.exists(csv_file):
        os.makedirs(csv_file)
    with open(csv_file, 'r', newline='') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)

        class_mapping = {}
        class_index = 1
        for row in csv_reader:
            class_name = row[0]
            if class_name not in class_mapping:
                class_mapping[class_name] = class_index
                class_index += 1
        print(class_mapping)

        for class_name in class_names:
            if class_name in class_mapping:
                class_index = class_mapping[class_name]
                positions.append(class_index)
        print(positions)
    return positions

project_names = [
    'ant_main', 'argouml', 'gwtportlets', 'javaclient', 'jedit', 'jgap',
    'jhotdraw', 'jmeter_core', 'JPMC', 'log4j', 'Mars', 'Maze', 'neuroph',
    'PDFBox', 'tomcat', 'wro4j', 'Xerces', 'xuml']

filepath = '../workOfData/'
dataset_keyDesign_root = 'dataset_keyDesign_FGCS/'
columns_to_remove = []

for project_name in project_names:
    node_txt_path = filepath + dataset_keyDesign_root + project_name + '/nodes.txt'
    traditional_metrics_csv = f'../datasets_0528/FGCS/{project_name}_DM+NM+INM.csv'
    output_traditional_metrics_csv = filepath + dataset_keyDesign_root + project_name + '/' + project_name + '.csv'
    processBinary_csv = filepath + dataset_keyDesign_root + project_name + '/Process-Binary.csv'

    remove_columns(traditional_metrics_csv, output_traditional_metrics_csv, columns_to_remove)
    class_names = read_classes_from_file(node_txt_path)
    class_positions = find_class_positions(class_names, output_traditional_metrics_csv)

    with open(output_traditional_metrics_csv, 'r', newline='') as infile:
        reader = csv.reader(infile)
        data = list(reader)

        header = data[0]

        reordered_data = [header] + [data[i] for i in class_positions]

        header.insert(0, "name")
        header.insert(1, "version")

        for row in reordered_data[1:]:  # 不跳过列名行
            if int(row[-1]) not in [0, 1]:
                row[-1] = '1'
            row.insert(0, project_name)  # 添加 'ant' 到第一列
            row.insert(1, '0.0')  # 添加 '1.4' 到第二列

        with open(processBinary_csv, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(reordered_data)

    print(f"\n12 OK for {project_name}!")
    print(f"结果保存到: {output_traditional_metrics_csv}")
    print(f"结果保存到: {processBinary_csv}")
