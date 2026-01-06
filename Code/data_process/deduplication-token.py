from tqdm import tqdm

def filter_tokens(input_file, class_file, output_file):
    with open(class_file, 'r') as f:
        class_names = set(line.strip() for line in f)

    with open(input_file, 'r') as f:
        lines = f.readlines()

    class_counts = {}

    filtered_lines = []
    for line in lines:
        class_name = line.split()[0]

        if class_name in class_names and class_counts.get(class_name, 0) == 0:
            filtered_lines.append(line)
            class_counts[class_name] = 1
        else:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    with open(output_file, 'w') as f:
        f.writelines(filtered_lines)



if __name__ == "__main__":

    project_names = ['ant_main', 'argouml', 'gwtportlets', 'javaclient', 'jedit', 'jgap',
                     'jhotdraw', 'jmeter_core', 'JPMC', 'log4j', 'Mars', 'Maze', 'neuroph',
                     'PDFBox', 'tomcat', 'wro4j', 'Xerces', 'xuml']
    filepath = '../'
    dataset_keyDesign_root = 'dataset_keyDesign_FGCS/'

    for project_name in project_names:

        compared_class_file = filepath + dataset_keyDesign_root + project_name + "/compared_Class.txt"
        input_file = filepath + dataset_keyDesign_root + project_name + "/repeated_tokens.txt"
        output_file = filepath + dataset_keyDesign_root + project_name + "/tokens.txt"
        input_cross_file = filepath + dataset_keyDesign_root + project_name + "/repeated_tokens_cross.txt"
        output_cross_file = filepath + dataset_keyDesign_root + project_name + "/tokens_cross.txt"

        filter_tokens(input_file, compared_class_file, output_file)
        filter_tokens(input_cross_file, compared_class_file, output_cross_file)

        print(f"\n002 OK for {project_name}!")