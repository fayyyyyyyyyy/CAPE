import javalang
from javalang.tree import MethodInvocation, SuperMethodInvocation, MemberReference, SuperMemberReference, PackageDeclaration, InterfaceDeclaration, ClassDeclaration, MethodDeclaration, ConstructorDeclaration, VariableDeclarator, CatchClauseParameter, FormalParameter, TryResource, ReferenceType, BasicType
from javalang.tree import IfStatement, WhileStatement, DoStatement, ForStatement, AssertStatement, BreakStatement, ContinueStatement, ReturnStatement, ThrowStatement, SynchronizedStatement, TryStatement, SwitchStatement, CatchClause, BlockStatement, StatementExpression, ForControl, SwitchStatementCase, EnhancedForControl
from javalang.tree import ClassCreator
from javalang.parser import JavaSyntaxError
import os
from tqdm import tqdm


# ===================================================================================================
project_names = ['ant_main', 'argouml', 'gwtportlets', 'javaclient', 'jedit', 'jgap',
                 'jhotdraw', 'jmeter_core', 'JPMC', 'log4j', 'Mars', 'Maze', 'neuroph',
                 'PDFBox', 'tomcat', 'wro4j', 'Xerces', 'xuml']

extract_original_data_list = [
                            'dataset/软件系统源代码/apache-ant-1.6.1/src/main/org/apache/tools',
                             'dataset/软件系统源代码/ArgoUML-0.9.5-src/src_new/org',
                             'dataset/软件系统源代码/gwt-portlets-0.9.5beta/gwt-portlets',
                             'dataset/软件系统源代码/Javaclient2-2.0/src',
                             'dataset/软件系统源代码/jedit5.1.0source/jEdit',
                             'dataset/软件系统源代码/jgap3.6.3/src/org',
                             'dataset/软件系统源代码/jhotdraw60b1/src/org/jhotdraw',
                             'dataset/软件系统源代码/jakarta-jmeter-2.0.1/src/core/org/apache/jmeter',
                             'dataset/软件系统源代码/JPMC20020123/src',
                             'dataset/软件系统源代码/apache-log4j-2.3-src',
                             'dataset/软件系统源代码/MarsProject_3.06/mars-sim/source',
                             'dataset/软件系统源代码/Maze/maze-solver/trunk/src/maze',
                             'dataset/软件系统源代码/neuroph_2.2/src/org/neuroph',
                             'dataset/软件系统源代码/PDFBox-2.0.7',
                             'dataset/软件系统源代码/apache-tomcat-7.0.10-src',
                             'dataset/软件系统源代码/wro4j-1.6.3',
                             'dataset/软件系统源代码/Xerces-2.11.0',
                             'dataset/软件系统源代码/xuml-compiler-0.4.8'
                             ]
# project_names = ['tomcat']
# extract_original_data_list = ['dataset/软件系统源代码/apache-tomcat-7.0.10-src']
# ===================================================================================================

filepath = '../'
dataset_keyDesign_root = 'dataset_keyDesign_FGCS/'

for project_name, extract_original_data in zip(project_names, extract_original_data_list):
    folder_path = filepath + extract_original_data
    compared_Class_Root = filepath + dataset_keyDesign_root + project_name + "/compared_Class.txt"
    output_file = filepath + dataset_keyDesign_root + project_name + '/repeated_tokens.txt'
    output_cross = filepath + dataset_keyDesign_root + project_name + '/repeated_tokens_cross.txt'

    with open(output_file, 'w') as output, open(output_cross, 'w') as output_file_cross:
        class_names = []
        with open(compared_Class_Root, 'r') as common:
            for line in common:
                class_names.append(line.strip())
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                if filename.endswith('.java'):  
                    file_path = os.path.join(root, filename)
                    with open(file_path, 'r', encoding='gbk', errors='ignore') as file:
                        java_source = file.read()

                        # 解析 Java 源码文件
                        try:
                            tree = javalang.parse.parse(java_source)
                            # print(file_path)
                        except JavaSyntaxError as e:
                            print("Syntax error:", e.description)
                            print(file_path)
                            exit()

                        # 提取特定类型的节点
                        selected_nodes = []
                        for path, node in tree:
                            if isinstance(node, (MethodInvocation, SuperMethodInvocation, MemberReference, SuperMemberReference)):
                                selected_nodes.append(node)
                            elif isinstance(node, (PackageDeclaration, InterfaceDeclaration, ClassDeclaration, MethodDeclaration, ConstructorDeclaration, VariableDeclarator, CatchClauseParameter, FormalParameter, TryResource, ReferenceType, BasicType)):
                                selected_nodes.append(node)
                            elif isinstance(node, (IfStatement, WhileStatement, DoStatement, ForStatement, AssertStatement, BreakStatement, ContinueStatement, ReturnStatement, ThrowStatement, SynchronizedStatement, TryStatement, SwitchStatement, CatchClause, BlockStatement, StatementExpression, ForControl, SwitchStatementCase, EnhancedForControl)):
                                selected_nodes.append(node)
                            elif isinstance(node, ClassCreator):
                                selected_nodes.append(node)

                        s = 0
                        selected_nodes_new = []
                        selected_nodes_cross = []
                        for node in selected_nodes:
                            if isinstance(node, (MethodDeclaration, InterfaceDeclaration, ClassDeclaration, ConstructorDeclaration, CatchClauseParameter, FormalParameter)):
                                selected_nodes_new.append(node.name)  # 只记录节点的 name 属性
                            elif isinstance(node, (MethodInvocation, SuperMethodInvocation)):
                                selected_nodes_new.append((node.member))
                            elif isinstance(node, (AssertStatement)):
                                selected_nodes_new.append(node.value)
                            elif isinstance(node, ClassCreator):
                                selected_nodes_new.append(node.type.name)
                            else:
                                selected_nodes_new.append(type(node).__name__)  # 记录其他节点类型的名称
                            if isinstance(node, PackageDeclaration):
                                package = node.name

                        for node in selected_nodes:
                            selected_nodes_cross.append(type(node).__name__)

                        filename_without_extension = os.path.splitext(filename)[0]
                        result = ".".join([package, filename_without_extension])
                        
                        if result in class_names:
                            output_head = result.ljust(10)
                            output.write(output_head + "\t")
                            tokens = ','.join(str(node) for node in selected_nodes_new)
                            output.write(tokens)
                            output.write("\n")

                            output_file_cross.write(output_head + "\t")
                            tokens_cross = ','.join(str(node) for node in selected_nodes_cross)
                            output_file_cross.write(tokens_cross)
                            output_file_cross.write("\n")

    print(f"\n02 OK for {project_name}!")
    print(f"结果保存到: {output_file}")
    print(f"结果保存到: {output_cross}")
