import json
from langchain.document_loaders import UnstructuredFileLoader, UnstructuredMarkdownLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import os

# 获取文件路径函数
def get_files(dir_path):
    file_list = []
    for filepath, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            if filename.endswith(".md") or filename.endswith(".txt") or filename.endswith(".json") or filename.endswith(".jsonl"):
                file_list.append(os.path.join(filepath, filename))
    return file_list

# 加载文件函数
def get_text(dir_path):
    file_lst = get_files(dir_path)
    docs = []
    print(f'Found {len(file_lst)} files in directory {dir_path}')
    for one_file in tqdm(file_lst):
        file_type = one_file.split('.')[-1]
        print(f'Processing file: {one_file}')
        try:
            if file_type == 'md':
                loader = UnstructuredMarkdownLoader(one_file)
                docs.extend(loader.load())
            elif file_type == 'txt':
                loader = UnstructuredFileLoader(one_file)
                docs.extend(loader.load())
            elif file_type == 'json' or file_type == 'jsonl':
                with open(one_file, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        print(f'File: {one_file} loaded successfully')
                        # 打印出部分数据以确认内容
                        print(json.dumps(data[:2], indent=2, ensure_ascii=False))  # 仅打印前两个元素
                        for item in data['conversation']:
                            docs.append({'content': item['system']})
                            docs.append({'content': item['input']})
                            docs.append({'content': item['output']})
                    except json.JSONDecodeError as e:
                        print(f'JSONDecodeError: {e} in file {one_file}')
            else:
                print(f'不符合条件的文件: {one_file}')
        except Exception as e:
            print(f'Error: {e} in file {one_file}')
    return docs

# 目标文件夹
tar_dir = [
    # "/group_share/Ancient_Books/dataset/chinese-poetry",
    "/group_share/Ancient_Books/dataset/data",
    # "/group_share/Ancient_Books/dataset/Classical-Modern/双语数据",
]

# 加载目标文件
docs = []
for dir_path in tar_dir:
    docs.extend(get_text(dir_path))

# 检查是否有文档加载成功
if not docs:
    print("没有加载到任何文档。请检查目标文件夹和文件格式。")
    exit()

# 对文本进行分块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
split_docs = text_splitter.split_documents(docs)

# 检查分块结果
if not split_docs:
    print("没有分块到任何文档内容。请检查文本分块设置。")
    exit()

# 加载开源词向量模型
embeddings = HuggingFaceEmbeddings(model_name="/group_share/Ancient_Books/model/sentence-transformer")

# 定义持久化路径
persist_directory = '/group_share/Ancient_Books/dataset/vector_db/chroma'
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# 构建和持久化向量数据库
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory=persist_directory
)
vectordb.persist()

print("向量数据库已构建并持久化完成")
