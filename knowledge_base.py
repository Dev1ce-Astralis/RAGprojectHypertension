import os
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader  # NEW
from datetime import datetime

def check_md5(md5_str: str):
    '''检查传入的md5字符串是否被处理过
        return True: 已经处理过    False: 没有处理过
    '''
    if not os.path.exists(config.md5_path):
        open(config.md5_path, 'w', encoding='utf-8').close()
        return False
    else:
        for line in open(config.md5_path, 'r', encoding='utf-8').readlines():
            line = line.strip()
            if line == md5_str:
                return True
        return False

def save_md5(md5_str: str):
    '''将传入的md5字符串保存到文件中'''
    with open(config.md5_path, 'a', encoding='utf-8') as f:
        f.write(md5_str + '\n')

def get_string_md5(input_str: str, encoding='utf-8'):
    '''将传入的字符串转成md5字符串'''
    str_bytes = input_str.encode(encoding=encoding)
    md5_obj = hashlib.md5()
    md5_obj.update(str_bytes)
    md5_hex = md5_obj.hexdigest()
    return md5_hex

class KnowledgeBaseService(object):
    def __init__(self):
        os.makedirs(config.persist_directory, exist_ok=True)
        self.chroma = Chroma(
            collection_name=config.collection_name,
            embedding_function=DashScopeEmbeddings(model=config.embedding_model_name),  # 修正：使用配置的模型
            persist_directory=config.persist_directory,
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len
        )

    def update_by_str(self, data, filename):
        """将传入的字符串，进行向量化，存入向量数据库中"""
        md5_hex = get_string_md5(data)
        if check_md5(md5_hex):
            return "[跳过]内容已经在存在的知识库中"
        # 分割并入库（原代码未完成，补全）                                    // NEW
        docs = self.spliter.create_documents([data], metadatas=[{"source": filename}])
        self.chroma.add_documents(docs)
        save_md5(md5_hex)
        return f"[成功]已添加 {filename}，共 {len(docs)} 个文本块"

    def update_by_file(self, file_bytes: bytes, filename: str) -> str:        # NEW
        """根据文件类型加载、分割、向量化并存入数据库                         // NEW
        :param file_bytes: 文件二进制内容                                    // NEW
        :param filename: 文件名（含扩展名）                                  // NEW
        :return: 处理结果字符串                                              // NEW
        """                                                                  # NEW
        # 计算 MD5（基于文件内容）                                           // NEW
        md5_hex = hashlib.md5(file_bytes).hexdigest()                       # NEW
        if check_md5(md5_hex):                                              # NEW
            return "[跳过]内容已经在存在的知识库中"                          # NEW
                                                                             # NEW
        # 临时保存文件以供加载器使用                                         // NEW
        temp_dir = "./temp_uploads"                                         # NEW
        os.makedirs(temp_dir, exist_ok=True)                                 # NEW
        temp_path = os.path.join(temp_dir, filename)                         # NEW
        with open(temp_path, 'wb') as f:                                     # NEW
            f.write(file_bytes)                                              # NEW
                                                                             # NEW
        try:                                                                 # NEW
            ext = os.path.splitext(filename)[1].lower()                      # NEW
            if ext == '.pdf':                                                # NEW
                loader = PyPDFLoader(temp_path)                              # NEW
            elif ext == '.txt':                                              # NEW
                loader = TextLoader(temp_path, encoding='utf-8')             # NEW
            elif ext == '.docx':                                             # NEW
                # 可添加 Docx2txtLoader，此处暂不实现                         # NEW
                raise ValueError("暂不支持 .docx 格式，请使用 TXT 或 PDF")    # NEW
            else:                                                            # NEW
                raise ValueError(f"不支持的文件格式: {ext}")                  # NEW
                                                                             # NEW
            documents = loader.load()                                        # NEW
            # 为每个文档添加 source 元数据                                    # NEW
            for doc in documents:                                            # NEW
                doc.metadata["source"] = filename                            # NEW
                                                                             # NEW
            # 分割文档                                                       # NEW
            docs = self.spliter.split_documents(documents)                   # NEW
            self.chroma.add_documents(docs)                                  # NEW
            save_md5(md5_hex)                                                # NEW
            return f"[成功]已添加 {filename}，共 {len(docs)} 个文本块"         # NEW
        finally:                                                             # NEW
            # 清理临时文件                                                    # NEW
            if os.path.exists(temp_path):                                    # NEW
                os.remove(temp_path)                                         # NEW