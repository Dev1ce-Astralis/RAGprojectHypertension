import time
import streamlit as st
from knowledge_base import KnowledgeBaseService

st.title("知识库更新服务")

uploader_file = st.file_uploader(
    "上传新的知识库文件",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=False,
)

if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

if uploader_file is not None:
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size / 1024

    st.subheader(f"文件名称: {file_name}")
    st.write(f"格式: {file_type} | 大小: {file_size:.2f} KB")

    # 根据文件类型选择更新方式                                    // NEW
    file_bytes = uploader_file.getvalue()                        # NEW
    with st.spinner("载入知识库中 ..."):
        time.sleep(1)
        # 调用新的 update_by_file 方法                            // NEW
        result = st.session_state["service"].update_by_file(file_bytes, file_name)  # NEW
        st.write(result)