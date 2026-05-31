import streamlit as st
import time 
from rag import RagService
import config_data as config

# 标题
st.title("智能高血压指南服务系统")
st.divider()


if "message" not in st.session_state:
    st.session_state["message"] = [{"role":"assistant","content":"您好！我是智能高血压指南服务系统，有什么可以帮助您的吗？"}]

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

for msg in st.session_state["message"]:
    st.chat_message(msg["role"]).write(msg["content"])

# 输入框
prompt = st.chat_input()

if prompt:

    # 显示用户输入
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role":"user","content":prompt})

    ai_res_list = []
    with st.spinner("AI thinking..."):
        res_stream = st.session_state["rag"].chain.stream({"input": prompt}, config.session_config)
        # yield

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

    st.chat_message("assistant").write_stream(capture(res_stream, ai_res_list))
    st.session_state["message"].append({"role": "assistant", "content": "".join(ai_res_list)})