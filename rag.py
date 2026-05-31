from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from file_history_store import get_history
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableWithMessageHistory



class RagService(object):
    def __init__(self):
        self.vector_store = VectorStoreService(
            embeddings=DashScopeEmbeddings(model=config.embedding_model_name)
        )

        # 修改 Prompt：强制不确定时不回答，并引用来源                    // NEW
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的已知参考资料为主，"
                 "简洁和专业的回答用户的问题。参考资料如下：\n{context}"),
                ("system", "并且我提供用户的历史纪录，如下"),
                MessagesPlaceholder("history"),
                ("system", '重要规则：若参考资料无法支撑准确回答，请明确回答'
                 '“我无法根据现有资料回答该问题”，不要编造答案。'
                 '若回答，请在回答结尾引用具体来源的元信息（如文件名、页码等）。'),  # NEW
                ("user", " 请回答用户提问：{input}"),
            ]
        )

        self.chat_model = ChatTongyi(model=config.chat_model_name)

        self.chain = self.__get_chain()

    def __get_chain(self):
        '''获取最终的执行链'''
         # 手动从向量库获取所有文档
        raw_data = self.vector_store.vector_store.get()
        all_documents = raw_data.get('documents', [])
        retriever = self.vector_store.get_retriever(documents=all_documents)

        def format_documents(docs: list[Document]):
            if not docs:
                return "无相关参考资料"

            formatted_str = ""
            for doc in docs:
                # 包含来源元数据，便于 LLM 引用                             
                formatted_str += f"- {doc.page_content}\n  来源: {doc.metadata.get('source', '未知')}\n\n"
            return formatted_str

        def format_for_retrieval(value: dict) -> str:
            return value["input"]

        def format_for_prompt_template(value):
            # {input, context, history}
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(format_for_retrieval) | retriever | format_documents
            }
            | RunnableLambda(format_for_prompt_template)
            | self.prompt_template
            | self.chat_model
            | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain

if __name__ == "__main__":
    session_config = {
        "configurable": {
            "session_id": "user_001",
        }
    }
    res = RagService().chain.invoke({"input": "围手术期高血压的诱发原因"}, session_config)
    print(res)