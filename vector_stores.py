from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import config_data as config

class VectorStoreService:

    def __init__(self, embeddings):
        """
        :param embeddings: 嵌入模型实例
        """
        self.embeddings = embeddings
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embeddings,
            persist_directory=config.persist_directory,
        )
        
    def get_retriever(self, documents, weights=None):
        """
        返回混合检索器（向量 + BM25），使用 EnsembleRetriever 加权融合
        :param documents: BM25 所需的文档列表（字符串列表），必须外部传入
        :param weights: 权重列表，长度与 retrievers 相同，默认 [0.5, 0.5]
        :return: EnsembleRetriever 对象
        """
        if not documents:
            raise ValueError("documents 不能为空，请提供用于 BM25 检索的文档列表")

        # 向量检索器
        vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": config.similarity_threshold}
        )

        # BM25 检索器
        bm25_retriever = BM25Retriever.from_texts(documents, k=config.bm25_k)

        # 默认等权重
        if weights is None:
            weights = [0.5, 0.5]

        # 融合检索器
        fusion_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=weights
        )
        return fusion_retriever


if __name__ == "__main__":
    from langchain_community.embeddings import DashScopeEmbeddings

    # 测试示例：使用融合检索器查询
    sample_docs = [
        "高血压病患者生气后，血压升至250/120毫米汞柱，出现癫痫样抽搐、呕吐、意识模糊等中枢神经系统功能障碍的症状，脑CT未见异常。该患者最可能的诊断是",
        "高血压脑病是高血压急症的一种，表现为血压急剧升高伴中枢神经系统症状。",
        "治疗原则是快速降压，首选硝普钠或拉贝洛尔。"
    ]

    retriever = VectorStoreService(
        DashScopeEmbeddings(model=config.embedding_model_name)
    ).get_retriever(
        documents=sample_docs,
        weights=[0.7, 0.3]   # 向量检索权重0.7，BM25权重0.3
    )

    query = "高血压脑病的诊断与治疗"
    res = retriever.invoke(query)

    # 输出检索结果摘要
    print(f"\n查询：{query}\n")
    for i, doc in enumerate(res, 1):
        content = doc.page_content.strip().replace('\n', ' ')
        if len(content) > 150:
            content = content[:150] + "..."
        print(f"{i}. {content}")