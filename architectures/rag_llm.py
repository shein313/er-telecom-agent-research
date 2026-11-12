import os
import json
import time
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from architectures.base import BaseAgent, AgentResponse

class RAGLLMAgent(BaseAgent):
    def __init__(
        self,
        kb_path: str = "data/knowledge_base.json",
        model: str = "openai/gpt-4o-mini",
        price_in_1m: float = 0.15,
        price_out_1m: float = 0.60,
        top_k: int = 2
    ):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.model = model
        self.price_in_1m = price_in_1m
        self.price_out_1m = price_out_1m
        self.top_k = top_k

        # Загружаем базу знаний и строим поисковый индекс
        with open(kb_path, "r", encoding="utf-8") as f:
            self.kb_docs = json.load(f)

        corpus = [f"{doc['title']}. {doc['content']}" for doc in self.kb_docs]
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(corpus)

        self.system_prompt = (
            "Ты — специалист первой линии технической поддержки интернет-провайдера ЭР-Телеком. "
            "Отвечай строго на основе предоставленных ниже выдержек из внутренней базы знаний. "
            "Указывай точные коды ошибок, цены, сроки и инструкции из регламента, если они есть в контексте. "
            "Если в предоставленных документах нет ответа на вопрос — не выдумывай, а вежливо сообщи, что переводишь диалог на оператора."
        )

    def _retrieve(self, query: str) -> str:
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()
        top_indices = similarities.argsort()[::-1][:self.top_k]

        retrieved_chunks = []
        for idx in top_indices:
            if similarities[idx] > 0.05:  # отсекаем совсем нерелевантные документы
                doc = self.kb_docs[idx]
                retrieved_chunks.append(f"[{doc['doc_id']} - {doc['title']}]:\n{doc['content']}")

        if not retrieved_chunks:
            return "Релевантных документов в базе знаний не найдено."
        return "\n\n".join(retrieved_chunks)

    def run(self, query: str) -> AgentResponse:
        start_time = time.perf_counter()

        # 1. Этап поиска (Retrieval)
        context = self._retrieve(query)

        # 2. Этап генерации (Generation)
        user_message = (
            f"Контекст из базы знаний:\n{context}\n\n"
            f"Вопрос клиента: {query}"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1
        )

        latency = time.perf_counter() - start_time
        answer = response.choices[0].message.content or ""

        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        cost = (prompt_tokens * self.price_in_1m + completion_tokens * self.price_out_1m) / 1_000_000

        return AgentResponse(
            answer=answer,
            latency_sec=round(latency, 3),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=round(cost, 6)
        )