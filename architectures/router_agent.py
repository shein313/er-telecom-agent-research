import os
import json
import time
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from architectures.base import BaseAgent, AgentResponse

class RouterAgent(BaseAgent):
    def __init__(
        self,
        kb_path: str = "data/knowledge_base.json",
        model: str = "openai/gpt-4o-mini",
        price_in_1m: float = 0.15,
        price_out_1m: float = 0.60
    ):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.model = model
        self.price_in_1m = price_in_1m
        self.price_out_1m = price_out_1m

        with open(kb_path, "r", encoding="utf-8") as f:
            self.kb_docs = json.load(f)

    def _classify(self, query: str) -> tuple[str, int, int]:
        router_prompt = (
            "Классифицируй обращение клиента интернет-провайдера строго в одну из категорий:\n"
            "- tech_issue (не работает интернет, Wi-Fi, роутер, ТВ-приставка)\n"
            "- billing (оплата, списания, баланс, обещанный платеж, тарифы)\n"
            "- escalation (расторжение договора, возврат денег, авария в подъезде, искрят провода, требование живого оператора)\n"
            "- out_of_scope (вопросы не по теме связи, просьбы написать код и т.д.)\n"
            "Ответь ТОЛЬКО названием категории одним словом."
        )
        res = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": router_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.0,
            max_tokens=15
        )
        category = (res.choices[0].message.content or "tech_issue").strip().lower()
        p_tok = res.usage.prompt_tokens if res.usage else 0
        c_tok = res.usage.completion_tokens if res.usage else 0
        return category, p_tok, c_tok

    def _retrieve_filtered(self, query: str, category: str) -> str:
        # Фильтруем базу знаний под конкретную ветку
        if category == "billing":
            filtered_docs = [d for d in self.kb_docs if d["category"] in ("billing", "tariffs")]
        else:
            filtered_docs = [d for d in self.kb_docs if d["category"] in ("tech_issue", "tv_issue", "router_setup")]

        if not filtered_docs:
            filtered_docs = self.kb_docs

        corpus = [f"{d['title']}. {d['content']}" for d in filtered_docs]
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(corpus)
        sims = cosine_similarity(vectorizer.transform([query]), vectors).flatten()
        top_idx = sims.argsort()[::-1][:2]

        chunks = [f"[{filtered_docs[i]['doc_id']}]: {filtered_docs[i]['content']}" for i in top_idx if sims[i] > 0.05]
        return "\n\n".join(chunks) if chunks else "Документов не найдено."

    def run(self, query: str) -> AgentResponse:
        start_time = time.perf_counter()
        total_prompt_tokens = 0
        total_completion_tokens = 0

        # Шаг 1: Маршрутизация
        category, p_tok, c_tok = self._classify(query)
        total_prompt_tokens += p_tok
        total_completion_tokens += c_tok

        # Шаг 2: Обработка по веткам
        if "escalation" in category:
            answer = (
                "Понял вас. Для решения этого вопроса (аварийная ситуация или операции по договору/возврату) "
                "я немедленно перевожу диалог на старшего специалиста / дежурного диспетчера. Пожалуйста, ожидайте."
            )
        elif "out_of_scope" in category:
            answer = "Я виртуальный помощник технической поддержки ЭР-Телеком и могу помочь только с вопросами по интернету, ТВ, оборудованию и тарифам."
        else:
            context = self._retrieve_filtered(query, category)
            sys_prompt = (
                f"Ты — специалист поддержки ЭР-Телеком (отдел: {category}). "
                "Отвечай строго по предоставленному регламенту. Если данных не хватает — переведи на оператора."
            )
            res = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"Регламент:\n{context}\n\nВопрос: {query}"}
                ],
                temperature=0.1
            )
            answer = res.choices[0].message.content or ""
            total_prompt_tokens += res.usage.prompt_tokens if res.usage else 0
            total_completion_tokens += res.usage.completion_tokens if res.usage else 0

        latency = time.perf_counter() - start_time
        cost = (total_prompt_tokens * self.price_in_1m + total_completion_tokens * self.price_out_1m) / 1_000_000

        return AgentResponse(
            answer=answer,
            latency_sec=round(latency, 3),
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            cost_usd=round(cost, 6)
        )