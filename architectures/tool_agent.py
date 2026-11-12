import os
import json
import time
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from architectures.base import BaseAgent, AgentResponse

class ToolCallingAgent(BaseAgent):
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

        corpus = [f"{doc['title']}. {doc['content']}" for doc in self.kb_docs]
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(corpus)

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge_base",
                    "description": "Поиск инструкций, тарифов и регламентов в базе знаний провайдера ЭР-Телеком.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "search_query": {"type": "string", "description": "Поисковый запрос по проблеме клиента"}
                        },
                        "required": ["search_query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "escalate_or_create_ticket",
                    "description": "Создать заявку на выезд мастера или перевести диалог на живого оператора/диспетчера.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "department": {"type": "string", "enum": ["field_engineer", "billing_operator", "emergency_dispatcher"]},
                            "reason": {"type": "string", "description": "Краткая суть проблемы"}
                        },
                        "required": ["department", "reason"]
                    }
                }
            }
        ]

    def _exec_search(self, search_query: str) -> str:
        vec = self.vectorizer.transform([search_query])
        sims = cosine_similarity(vec, self.doc_vectors).flatten()
        top_idx = sims.argsort()[::-1][:2]
        res = [f"[{self.kb_docs[i]['doc_id']}]: {self.kb_docs[i]['content']}" for i in top_idx if sims[i] > 0.05]
        return "\n".join(res) if res else "Ничего не найдено."

    def _exec_ticket(self, department: str, reason: str) -> str:
        return f"Заявка успешно зарегистрирована в системе. Направление: {department}. Причина: {reason}. Номер обращения: #ERT-2026."

    def run(self, query: str) -> AgentResponse:
        start_time = time.perf_counter()
        total_prompt_tokens = 0
        total_completion_tokens = 0

        messages = [
            {
                "role": "system",
                "content": (
                    "Ты — автономный AI-агент первой линии поддержки ЭР-Телеком. "
                    "Перед ответом на вопросы по услугам, техническим сбоям или тарифам всегда используй search_knowledge_base. "
                    "Если по регламенту требуется выезд мастера, расторжение договора или фиксация аварии — вызови escalate_or_create_ticket."
                )
            },
            {"role": "user", "content": query}
        ]

        # Максимум 3 итерации цикла агента (защита от бесконечного цикла)
        for _ in range(3):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.1
            )
            total_prompt_tokens += response.usage.prompt_tokens if response.usage else 0
            total_completion_tokens += response.usage.completion_tokens if response.usage else 0

            msg = response.choices[0].message
            messages.append(msg)

            if not msg.tool_calls:
                answer = msg.content or ""
                break

            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                if fn_name == "search_knowledge_base":
                    tool_result = self._exec_search(args.get("search_query", query))
                elif fn_name == "escalate_or_create_ticket":
                    tool_result = self._exec_ticket(args.get("department", "billing_operator"), args.get("reason", ""))
                else:
                    tool_result = "Неизвестный инструмент."

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })
        else:
            answer = "Превышен лимит шагов обработки. Перевожу на оператора."

        latency = time.perf_counter() - start_time
        cost = (total_prompt_tokens * self.price_in_1m + total_completion_tokens * self.price_out_1m) / 1_000_000

        return AgentResponse(
            answer=answer,
            latency_sec=round(latency, 3),
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            cost_usd=round(cost, 6)
        )