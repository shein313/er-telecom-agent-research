import os
import time
from openai import OpenAI
from architectures.base import BaseAgent, AgentResponse

class SimpleLLMAgent(BaseAgent):
    def __init__(self, model: str = "openai/gpt-4o-mini", price_in_1m: float = 0.15, price_out_1m: float = 0.60):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.model = model
        self.price_in_1m = price_in_1m
        self.price_out_1m = price_out_1m
        self.system_prompt = (
            "Ты — вежливый и компетентный специалист первой линии технической поддержки "
            "интернет-провайдера. Отвечай кратко, по делу и понятным языком. "
            "Не выдумывай несуществующие тарифы и не утверждай, что проверил линию, если у тебя нет доступа к диагностике. "
            "Если вопрос требует доступа к биллингу, расторжения договора или выезда мастера — явно укажи, что переводишь диалог на оператора."
        )

    def run(self, query: str) -> AgentResponse:
        start_time = time.perf_counter()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.2
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