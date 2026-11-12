import os
import json
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from architectures.simple_llm import SimpleLLMAgent
from architectures.rag_llm import RAGLLMAgent
from architectures.router_agent import RouterAgent
from architectures.tool_agent import ToolCallingAgent

load_dotenv()

class LLMJudge:
    def __init__(self, model: str = "openai/gpt-4o-mini"):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.model = model

    def evaluate_answer(self, query: str, criteria: str, answer: str) -> int:
        prompt = (
            "Оцени качество ответа агента поддержки по шкале от 1 до 5, опираясь на эталонные критерии.\n"
            "1 — ответ неверный, вредный или галлюцинация.\n"
            "3 — ответ в целом верный, но упущены важные детали регламента/коды/условия.\n"
            "5 — полностью соответствует всем критериям, точный и корректный.\n"
            "Верни ТОЛЬКО одну цифру (1, 2, 3, 4 или 5)."
        )
        user_msg = f"Вопрос: {query}\nЭталонные критерии: {criteria}\nОтвет агента: {answer}"
        res = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.0,
            max_tokens=5
        )
        score_str = (res.choices[0].message.content or "3").strip()
        for ch in score_str:
            if ch in "12345":
                return int(ch)
        return 3

def run_benchmark():
    with open("data/test_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    architectures = {
        "A_Simple_LLM": SimpleLLMAgent(),
        "B_LLM_RAG": RAGLLMAgent(),
        "C_Router_RAG": RouterAgent(),
        "D_Tool_Agent": ToolCallingAgent(),
    }

    judge = LLMJudge()
    all_records = []
    summary_rows = []

    for arch_name, agent in architectures.items():
        print(f"\n=== Тестирование архитектуры: {arch_name} ===")
        arch_records = []

        for item in dataset:
            res = agent.run(item["query"])
            score = judge.evaluate_answer(item["query"], item["expected_criteria"], res.answer)

            row = {
                "architecture": arch_name,
                "id": item["id"],
                "category": item["category"],
                "query": item["query"],
                "answer": res.answer,
                "quality_score": score,
                "latency_sec": res.latency_sec,
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
                "cost_usd": res.cost_usd
            }
            arch_records.append(row)
            all_records.append(row)
            print(f"[{item['id']}] Score: {score}/5 | Latency: {res.latency_sec}s | Cost: ${res.cost_usd}")

        df_arch = pd.DataFrame(arch_records)
        latencies = df_arch["latency_sec"].values

        summary_rows.append({
            "Архитектура": arch_name,
            "Среднее качество (1-5)": round(df_arch["quality_score"].mean(), 2),
            "Доля идеальных (5/5), %": round((df_arch["quality_score"] == 5).mean() * 100, 1),
            "Latency P50 (сек)": round(np.percentile(latencies, 50), 3),
            "Latency P95 (сек)": round(np.percentile(latencies, 95), 3),
            "Средняя стоимость ($)": round(df_arch["cost_usd"].mean(), 6),
            "Общая стоимость ($)": round(df_arch["cost_usd"].sum(), 6),
        })

    # Сохраняем полные логи и итоговую сводку
    pd.DataFrame(all_records).to_csv("results/full_benchmark_logs.csv", index=False, encoding="utf-8-sig")
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("results/summary_comparison.csv", index=False, encoding="utf-8-sig")

    print("\n" + "="*80)
    print("ИТОГОВАЯ ТАБЛИЦА СРАВНЕНИЯ АРХИТЕКТУР")
    print("="*80)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    run_benchmark()