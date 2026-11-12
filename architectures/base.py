from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class AgentResponse:
    answer: str
    latency_sec: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float

class BaseAgent(ABC):
    @abstractmethod
    def run(self, query: str) -> AgentResponse:
        pass