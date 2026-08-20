from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage


class EvaluatorMemory:
    def __init__(self):
        self._history = InMemoryChatMessageHistory()

    def add_evaluation_input(self, content: str) -> None:
        self._history.add_user_message(content)

    def add_verdict(self, content: str) -> None:
        self._history.add_ai_message(content)

    def get_messages(self) -> list[BaseMessage]:
        return self._history.messages

    def clear(self) -> None:
        self._history.clear()