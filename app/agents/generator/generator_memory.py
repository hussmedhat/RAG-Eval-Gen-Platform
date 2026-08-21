from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage

class GenerateMemory:

    def __init__(self):
        self._history= InMemoryChatMessageHistory()

    def add_user_message(self,message:str)-> None:
        self._history.add_user_message(message)

    def add_ai_message(self,message:str)-> None:
        self._history.add_ai_message(message)

    def get_messages(self)->list[BaseMessage]:
        return self._history.messages

    def clear(self)-> None:
        self._history.clear()
