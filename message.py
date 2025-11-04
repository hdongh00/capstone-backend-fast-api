from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from bson import ObjectId
from datetime import datetime

class MessagesCollectionHistory(BaseChatMessageHistory):
    def __init__(self, collection, user_code, conv_id):
        self.collection = collection
        self.user_code = user_code
        self.conv_id = ObjectId(conv_id) if conv_id else None


    def _filter(self):
        f = {"userCode": self.user_code}
        if self.conv_id:
            f["convId"] = self.conv_id
        return f


    def get_messages(self):
        docs = list(self.collection.find(self._filter()).sort("createdAt", 1))
        out = []
        for d in docs:
            role = d.get("role")
            content = d.get("content", "")
            if role == "user":
                out.append(HumanMessage(content=content))
            else:
                out.append(AIMessage(content=content))
        return out


    def add_message(self, message):
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        self.collection.insert_one({
        "convId": self.conv_id,
        "content": message.content,
        "createdAt": datetime.utcnow(),
        "role": role,
        "userCode": self.user_code,
        })


    def clear(self):
        self.collection.delete_many(self._filter())
