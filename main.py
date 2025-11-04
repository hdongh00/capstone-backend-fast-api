import os
from fastapi import FastAPI, Header, Request
from fastapi.security.utils import get_authorization_scheme_param
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from starlette.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from message import MessagesCollectionHistory
import jwt
from pymongo import MongoClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import uvicorn

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
JWT_SECRET = os.getenv("JWT_SECRET")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"],allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
client = MongoClient(MONGODB_URI)
collection = client["chatbot"]["messages"]

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=GOOGLE_API_KEY,temperature=0.3,)

prompt = ChatPromptTemplate.from_messages([
("system", "너는 사용자의 감정을 깊게 공감하는 정신 건강 상담 챗봇이고, 너는 무조건 사용자에게 오늘 하루에 있었던 일에 대하여 지속적으로 질문하고 공감해줘야한다. 또한 절대로 전에 했던 history 대화에 있는 질문에 답변해선 안된다."),
MessagesPlaceholder("history"),
("human", "{input}")
])

chain = prompt | model

def decode_jwt(authorization):
    scheme, token = get_authorization_scheme_param(authorization)
    payload = jwt.decode(token, JWT_SECRET, "HS256")
    user_code = payload.get("userCode")
    return int(user_code)


@app.post("/chat")
async def chat(request: Request, authorization: str = Header(...)):
    body = await request.json()
    message = body.get("message")
    print(message)

    conv_id = body.get("convId")
    user_code = decode_jwt(authorization)
    history = MessagesCollectionHistory(collection, user_code, conv_id)
    past = history.get_messages()

    def get_streaming_response():
        chunks = []
        for chunk in chain.stream({"history": past, "input": message}):
            text = getattr(chunk, "content", str(chunk))
            chunks.append(text)
            yield text

        full = "".join(chunks)
        history.add_message(HumanMessage(content=message))
        if full.strip():
            history.add_message(AIMessage(content=full))

    return StreamingResponse(get_streaming_response(), media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)