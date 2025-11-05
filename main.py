# 라이브러리들
import os
from fastapi import FastAPI, Header, Request
from fastapi.security.utils import get_authorization_scheme_param
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from starlette.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import jwt
from pymongo import MongoClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import uvicorn
# 직접 만든 클래스들
from request import Axios
from message import MessagesCollectionHistory

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
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 이 history 대화를 모두 요약하여 철저히 사용자의 시선에서 한 편의 일기를 써야하는 일기 마스터이다. 어투는 오늘은 ~ 했다 또는 ~가 있었다 등 과거형으로 집필해야하며, 안좋은 내용이 있더라도 객관적으로 작성해야한다."),
    MessagesPlaceholder("history"),
    ("human", "이 history를 기반으로 모두 대화를 요약하고, 내 관점에서 일기를 작성해줘.")
])

chain = prompt | model
summary_chain = summary_prompt | model

def get_token(authorization):
    scheme, token = get_authorization_scheme_param(authorization)
    return token

def decode_jwt(authorization):
    token = get_token(authorization)
    payload = jwt.decode(token, JWT_SECRET, "HS256")
    user_code = payload.get("userCode")
    return int(user_code)

async def get_body_and_user(request: Request, authorization):
    body = await request.json()
    body_data = {}
    for b in body:
        body_data[b] = body.get(b)
    return body_data, decode_jwt(authorization)


@app.post("/chat")
async def chat(request: Request, authorization: str = Header(...)):
    req, user_code = await get_body_and_user(request, authorization)
    message = req["message"]
    history = MessagesCollectionHistory(collection, user_code, req["convId"])
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

@app.post("/summary")
async def summary(request: Request,authorization: str = Header(...)):
    req, user_code = await get_body_and_user(request, authorization)
    history = MessagesCollectionHistory(collection, user_code, req["convId"])
    past = history.get_messages()
    result = summary_chain.invoke({"history": past})
    axios = Axios(get_token(authorization), "text/plain")
    r = axios.post("/auth/summary", result.content)
    return result.content


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)