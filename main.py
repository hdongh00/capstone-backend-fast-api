import os
from datetime import datetime, date
from typing import Optional, Dict, Any

import jwt
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.utils import get_authorization_scheme_param
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_google_genai import ChatGoogleGenerativeAI
from pymongo import MongoClient
from pytz import timezone
from starlette.responses import StreamingResponse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from mariadb import MariaAnalysisRepo
from message import MessagesCollectionHistory
from langchain_openai import ChatOpenAI

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
JWT_SECRET = os.getenv("JWT_SECRET")
MODEL_PATH = "monologg/koelectra-base-v3-discriminator"

seoul_tz = timezone("Asia/Seoul")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = MongoClient(MONGODB_URI)
db = client["chatbot"]
CHAT_COLLECTION = db["messages"]
llm = ChatOpenAI(model="gpt-4.1-nano",openai_api_key=OPEN_AI_KEY)
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     google_api_key=GOOGLE_API_KEY,
#     temperature=0.3,
# )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 사용자의 감정을 깊게 공감하는 정신 건강 상담 챗봇이고, "
            "너는 무조건 사용자에게 오늘 하루에 있었던 일에 대하여 지속적으로 질문하고 공감해줘야한다.",
        ),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)

summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 이 history 대화를 모두 요약하여 철저히 사용자의 시선에서 한 편의 일기를 써야하는 일기 마스터이다. "
            "어투는 오늘은 ~ 했다 또는 ~가 있었다 등 과거형으로 집필해야하며, "
            "안좋은 내용이 있더라도 객관적으로 작성해야한다. 또한 대화를 한 것을 요약하여 일기를 작성하는 것이 아니라, user가 무슨 일을 겪고, 어떤 일이 있었는지 등으로 작성해야 한다.",
        ),
        MessagesPlaceholder("history"),
        ("human", "이 history를 기반으로 모두 대화를 요약하고, 내 관점에서 일기를 작성해줘."),
    ]
)

chain = prompt | llm
summary_chain = summary_prompt | llm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
emotion_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=3).to(device)
emotion_model.eval()

LABELS = ["부정", "중립", "긍정"]

EMOTION_SCALE = {
    1: "매우 나쁨",
    2: "나쁨",
    3: "조금 나쁨",
    4: "약간 나쁨",
    5: "보통",
    6: "약간 좋음",
    7: "좋음",
    8: "매우 좋음",
}

def get_token(authorization: str) -> str:
    scheme, token = get_authorization_scheme_param(authorization)
    return token


def decode_jwt(authorization: str) -> int:
    token = get_token(authorization)
    payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    user_code = payload.get("userCode")
    return int(user_code)


async def get_body_and_user(request: Request, authorization: str):
    body = await request.json()
    body_data = {}
    for b in body:
        body_data[b] = body.get(b)
    return body_data, decode_jwt(authorization)


def get_emotion_label_from_score(score: float) -> str:
    score_int = max(1, min(8, int(round(score))))
    return EMOTION_SCALE.get(score_int, "알 수 없음")


def analyze_emotion_score(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = emotion_model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)[0]
    probs = probabilities.tolist()

    scale_score = round(probs[0] * 2 + probs[1] * 5 + probs[2] * 8, 2)

    sorted_probs, indices = torch.sort(probabilities, descending=True)
    predicted_label = "중립" if (sorted_probs[0] - sorted_probs[1]) < 0.03 else LABELS[indices[0].item()]

    scores = {LABELS[i]: round(p * 100, 2) for i, p in enumerate(probs)}

    return {
        "예측": predicted_label,
        "확률": scores,
        "척도값": scale_score,
        "척도 해석": get_emotion_label_from_score(scale_score),
    }


@app.post("/chat")
async def chat(request: Request, authorization: str = Header(...)):
    req, user_code = await get_body_and_user(request, authorization)
    message = req["message"]
    conv_id = req["convId"]

    history = MessagesCollectionHistory(CHAT_COLLECTION, user_code, conv_id)
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
async def summary(request: Request, authorization: str = Header(...)):
    req, user_code = await get_body_and_user(request, authorization)
    conv_id, year, month, day = (req["convId"],req["year"],req["month"],req["day"])

    history = MessagesCollectionHistory(CHAT_COLLECTION, user_code, conv_id)
    past = history.get_messages(year=year, month=month, day=day)

    result = summary_chain.invoke({"history": past})
    summary_text = (result.content or "").strip()

    repo = MariaAnalysisRepo()
    target_date = datetime(year, month, day, tzinfo=seoul_tz).date()
    create_at = datetime(year, month, day, tzinfo=seoul_tz).replace(tzinfo=None)

    latest = repo.get_latest_by_user_and_date(user_code=user_code, target_date=target_date)

    if latest and latest.get("analysis_code"):
        repo.update(
            analysis_code=int(latest["analysis_code"]),
            emotion_score=float(latest.get("emotion_score") or 0.0),
            emotion_name=latest.get("emotion_name") or "",
            summary=summary_text,
            create_at=create_at,
        )
    else:
        repo.insert(
            user_code=user_code,
            emotion_score=0.0,
            emotion_name="",
            summary=summary_text,
            create_at=create_at,
        )

    return summary_text


@app.post("/analyze")
async def analyze(request: Request, authorization: str = Header(...)):
    req, user_code = await get_body_and_user(request, authorization)
    conv_id = req["convId"]

    history = MessagesCollectionHistory(CHAT_COLLECTION, user_code, conv_id)
    messages = history.get_messages()

    user_utterances = [msg.content for msg in messages if isinstance(msg, HumanMessage)]
    if not user_utterances:
        print("사용자 대화 없음")
        return {"message": "분석할 사용자 대화가 없습니다.", "result": None}

    analysis_results = []
    total_scale_score = 0.0

    for text in user_utterances:
        analysis = analyze_emotion_score(text)
        analysis_results.append(
            {
                "text": text,
                "prediction": analysis["예측"],
                "score_scale": analysis["척도값"],
                "probs": analysis["확률"],
            }
        )
        total_scale_score += analysis["척도값"]

    avg_scale_score = total_scale_score / len(user_utterances)
    overall_emotion_label = get_emotion_label_from_score(avg_scale_score)

    repo = MariaAnalysisRepo()
    target_date = datetime.now(seoul_tz).date()
    create_at = datetime.now(seoul_tz).replace(tzinfo=None)

    latest = repo.get_latest_by_user_and_date(user_code=user_code, target_date=target_date)
    print(avg_scale_score)
    if latest and latest.get("analysis_code"):
        summary_keep = None if latest.get("summary") is None else latest.get("summary")
        repo.update(
            analysis_code=int(latest["analysis_code"]),
            emotion_score=round((avg_scale_score/2), 2),
            emotion_name=overall_emotion_label,
            summary=summary_keep,
            create_at=create_at,
        )
    else:
        repo.insert(
            user_code=user_code,
            emotion_score=round((avg_scale_score / 2), 2),
            emotion_name=overall_emotion_label,
            summary=None,
            create_at=create_at,
        )

    return {
        "userCode": user_code,
        "convId": conv_id,
        "utterance_count": len(user_utterances),
        "details": analysis_results,
        "overall_stats": {
            "average_scale_score": round(avg_scale_score, 2),
            "overall_emotion": overall_emotion_label,
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)