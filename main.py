import os
from fastapi import FastAPI, Header, Request, HTTPException
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
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime, timedelta
from pytz import timezone
from request import Axios
from message import MessagesCollectionHistory

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
JWT_SECRET = os.getenv("JWT_SECRET")
MODEL_PATH = "monologg/koelectra-base-v3-discriminator"

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
collection = db["messages"]
analysis_collection = db["daily_analysis"]

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
)

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
            "안좋은 내용이 있더라도 객관적으로 작성해야한다.",
        ),
        MessagesPlaceholder("history"),
        ("human", "이 history를 기반으로 모두 대화를 요약하고, 내 관점에서 일기를 작성해줘."),
    ]
)

chain = prompt | model
summary_chain = summary_prompt | model

def get_token(authorization: str) -> str:
    scheme, token = get_authorization_scheme_param(authorization)
    return token

def decode_jwt(authorization: str) -> int:
    token = get_token(authorization)
    payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    user_code = payload.get("userCode")
    if user_code is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    return int(user_code)

async def get_body_and_user(request: Request, authorization: str):
    body = await request.json()
    return body, decode_jwt(authorization)

LABELS = ["부정", "중립", "긍정"]
EMOTION_SCALE = {
    1: "매우 부정적",
    2: "부정적",
    3: "다소 부정적",
    4: "약간 부정적",
    5: "중립적",
    6: "약간 긍정적",
    7: "긍정적",
    8: "매우 긍정적",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    emotion_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=3
    )
except Exception:
    tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
    emotion_model = AutoModelForSequenceClassification.from_pretrained(
        "beomi/kcbert-base", num_labels=3
    )

emotion_model.to(device)
emotion_model.eval()

def get_emotion_label_from_score(score: float) -> str:
    score_int = max(1, min(8, int(round(score))))
    return EMOTION_SCALE.get(score_int, "알 수 없음")

def analyze_emotion_score(text: str) -> dict:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    ).to(device)

    with torch.no_grad():
        outputs = emotion_model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)[0]
    probs = probabilities.tolist()

    if len(probs) == 3:
        scale_score = round(probs[0] * 2 + probs[1] * 5 + probs[2] * 8, 2)
    else:
        scale_score = 5.0

    sorted_probs, indices = torch.sort(probabilities, descending=True)

    if len(probs) >= 2 and (sorted_probs[0] - sorted_probs[1]) < 0.03:
        predicted_label = "중립"
    else:
        predicted_label = (
            LABELS[indices[0].item()]
            if indices[0].item() < len(LABELS)
            else "알수없음"
        )

    scores = {LABELS[i]: round(p * 100, 2) for i, p in enumerate(probs)}

    return {
        "예측": predicted_label,
        "확률": scores,
        "척도값": scale_score,
        "척도 해석": get_emotion_label_from_score(scale_score),
    }

seoul_tz = timezone("Asia/Seoul")
scheduler = AsyncIOScheduler(timezone=seoul_tz)

def run_daily_emotion_analysis():
    now = datetime.now(seoul_tz)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - timedelta(days=1)
    yesterday_end = today_start

    cursor = collection.find(
        {"createdAt": {"$gte": yesterday_start, "$lt": yesterday_end}}
    )

    user_conv_map = {}
    for doc in cursor:
        user_code = doc.get("userCode")
        conv_id = doc.get("convId")
        role = doc.get("role")
        content = doc.get("content")

        if not user_code or not conv_id or not content:
            continue
        if role not in ["user", "human", "HumanMessage"]:
            continue

        key = (int(user_code), str(conv_id))
        user_conv_map.setdefault(key, []).append(content)

    for (user_code, conv_id), texts in user_conv_map.items():
        if not texts:
            continue

        analysis_results = []
        total_scale_score = 0.0

        for text in texts:
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

        avg_scale_score = total_scale_score / len(texts)
        overall_emotion_label = get_emotion_label_from_score(avg_scale_score)

        doc = {
            "userCode": user_code,
            "convId": conv_id,
            "date": yesterday_start.date().isoformat(),
            "utterance_count": len(texts),
            "details": analysis_results,
            "overall_stats": {
                "average_scale_score": round(avg_scale_score, 2),
                "overall_emotion": overall_emotion_label,
            },
            "createdAt": datetime.now(seoul_tz),
        }

        analysis_collection.update_one(
            {"userCode": user_code, "convId": conv_id, "date": doc["date"]},
            {"$set": doc},
            upsert=True,
        )

@app.on_event("startup")
def start_scheduler():
    scheduler.add_job(run_daily_emotion_analysis, "cron", hour=0, minute=0)
    scheduler.start()

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
async def summary(request: Request, authorization: str = Header(...)):
    req, user_code = await get_body_and_user(request, authorization)
    history = MessagesCollectionHistory(collection, user_code, req["convId"])
    past = history.get_messages()
    result = summary_chain.invoke({"history": past})
    axios = Axios(get_token(authorization), "text/plain")
    _ = axios.post("/auth/summary", result.content)
    return result.content

@app.post("/analyze")
async def analyze(request: Request, authorization: str = Header(...)):
    req, user_code = await get_body_and_user(request, authorization)
    history = MessagesCollectionHistory(collection, user_code, req["convId"])
    messages = history.get_messages()

    user_utterances = [msg.content for msg in messages if isinstance(msg, HumanMessage)]
    if not user_utterances:
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

    return {
        "userCode": user_code,
        "convId": req["convId"],
        "utterance_count": len(user_utterances),
        "details": analysis_results,
        "overall_stats": {
            "average_scale_score": round(avg_scale_score, 2),
            "overall_emotion": overall_emotion_label,
        },
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
