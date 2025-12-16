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
from sqlalchemy import create_engine, text

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
JWT_SECRET = os.getenv("JWT_SECRET")
MODEL_PATH = os.getenv("MODEL_PATH", "monologg/koelectra-base-v3-discriminator")

MARIADB_HOST = os.getenv("MARIADB_HOST")
MARIADB_PORT = os.getenv("MARIADB_PORT", "3306")
MARIADB_DB = os.getenv("MARIADB_DB")
MARIADB_USER = os.getenv("MARIADB_USER")
MARIADB_PASSWORD = os.getenv("MARIADB_PASSWORD")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = MongoClient(MONGODB_URI)
mongo_db = client["chatbot"]
collection = mongo_db["messages"]
analysis_collection = mongo_db["daily_analysis"]

maria_engine = None
if all([MARIADB_HOST, MARIADB_DB, MARIADB_USER, MARIADB_PASSWORD]):
    maria_engine = create_engine(
        f"mysql+pymysql://{MARIADB_USER}:{MARIADB_PASSWORD}@{MARIADB_HOST}:{MARIADB_PORT}/{MARIADB_DB}?charset=utf8mb4",
        pool_pre_ping=True,
        pool_recycle=3600,
    )

llm = ChatGoogleGenerativeAI(
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

incremental_summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 사용자의 대화 기록 중 '새로 추가된 부분'만 요약해 일기처럼 이어서 쓸 수 있게 정리하는 역할이다. "
            "이미 요약된 내용과 중복되는 서술은 피하고, 새로 생긴 사건/감정/생각/행동 변화만 과거형으로 간결하게 작성한다.",
        ),
        MessagesPlaceholder("history"),
        ("human", "이 새로 추가된 대화만 요약해서 '추가 기록' 형태로 작성해줘."),
    ]
)

chain = prompt | llm
summary_chain = summary_prompt | llm
incremental_summary_chain = incremental_summary_prompt | llm

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
    emotion_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=3)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
    emotion_model = AutoModelForSequenceClassification.from_pretrained("beomi/kcbert-base", num_labels=3)

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
        predicted_label = LABELS[indices[0].item()] if indices[0].item() < len(LABELS) else "알수없음"

    scores = {LABELS[i]: round(p * 100, 2) for i, p in enumerate(probs)}

    return {
        "예측": predicted_label,
        "확률": scores,
        "척도값": scale_score,
        "척도 해석": get_emotion_label_from_score(scale_score),
    }

def load_user_messages_after(user_code: int, after_utc: datetime | None):
    query = {"userCode": user_code}
    if after_utc is not None:
        query["createdAt"] = {"$gt": after_utc}
    cursor = collection.find(query, {"_id": 0, "role": 1, "content": 1, "createdAt": 1}).sort("createdAt", 1)
    msgs = []
    for doc in cursor:
        role = doc.get("role")
        content = doc.get("content")
        if not content:
            continue
        if role in ["user", "human", "HumanMessage"]:
            msgs.append(HumanMessage(content=content))
        elif role in ["ai", "assistant", "AIMessage"]:
            msgs.append(AIMessage(content=content))
    return msgs

def load_all_user_human_texts(user_code: int):
    cursor = collection.find({"userCode": user_code}, {"_id": 0, "role": 1, "content": 1}).sort("createdAt", 1)
    texts = []
    for doc in cursor:
        if doc.get("role") in ["user", "human", "HumanMessage"] and doc.get("content"):
            texts.append(doc["content"])
    return texts

def recompute_emotion_from_all_logs(user_code: int):
    texts = load_all_user_human_texts(user_code)
    if not texts:
        return 0.0, "알 수 없음"
    total = 0.0
    for t in texts:
        total += analyze_emotion_score(t)["척도값"]
    avg = total / len(texts)
    return round(avg, 2), get_emotion_label_from_score(avg)

def maria_get_latest_row(user_code: int):
    if maria_engine is None:
        return None
    sql = text(
        """
        SELECT analysis_code, user_code, emotion_score, emotion_name, summary, created_at
        FROM analysis_result
        WHERE user_code = :user_code
        ORDER BY created_at DESC
        LIMIT 1
        """
    )
    with maria_engine.begin() as conn:
        return conn.execute(sql, {"user_code": user_code}).fetchone()

def maria_insert_new_row(user_code: int, emotion_score: float, emotion_name: str, summary: str, created_at_utc: datetime):
    if maria_engine is None:
        return
    sql = text(
        """
        INSERT INTO analysis_result (user_code, emotion_score, emotion_name, summary, created_at)
        VALUES (:user_code, :emotion_score, :emotion_name, :summary, :created_at)
        """
    )
    with maria_engine.begin() as conn:
        conn.execute(
            sql,
            {
                "user_code": user_code,
                "emotion_score": float(emotion_score),
                "emotion_name": str(emotion_name)[:25],
                "summary": (summary or "")[:3000],
                "created_at": created_at_utc,
            },
        )

def maria_update_row(analysis_code: int, user_code: int, emotion_score: float, emotion_name: str, summary: str, created_at_utc: datetime):
    if maria_engine is None:
        return
    sql = text(
        """
        UPDATE analysis_result
        SET emotion_score = :emotion_score,
            emotion_name = :emotion_name,
            summary = :summary,
            created_at = :created_at
        WHERE analysis_code = :analysis_code AND user_code = :user_code
        """
    )
    with maria_engine.begin() as conn:
        conn.execute(
            sql,
            {
                "analysis_code": int(analysis_code),
                "user_code": int(user_code),
                "emotion_score": float(emotion_score),
                "emotion_name": str(emotion_name)[:25],
                "summary": (summary or "")[:3000],
                "created_at": created_at_utc,
            },
        )

def build_incremental_summary(user_code: int):
    created_at_utc = datetime.utcnow()
    last = maria_get_latest_row(user_code)

    last_time_utc = None
    prev_summary = ""
    last_analysis_code = None

    if last is not None:
        last_analysis_code = int(last[0])
        prev_summary = (last[4] or "")
        last_time_utc = last[5]

    new_messages = load_user_messages_after(user_code, last_time_utc)

    new_human_exists = any(isinstance(m, HumanMessage) for m in new_messages)
    if not new_human_exists:
        emotion_score, emotion_name = recompute_emotion_from_all_logs(user_code)
        if last_analysis_code is None:
            if prev_summary:
                maria_insert_new_row(user_code, emotion_score, emotion_name, prev_summary, created_at_utc)
            else:
                maria_insert_new_row(user_code, emotion_score, emotion_name, "", created_at_utc)
        else:
            maria_update_row(last_analysis_code, user_code, emotion_score, emotion_name, prev_summary, created_at_utc)
        return prev_summary

    if prev_summary:
        inc = incremental_summary_chain.invoke({"history": new_messages}).content
        merged = (prev_summary.strip() + "\n\n[추가 기록]\n" + inc.strip()).strip()
    else:
        merged = summary_chain.invoke({"history": new_messages}).content.strip()

    emotion_score, emotion_name = recompute_emotion_from_all_logs(user_code)

    if last_analysis_code is None:
        maria_insert_new_row(user_code, emotion_score, emotion_name, merged, created_at_utc)
    else:
        maria_update_row(last_analysis_code, user_code, emotion_score, emotion_name, merged, created_at_utc)

    return merged

seoul_tz = timezone("Asia/Seoul")
scheduler = AsyncIOScheduler(timezone=seoul_tz)

def run_midnight_job():
    now = datetime.now(seoul_tz)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - timedelta(days=1)
    yesterday_end = today_start

    cursor = collection.find(
        {"createdAt": {"$gte": yesterday_start, "$lt": yesterday_end}},
        {"_id": 0, "userCode": 1},
    )

    user_set = set()
    for doc in cursor:
        uc = doc.get("userCode")
        if uc is not None:
            try:
                user_set.add(int(uc))
            except Exception:
                pass

    for uc in user_set:
        try:
            build_incremental_summary(uc)
        except Exception:
            continue

@app.on_event("startup")
def start_scheduler():
    scheduler.add_job(run_midnight_job, "cron", hour=0, minute=0)
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
            text_out = getattr(chunk, "content", str(chunk))
            chunks.append(text_out)
            yield text_out

        full = "".join(chunks)
        history.add_message(HumanMessage(content=message))
        if full.strip():
            history.add_message(AIMessage(content=full))

    return StreamingResponse(get_streaming_response(), media_type="text/plain")

@app.post("/summary")
async def summary(request: Request, authorization: str = Header(...)):
    _, user_code = await get_body_and_user(request, authorization)

    if maria_engine is None:
        raise HTTPException(status_code=500, detail="MariaDB is not configured")

    merged_summary = build_incremental_summary(user_code)

    axios = Axios(get_token(authorization), "text/plain")
    _ = axios.post("/auth/summary", merged_summary)

    return merged_summary

@app.post("/analyze")
async def analyze(request: Request, authorization: str = Header(...)):
    _, user_code = await get_body_and_user(request, authorization)

    texts = load_all_user_human_texts(user_code)
    if not texts:
        return {"message": "분석할 사용자 대화가 없습니다.", "result": None}

    analysis_results = []
    total_scale_score = 0.0

    for t in texts:
        a = analyze_emotion_score(t)
        analysis_results.append(
            {
                "text": t,
                "prediction": a["예측"],
                "score_scale": a["척도값"],
                "probs": a["확률"],
            }
        )
        total_scale_score += a["척도값"]

    avg_scale_score = total_scale_score / len(texts)
    overall_emotion_label = get_emotion_label_from_score(avg_scale_score)

    return {
        "userCode": user_code,
        "utterance_count": len(texts),
        "details": analysis_results,
        "overall_stats": {
            "average_scale_score": round(avg_scale_score, 2),
            "overall_emotion": overall_emotion_label,
        },
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
