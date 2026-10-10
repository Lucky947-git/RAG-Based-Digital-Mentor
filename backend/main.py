from __future__ import annotations
import os
import uuid
import json
import logging
import re
from contextlib import asynccontextmanager
from typing import Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
import httpx
from groq import APIStatusError, AuthenticationError, RateLimitError
from backend.config import CORS_ORIGINS, GROQ_MODEL, SESSION_PATH
from backend.rag import ingest_pdfs, status as rag_status
from backend.workflow import reply, save_session
from yt_suggester import suggest_shorts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("viveka")
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Reuse keep-alive connections to Sarvam; opening a new TLS connection for
    # every spoken question made playback noticeably slower.
    app.state.sarvam_client = httpx.AsyncClient(timeout=httpx.Timeout(28.0, connect=6.0))
    try:
        yield
    finally:
        await app.state.sarvam_client.aclose()

app = FastAPI(title="Viveka — A Digital Mentor for Students", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "static")), name="static")

@app.middleware("http")
async def keep_reflection_pages_fresh(request: Request, call_next):
    response = await call_next(request)
    if request.url.path in {"/static/growth-pond.html", "/static/reflections.html"}:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response

def _sarvam_key() -> str:
    key = os.getenv("SARVAM_API_KEY", "").strip()
    if not key:
        raise HTTPException(status_code=503, detail="Voice features need SARVAM_API_KEY in .env. Restart the app after setting it.")
    return key

def _groq_error_message(exc: APIStatusError) -> str | None:
    """Extract Groq's short validation message without exposing credentials."""
    body = getattr(exc, "body", None)
    error = body.get("error") if isinstance(body, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    if not isinstance(message, str) or not message.strip():
        return None
    key = os.getenv("GROQ_API_KEY", "").strip()
    if key:
        message = message.replace(key, "[redacted]")
    message = re.sub(r"\s+", " ", message).strip()
    return message[:280]

@app.post("/api/voice/transcribe")
async def transcribe_voice(request: Request, file: UploadFile = File(...), language_code: str = Form("en-IN")):
    allowed = {"en-IN", "hi-IN", "bn-IN", "gu-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN"}
    if language_code not in allowed:
        raise HTTPException(status_code=422, detail="Choose a supported transcription language.")
    audio = await file.read(8 * 1024 * 1024 + 1)
    if not audio or len(audio) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="The recording is empty or too large. Please record a shorter message.")
    try:
        response = await request.app.state.sarvam_client.post(
                "https://api.sarvam.ai/speech-to-text",
                headers={"api-subscription-key": _sarvam_key()},
                data={"model": "saaras:v3", "mode": "transcribe", "language_code": language_code},
                files={"file": ("voice.wav", audio, "audio/wav")},
        )
        if response.status_code in (401, 403):
            raise HTTPException(status_code=503, detail="Sarvam did not accept the configured API key. Check SARVAM_API_KEY in .env.")
        response.raise_for_status()
        transcript = response.json().get("transcript", "").strip()
        if not transcript:
            raise HTTPException(status_code=422, detail="I couldn’t hear clear words in that recording. Please try again.")
        return {"transcript": transcript, "language_code": response.json().get("language_code")}
    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        logger.warning("Sarvam transcription rejected the audio with HTTP %s", exc.response.status_code)
        raise HTTPException(status_code=502, detail="Sarvam couldn’t transcribe that recording. Try a shorter recording and speak clearly.") from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail="Could not connect to Sarvam. Check your internet connection and retry.") from exc

class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    language_code: str = Field(default="en-IN", pattern="^(en-IN|hi-IN)$")

@app.post("/api/voice/speak")
async def speak_question(payload: SpeechRequest, request: Request):
    try:
        response = await request.app.state.sarvam_client.post(
                "https://api.sarvam.ai/text-to-speech",
                headers={"api-subscription-key": _sarvam_key(), "Content-Type": "application/json"},
                json={"text": payload.text, "language_code": payload.language_code, "model": "bulbul:v3", "speaker": "shubh", "output_audio_codec": "wav", "speech_sample_rate": 24000},
            )
        if response.status_code in (401, 403):
            raise HTTPException(status_code=503, detail="Sarvam did not accept the configured API key. Check SARVAM_API_KEY in .env.")
        response.raise_for_status()
        audio = response.json().get("audios", [])
        if not audio:
            raise HTTPException(status_code=502, detail="Sarvam did not return speech audio.")
        return {"audio_base64": audio[0], "content_type": "audio/wav"}
    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        logger.warning("Sarvam speech synthesis returned HTTP %s", exc.response.status_code)
        raise HTTPException(status_code=502, detail="Sarvam couldn’t speak that question just now.") from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail="Could not connect to Sarvam. Check your internet connection and retry.") from exc
    except (ValueError, KeyError) as exc:
        logger.warning("Sarvam speech response was not valid JSON audio")
        raise HTTPException(status_code=502, detail="Sarvam returned an invalid speech response. Please retry.") from exc

def _short_topic(text: str) -> str:
    rules = [
        (r"exam|fail|failure|marks|result|revision|study|studying|padhai", "exam stress and study focus"),
        (r"fear|afraid|scared|darr|anxious|anxiety|panic|worry|worried", "fear and anxiety"),
        (r"parent|family|expectation|ghar|maa|papa", "family expectations and communication"),
        (r"motivat|hopeless|give up|tired|energy", "motivation and inner strength"),
        (r"focus|distract|phone|concentrat", "focus and avoiding distractions"),
        (r"career|job|future|purpose|college", "career and purpose"),
        (r"relationship|friend|breakup|love", "relationships and communication"),
        (r"confidence|doubt|worth|compare|inadequate", "confidence and self-belief"),
        (r"decision|choose|confus|overthink", "decision-making and overthinking"),
        (r"stress|pressure|overwhelm", "student stress and pressure"),
    ]
    lower = text.lower()
    matched = list(dict.fromkeys(topic for pattern, topic in rules if re.search(pattern, lower)))
    if matched:
        # Keep Shorts focused on the newest answer; multiple overlapping
        # keywords often map to the same concern and make the query noisy.
        return matched[0]
    if re.search(r"\b(nothing|not sure|don't know|dont know|idk|what my query is about)\b", lower):
        return "student wellbeing"
    # Unknown wording must not be copied into the third-party search query.
    return "student wellbeing"

@app.post("/api/shorts")
async def related_shorts(payload: dict[str, Any]):
    context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    student_turns = [str(m.get("content", "")) for m in messages if isinstance(m, dict) and m.get("role") == "user" and str(m.get("content", "")).strip()]
    latest = student_turns[-1] if student_turns else str(context.get("main_topic") or context.get("problem") or "student wellbeing")
    topic = _short_topic(latest)
    if topic == "student wellbeing":
        topic = _short_topic(str(context.get("main_topic") or context.get("problem") or "student wellbeing"))
    # Send only the normalized topic category to YouTube. Never forward a
    # student's raw conversation or personal details to an external search.
    videos = await __import__("asyncio").to_thread(suggest_shorts, topic, 2)
    return {"topic": topic, "videos": videos}

class Turn(BaseModel):
    session_id: Optional[str] = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    message: str = Field(min_length=1, max_length=12000)

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message cannot be blank")
        return value

    @field_validator("messages")
    @classmethod
    def bounded_history(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(value) > 100:
            raise ValueError("conversation history is too long")
        safe = []
        for item in value:
            role, content = item.get("role"), item.get("content", "")
            if role not in {"user", "assistant"} or not isinstance(content, str) or len(content) > 12000:
                raise ValueError("conversation history contains an invalid message")
            safe.append({"role": role, "content": content})
        return safe

    @field_validator("context")
    @classmethod
    def bounded_context(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(json.dumps(value, ensure_ascii=False)) > 64000:
            raise ValueError("session context is too large")
        return value

class Ending(BaseModel):
    session_id: str = Field(min_length=36, max_length=36)
    messages: list[dict[str, Any]]
    context: dict[str, Any] = Field(default_factory=dict)

    @field_validator("messages")
    @classmethod
    def bounded_messages(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(value) > 200:
            raise ValueError("conversation history is too long")
        safe=[]
        for item in value:
            role, content = item.get("role"), item.get("content", "")
            if role not in {"user", "assistant"} or not isinstance(content, str) or len(content) > 12000:
                raise ValueError("conversation history contains an invalid message")
            safe.append({k:v for k,v in item.items() if k in {"role", "content", "passages", "interpretation", "follow_up"}})
        return safe

    @field_validator("context")
    @classmethod
    def bounded_context(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(json.dumps(value, ensure_ascii=False)) > 64000:
            raise ValueError("session context is too large")
        return value

@app.get("/")
def home():
    return FileResponse(os.path.join(ROOT, "static", "index.html"))

@app.get("/api/status")
def status():
    try:
        rag = rag_status()
    except Exception:
        rag = {"indexed_chunks": 0, "available": False}
    return {"groq_configured": bool(os.getenv("GROQ_API_KEY")), "groq_model": GROQ_MODEL, "rag": rag}

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/sessions")
def saved_sessions(limit: int = 16):
    limit = max(1, min(int(limit), 50))
    if not SESSION_PATH.exists():
        return {"sessions": []}
    records = []
    for path in SESSION_PATH.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(record, dict) and record.get("session_id"):
                records.append(record)
        except (OSError, json.JSONDecodeError):
            logger.warning("Skipping unreadable saved reflection record: %s", path.name)
    records.sort(key=lambda record: str(record.get("ended_at") or ""), reverse=True)
    return {"sessions": records[:limit]}

@app.post("/api/session/start")
def start_session(payload: dict[str, Any]):
    problem = str(payload.get("problem") or "").strip()[:12000]
    if not problem:
        raise HTTPException(status_code=422, detail="Tell Viveka what has been on your mind first.")
    session_id = str(uuid.uuid4())
    context = {
        "problem": problem,
        "main_topic": None,
        "questions_asked_count": 0,
        "questions_asked": [],
        "responses": [], "causes": [], "concerns": [], "sources_used": [],
    }
    try:
        result = reply(session_id, [{"role": "user", "content": problem}], context, problem, "exploration")
        answer = result.get("answer", {})
        question = str(answer.get("follow_up") or "").strip()
        if not question:
            raise RuntimeError("The mentor did not return a follow-up question.")
        update = answer.get("context_update") or {}
        context["main_topic"] = update.get("main_topic") or "Student reflection"
        context["questions_asked_count"] = 1
        context["questions_asked"] = [question]
        context["responses"] = [problem]
        _autosave_chat_turn(session_id, [{"role": "user", "content": problem}], context, problem,
                            {"role": "assistant", "content": answer.get("acknowledgement", ""), "follow_up": question})
        return {"session_id": session_id, "question": question, "acknowledgement": answer.get("acknowledgement", ""), "context": context}
    except Exception as exc:
        logger.exception("Could not generate the opening mentor question")
        raise HTTPException(status_code=503, detail="Viveka couldn’t reach the mentor AI to shape the first question. Check the server connection and try again.") from exc


def _needs_safety_check(text: str) -> bool:
    value = text.casefold()
    direct = re.search(r"\b(suicid\w*|self[ -]?harm|hurt myself|kill myself|end my life|want to die|don't want to (?:live|be here)|do not want to (?:live|be here))\b", value)
    hopeless = re.search(r"\b(hopeless|everything feels pointless|nothing matters|no point in living)\b", value)
    denial = re.search(r"\b(not|never|no longer)(?:\s+[\w']+){0,5}\s+(?:suicid\w*|hurting myself|hurt myself|self[ -]?harm)\b", value)
    return bool((direct or hopeless) and not denial)

def _autosave_chat_turn(session_id: str, messages: list[dict[str, Any]], context: dict[str, Any],
                        user_message: str, assistant_message: dict[str, Any] | None = None) -> None:
    """Keep ongoing and interrupted chats visible in My Reflections / the pond."""
    transcript = list(messages)
    if not transcript or transcript[-1].get("role") != "user" or transcript[-1].get("content") != user_message:
        transcript.append({"role": "user", "content": user_message})
    if assistant_message:
        transcript.append(assistant_message)
    saved_context = dict(context)
    responses = saved_context.get("responses", []) or []
    if not isinstance(responses, list):
        responses = [str(responses)]
    saved_context["responses"] = list(dict.fromkeys([*responses, user_message]))
    if not saved_context.get("outcome"):
        saved_context["outcome"] = "Conversation in progress; saved automatically."
    try:
        save_session(session_id, transcript[-200:], saved_context)
    except Exception:
        logger.exception("Could not autosave conversation %s", session_id)

@app.post("/api/chat")
def chat(turn: Turn):
    if turn.session_id:
        try:
            session_id = str(uuid.UUID(turn.session_id))
        except (ValueError, AttributeError):
            raise HTTPException(status_code=422, detail="session_id must be a valid UUID")
    else:
        session_id = str(uuid.uuid4())
    try:
        question_count = max(0, min(6, int(turn.context.get("questions_asked_count", 0) or 0)))
        conversation_choice = turn.context.get("conversation_choice")
        risk_text = f"{turn.context.get('problem', '')} {turn.message}"
        if not turn.context.get("safety_check_asked") and _needs_safety_check(risk_text):
            safety_question = "Are you in immediate danger right now, or thinking about hurting yourself?"
            prior_questions = turn.context.get("questions_asked", []) or []
            if not isinstance(prior_questions, list):
                prior_questions = [str(prior_questions)]
            prior_responses = turn.context.get("responses", []) or []
            if not isinstance(prior_responses, list):
                prior_responses = [str(prior_responses)]
            context = {
                **turn.context,
                "session_id": session_id,
                "safety_check_asked": True,
                "questions_asked": list(dict.fromkeys([*prior_questions, safety_question])),
                "questions_asked_count": min(6, question_count + 1),
                "responses": list(dict.fromkeys([*prior_responses, turn.message])),
                "conversation_stage": "safety check",
            }
            _autosave_chat_turn(session_id, turn.messages, context, turn.message,
                                {"role": "assistant", "content": "I’m really glad you told me. You deserve support with this, and you don’t have to carry it alone.", "follow_up": safety_question})
            return {
                "session_id": session_id,
                "phase": "exploration",
                "should_continue": True,
                "acknowledgement": "I’m really glad you told me. You deserve support with this, and you don’t have to carry it alone. If you’re in India, Tele-MANAS is available free, 24/7 at 14416. Please reach out to someone you trust or a qualified professional too. If you’re in immediate danger, contact local emergency services and stay with someone you trust.",
                "reply": "",
                "reflectionQuestion": safety_question,
                "context": context,
                "sources": [],
            }
        if question_count >= 6 and conversation_choice not in {"guidance", "explore"}:
            context = {**turn.context, "session_id": session_id, "awaiting_session_choice": True}
            choice_message = "Thank you for sharing that with me. We’ve explored this together. Would you like to hear a thought grounded in Vivekananda’s writings, or keep exploring what matters to you?"
            _autosave_chat_turn(session_id, turn.messages, context, turn.message,
                                {"role": "assistant", "content": choice_message})
            return {
                "session_id": session_id,
                "phase": "choice",
                "should_continue": False,
                "acknowledgement": choice_message,
                "context": context,
            }
        phase = "exploration" if question_count < 6 or conversation_choice == "explore" else "guidance"
        result = reply(session_id, turn.messages, turn.context, turn.message, phase)
        answer = result["answer"]
        update = answer.get("context_update") or {}
        context = {**turn.context, **update}
        # Preserve accumulated facts even if the model only returns this turn's details.
        for key in ("responses", "causes", "concerns", "guidance"):
            prior = turn.context.get(key, []) or []
            fresh = update.get(key, []) or []
            if not isinstance(prior, list): prior = [prior]
            if not isinstance(fresh, list): fresh = [fresh]
            context[key] = list(dict.fromkeys([*prior, *fresh]))
        if update.get("problem") is None: context["problem"] = turn.context.get("problem")
        if update.get("main_topic") is None: context["main_topic"] = turn.context.get("main_topic")
        if update.get("next_step") is None: context["next_step"] = turn.context.get("next_step")
        if update.get("outcome") is None: context["outcome"] = turn.context.get("outcome")
        prior_responses = turn.context.get("responses", []) or []
        if not isinstance(prior_responses, list): prior_responses = [prior_responses]
        context["responses"] = list(dict.fromkeys([*prior_responses, turn.message]))
        prior_questions = turn.context.get("questions_asked", []) or []
        if not isinstance(prior_questions, list): prior_questions = [prior_questions]
        if answer.get("follow_up"):
            context["questions_asked"] = list(dict.fromkeys([*prior_questions, answer["follow_up"]]))
        context["questions_asked_count"] = min(6, question_count + (1 if phase == "exploration" and answer.get("follow_up") else 0))
        prior_sources = turn.context.get("sources_used", []) or []
        seen_sources = {(x.get("book"), x.get("page"), x.get("chunk_id")) for x in prior_sources if isinstance(x, dict)} if isinstance(prior_sources, list) else set()
        current_sources = answer.get("passages", []) or []
        # Keep only source references in session context. Full passage text is
        # already attached to the assistant turn for display/export; copying it
        # into context caused every later LLM request to grow until Groq 413.
        source_refs = [
            {key: passage.get(key) for key in ("book", "volume", "chapter", "section", "page", "chunk_id")}
            for passage in current_sources
        ]
        prior_refs = [
            {key: passage.get(key) for key in ("book", "volume", "chapter", "section", "page", "chunk_id")}
            for passage in prior_sources if isinstance(passage, dict)
        ] if isinstance(prior_sources, list) else []
        context["sources_used"] = [
            *prior_refs,
            *[x for x in source_refs if (x.get("book"), x.get("page"), x.get("chunk_id")) not in seen_sources],
        ]
        context["conversation_stage"] = "listening" if phase == "exploration" else "reflecting and guiding"
        context["conversation_choice"] = None
        context["awaiting_session_choice"] = False
        message = {"role": "assistant", **answer}
        _autosave_chat_turn(session_id, turn.messages, context, turn.message, message)
        visible_answer = "\n\n".join(filter(None, [answer.get("acknowledgement"), answer.get("source_supported"), answer.get("interpretation"), answer.get("follow_up") or answer.get("reflection_question")]))
        sources = [{"title": p.get("book"), "volume": p.get("volume"), "chapter": p.get("chapter"), "section": p.get("section"), "page": p.get("page"), "text": p.get("text"), "chunk_id": p.get("chunk_id")} for p in answer.get("passages", [])]
        return {"session_id": session_id, "phase": phase, "should_continue": phase == "exploration", "reply": answer.get("acknowledgement", "") if phase == "exploration" else (answer.get("source_supported") or ""), "sourceType": "documented" if sources else None, "sources": sources, "interpretation": answer.get("interpretation", ""), "reflectionQuestion": answer.get("follow_up", "") if phase == "exploration" else answer.get("reflection_question", ""), "reflectionOptions": answer.get("reflection_options", []), "nextAction": answer.get("next_action") or context.get("next_step") or "", "acknowledgement": answer.get("acknowledgement", ""), "answer": visible_answer, "follow_up_question": answer.get("follow_up"), "message": message, "context": context}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Chat request failed")
        _autosave_chat_turn(session_id, turn.messages, turn.context, turn.message)
        if "GROQ_API_KEY" in str(exc):
            safe = "Groq API key is missing. Add GROQ_API_KEY=your_new_key to the project's .env file, then restart uvicorn. Do not paste the key into chat."
        elif isinstance(exc, AuthenticationError):
            safe = "Groq rejected the API key. Check that the newly generated GROQ_API_KEY in .env is valid, then restart uvicorn."
        elif isinstance(exc, RateLimitError):
            safe = "Groq rate limit or quota reached. Check your Groq account usage, then try again shortly."
        elif isinstance(exc, APIStatusError):
            code = getattr(exc, "status_code", None)
            if code == 413:
                safe = "The mentor request exceeded Groq's input limit. Conversation memory and book excerpts are now size-limited; restart the server and retry in a fresh chat if this message persists."
            elif code == 404:
                safe = f"Groq model '{GROQ_MODEL}' was not found or is unavailable to this account. Change GROQ_MODEL in .env and restart."
            elif code and code >= 500:
                safe = "Groq is temporarily unavailable. Please try again shortly."
            elif code == 400:
                reason = _groq_error_message(exc)
                safe = f"Groq rejected the request (HTTP 400): {reason}" if reason else "Groq rejected the chat request (HTTP 400). Check the server terminal for the validation reason."
            else:
                safe = f"Groq returned an API error (HTTP {code or 'unknown'}). Check the server terminal for details."
        elif isinstance(exc, httpx.RequestError) or (isinstance(exc, RuntimeError) and "client has been closed" in str(exc).lower()):
            safe = "Could not connect to Groq. Check your internet connection, firewall, or proxy, then retry."
        else:
            safe = "The mentor hit an internal error. Check the server terminal for details and retry."
        raise HTTPException(status_code=503, detail=safe) from exc

@app.post("/api/session/end")
def end_session(ending: Ending):
    try:
        session_id = str(uuid.UUID(ending.session_id))
        return save_session(session_id, ending.messages[:200], ending.context)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="session_id must be a valid UUID") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not save the session summary.") from exc

@app.post("/api/knowledge/ingest")
def ingest():
    try:
        return ingest_pdfs()
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Knowledge base ingestion failed. Check the PDF files and server logs.") from exc
