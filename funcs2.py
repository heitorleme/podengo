# funcs2.py
import os
import re
import io
import cv2
import math
import time
import json
import base64
import hashlib
import shutil
import random
import mimetypes
import threading
import subprocess
import tempfile
import warnings
from math import ceil
from pathlib import Path
from typing import Any, Iterable, Optional, List, Tuple, Set, Dict
from urllib.parse import urlparse, urlunparse, urlsplit, unquote
from tempfile import TemporaryDirectory, mkdtemp

import pydantic; print(pydantic.__version__)  # debug: mostra versão do pydantic
import pandas as pd
import requests
import tiktoken
from apify_client import ApifyClient
from filetype import guess
from instaloader import Post
import instaloader
import http.cookiejar
from pymongo import MongoClient
from pymongo.errors import BulkWriteError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from loguru import logger
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

# OpenAI (cliente moderno)
from openai import OpenAI, AsyncOpenAI, RateLimitError, APIConnectionError, APITimeoutError

# Whisper local é opcional; não exigimos para rodar (você usa a API do OpenAI)
try:
    import whisper  # opcional
except Exception:
    whisper = None

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Config / Settings com fallback para st.secrets + os.environ
# ─────────────────────────────────────────────────────────────
from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

REQUIRED_KEYS = [
    "APIFY_KEY",
    "OPENAI_API_KEY",
    "MONGO_HOST",
    "MONGO_PORT",
    "MONGO_USER",
    "MONGO_PASSWORD",
    "MONGO_DB_NAME",
]

BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env.txt"),  # opcional local
        env_file_encoding="utf-8",
        extra="ignore",
    )

    RAPID_KEY: Optional[str] = None
    APIFY_KEY: SecretStr
    OPENAI_API_KEY: SecretStr
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    OPENAI_TRANSCRIBE_MODEL: str = "whisper-1"

    MONGO_HOST: str
    MONGO_PORT: int
    MONGO_USER: str
    MONGO_PASSWORD: str
    MONGO_DB_NAME: str

_settings_cache: Optional[Settings] = None

def get_settings() -> Settings:
    """Carrega segredos de st.secrets (se houver) + env e valida via Pydantic."""
    global _settings_cache
    if _settings_cache is not None:
        return _settings_cache

    env = dict(os.environ)

    # Merge com st.secrets (Streamlit Cloud / local .streamlit/secrets.toml)
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for k, v in st.secrets.items():
                env.setdefault(k, str(v))
    except Exception:
        pass

    missing = [k for k in REQUIRED_KEYS if not env.get(k)]
    if missing:
        msg = (
            "Configuração ausente. Defina as chaves obrigatórias nas **App secrets** do Streamlit Cloud "
            "ou como variáveis de ambiente:\n  - " + "\n  - ".join(missing) +
            "\n\nNo Streamlit Cloud: Manage app ▸ Settings ▸ Secrets.\n"
        )
        raise RuntimeError(msg)

    # Instancia e deixa o Pydantic fazer cast/validação
    try:
        # Aceita também as variáveis opcionais de modelo
        allowed = set(REQUIRED_KEYS + ["RAPID_KEY", "OPENAI_CHAT_MODEL", "OPENAI_TRANSCRIBE_MODEL"])
        data = {k: env[k] for k in env if k.upper() in allowed}
        _settings_cache = Settings(**data)
        # Converte SecretStr -> str
        _settings_cache.APIFY_KEY = _settings_cache.APIFY_KEY.get_secret_value()  # type: ignore
        _settings_cache.OPENAI_API_KEY = _settings_cache.OPENAI_API_KEY.get_secret_value()  # type: ignore
        return _settings_cache
    except ValidationError as e:
        raise RuntimeError(f"Falha ao validar configurações: {e}") from e

# ─────────────────────────────────────────────────────────────
# Clientes preguiçosos (OpenAI / Apify) criados on-demand
# ─────────────────────────────────────────────────────────────
_apify_client: Optional[ApifyClient] = None
_openai_sync: Optional[OpenAI] = None
_openai_async: Optional[AsyncOpenAI] = None

def get_clients() -> Tuple[ApifyClient, OpenAI, AsyncOpenAI]:
    global _apify_client, _openai_sync, _openai_async
    s = get_settings()

    if _apify_client is None:
        _apify_client = ApifyClient(s.APIFY_KEY)  # type: ignore[arg-type]

    if _openai_sync is None:
        _openai_sync = OpenAI(api_key=s.OPENAI_API_KEY)  # type: ignore[arg-type]

    if _openai_async is None:
        _openai_async = AsyncOpenAI(api_key=s.OPENAI_API_KEY)  # type: ignore[arg-type]

    return _apify_client, _openai_sync, _openai_async

# Caminho para mídia temporária
media = str(BASE_DIR / "media")

# ─────────────────────────────────────────────────────────────
# Util / Logging
# ─────────────────────────────────────────────────────────────
PRINT_LOCK = threading.Lock()
def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def tlog(msg: str) -> None:
    with PRINT_LOCK:
        print(f"[{_ts()}] {msg}", flush=True)

# ─────────────────────────────────────────────────────────────
# Mongo helpers
# ─────────────────────────────────────────────────────────────
MONGO_FIELDS  = ["url", "ownerUsername", "caption", "type", "timestamp", "transcricao", "framesDescricao"]
MEDIA_FIELDS  = ["mediaUrl", "mediaLocalPath", "mediaLocalPaths"]
OTHER_FIELDS  = ["likesCount", "commentsCount", "videoPlayCount"]
OUTPUT_FIELDS = MONGO_FIELDS + MEDIA_FIELDS + OTHER_FIELDS

def _connect_to_mongo(
    HOST: Optional[str] = None,
    PORT: Optional[int] = None,
    USERNAME: Optional[str] = None,
    PASSWORD: Optional[str] = None,
    DATABASE: Optional[str] = None,
):
    try:
        s = get_settings()
        host = HOST or s.MONGO_HOST
        port = PORT or s.MONGO_PORT
        user = USERNAME or s.MONGO_USER
        pwd  = PASSWORD or s.MONGO_PASSWORD
        dbname = DATABASE or s.MONGO_DB_NAME

        client_mongo = MongoClient(host, port, username=user, password=pwd)
        db = client_mongo[dbname]
        logger.info(f"Connected to MongoDB: {dbname}")
        return db
    except Exception as e:
        logger.error(f"Erro conectando ao MongoDB: {e}")
        return None

def _mongo_collection():
    db = _connect_to_mongo()
    if db is None:
        raise RuntimeError("Não foi possível conectar ao MongoDB (verifique segredos).")
    return db["video-transcript"]

def _load_docs_by_urls(urls: List[str], fields: List[str] = MONGO_FIELDS) -> Dict[str, dict]:
    if not urls:
        return {}
    col = _mongo_collection()
    projection = {f: 1 for f in fields}
    projection["_id"] = 0
    cursor = col.find({"url": {"$in": urls}}, projection)
    out: Dict[str, dict] = {}
    for doc in cursor:
        u = doc.get("url")
        if u:
            out[u] = doc
    return out

def _upload_para_mongo(resultado: dict):
    try:
        col = _mongo_collection()
        if "url" not in resultado:
            logger.error("Erro: dicionário não contém a chave 'url'.")
            return None
        if col.find_one({"url": resultado["url"]}):
            print(f"Item com url '{resultado['url']}' já existe na base. Upload ignorado.")
            return None
        col.insert_one(resultado)
        preview = str(resultado)[:20]
        print(f"Item {preview}... uploadado com sucesso")
    except Exception as e:
        logger.error(f"Erro no upload: {e}")
        return None

def upload_muitos_para_mongo(resultados: List[dict]) -> dict:
    col = _mongo_collection()
    if not resultados:
        return {"inserted": 0, "skipped": 0}

    try:
        col.create_index("url", unique=True, name="uniq_url")
    except Exception:
        pass

    resultados = [r for r in resultados if isinstance(r, dict) and r.get("url")]

    seen = set()
    deduped = []
    for r in resultados:
        u = r["url"]
        if u not in seen:
            deduped.append(r)
            seen.add(u)

    existentes = set(x["url"] for x in col.find({"url": {"$in": list(seen)}}, {"url": 1, "_id": 0}))
    novos = [r for r in deduped if r["url"] not in existentes]

    if not novos:
        return {"inserted": 0, "skipped": len(resultados)}

    try:
        res = col.insert_many(novos, ordered=False)
        return {"inserted": len(res.inserted_ids), "skipped": len(resultados) - len(novos)}
    except BulkWriteError as e:
        ok = e.details.get("nInserted", 0)
        return {"inserted": ok, "skipped": len(resultados) - ok}

# ─────────────────────────────────────────────────────────────
# Áudio / Transcrição
# ─────────────────────────────────────────────────────────────
class AudioModel:
    """Wrapper mínimo para transcrição usando API do OpenAI (whisper-1)."""
    def __init__(self, model: Optional[str] = None):
        s = get_settings()
        self.model = model or s.OPENAI_TRANSCRIBE_MODEL

    def transcribe(self, file_path: str) -> dict:
        try:
            _, client, _ = get_clients()
            with open(file_path, "rb") as f:
                r = client.audio.transcriptions.create(model=self.model, file=f)
            return r.model_dump() if hasattr(r, "model_dump") else {"text": getattr(r, "text", None)}
        except Exception as e:
            return {"error": str(e)}

def _tipo_midia(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or ""
    if mime.startswith("video"):
        return "video"
    if mime.startswith("image"):
        return "image"
    ext = path.lower()
    if ext.endswith((".mp4", ".mov", ".mpeg4", ".webm")):
        return "video"
    if ext.endswith((".png", ".jpeg", ".jpg", ".webp", ".gif", ".heic", ".heif")):
        return "image"
    kind = guess(path)
    if kind:
        if kind.mime.startswith("video"):
            return "video"
        if kind.mime.startswith("image"):
            return "image"
    return "desconhecido"

def _download_file(url: str, pasta_destino: str = media) -> str:
    os.makedirs(pasta_destino, exist_ok=True)
    nome = os.path.basename(urlparse(url).path) or "media"
    if not os.path.splitext(nome)[1]:
        nome += ".mp4"  # fallback
    caminho = os.path.join(pasta_destino, nome)
    r = requests.get(url, allow_redirects=True, stream=True, timeout=30)
    r.raise_for_status()
    with open(caminho, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    return caminho

def _build_atempo_chain(speed: float) -> str:
    if speed <= 0:
        raise ValueError("speed deve ser > 0")
    factors: List[float] = []
    s = speed
    while s > 2.0:
        factors.append(2.0); s /= 2.0
    while s < 0.5:
        factors.append(0.5); s /= 0.5
    if not math.isclose(s, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        s = max(0.5, min(2.0, s)); factors.append(s)
    if not factors:
        factors = [1.0]
    return ",".join(f"atempo={f}" for f in factors)

def _run_ffmpeg(cmd_args: list):
    proc = subprocess.run(
        cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"FFmpeg falhou:\n{err}")

def get_video_frames(path: str, every_nth: int = 1) -> List[str]:
    if cv2 is None:
        raise ImportError("opencv-python não instalado para get_video_frames")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir vídeo: {path}")
    frames_b64: List[str] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_nth == 0:
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                frames_b64.append(base64.b64encode(buf).decode("utf-8"))
        idx += 1
    cap.release()
    return frames_b64

def _audio_temp_speed_from_video_ffmpeg(
    video_path: str,
    speed: float = 2.0,
    sample_rate: int = 16000,
    mono: bool = True,
) -> Tuple[str, str, TemporaryDirectory]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {video_path}")
    atempo_chain = _build_atempo_chain(speed)
    tmpdir_ctx = TemporaryDirectory(prefix="transc_")
    tmpdir = tmpdir_ctx.name
    out_wav = os.path.join(tmpdir, f"audio_{speed}x.wav")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path, "-vn",
        "-filter:a", atempo_chain,
        "-ar", str(sample_rate),
    ]
    if mono:
        cmd += ["-ac", "1"]
    cmd += [out_wav]
    _run_ffmpeg(cmd)
    return tmpdir, out_wav, tmpdir_ctx

def transcrever_video_em_speed(
    video_path: str, speed: float = 2.0, sample_rate: int = 16000,
) -> str:
    tmpdir, wav_path, tmpdir_ctx = _audio_temp_speed_from_video_ffmpeg(
        video_path, speed=speed, sample_rate=sample_rate
    )
    try:
        resp = AudioModel().transcribe(wav_path)
        return resp.get("text", "") if isinstance(resp, dict) else (resp or "")
    finally:
        try:
            tmpdir_ctx.cleanup()
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────
# Apify / Scrapers
# ─────────────────────────────────────────────────────────────
INSTAGRAM_ACTOR = "apify/instagram-scraper"
TIKTOK_ACTOR    = "clockworks/tiktok-scraper"
POLL_SEC = 2  # polling em segundos

def _shortcode(u: str) -> Optional[str]:
    m = re.search(r"instagram\.com/(?:reel|p)/([^/?#]+)", u, re.I)
    return m.group(1) if m else None

def _page_items(page) -> List[dict]:
    if hasattr(page, "items"):
        return getattr(page, "items") or []
    if hasattr(page, "model_dump"):
        try:
            data = page.model_dump()
            if isinstance(data, dict):
                return data.get("items", []) or []
        except Exception:
            pass
    if isinstance(page, dict):
        return page.get("items", []) or []
    return []

def _safe_filename_from_url(u: str) -> str:
    parsed = urlparse(u)
    base = os.path.basename(parsed.path) or ""
    if not base or "." not in base:
        h = hashlib.sha256(u.encode("utf-8")).hexdigest()[:16]
        base = f"static_{h}.mp4"
    return base

def _download_static_resource(u: str, pasta_destino: str) -> str:
    os.makedirs(pasta_destino, exist_ok=True)
    fname = _safe_filename_from_url(u)
    dest_path = os.path.join(pasta_destino, fname)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return dest_path
    with requests.get(u, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
    return dest_path

def _fetch_static_resources(urls_static: List[str], max_results: int) -> List[Dict[str, Optional[str]]]:
    posts = list(dict.fromkeys(urls_static))
    if not posts:
        tlog("[STATIC] Nenhuma URL estática válida encontrada.")
        return []
    out: List[Dict[str, Optional[str]]] = []
    for u in posts:
        if len(out) >= max_results:
            tlog(f"[STATIC] ⛔ Atingiu max_results={max_results}")
            break
        try:
            local_path = None
            try:
                local_path = _download_static_resource(u, pasta_destino=media)
            except Exception as e:
                tlog(f"[STATIC] ⚠️ Falha ao baixar {u}: {e}")
            payload: Dict[str, Optional[str]] = {"url": u, "mediaUrl": u}
            if local_path:
                payload["mediaLocalPath"] = local_path
                payload["mediaLocalPaths"] = [local_path]
            else:
                payload["mediaLocalPath"] = None
                payload["mediaLocalPaths"] = None
            out.append(payload)
        except Exception as e:
            tlog(f"[STATIC] ❌ Erro processando {u}: {e}")
    tlog(f"[STATIC] 🏁 Concluído com {len(out)} resultado(s)")
    return out

def _fetch_instagram(urls_instagram: List[str], api_token: str, max_results: int) -> List[Dict[str, Optional[str]]]:
    apify_client, _, _ = get_clients()
    posts = [u for u in urls_instagram if _shortcode(u)]
    posts = list(dict.fromkeys(posts))
    if not posts:
        tlog("[IG] Nenhuma URL de post válida encontrada.")
        return []
    tlog(f"[IG] ▶️ Iniciando run único com {len(posts)} URL(s) | actor={INSTAGRAM_ACTOR}")
    run = apify_client.actor(INSTAGRAM_ACTOR).start(
        run_input={
            "directUrls": posts,
            "resultsType": "details",
            "addParentData": False,
        }
    )
    run_id = run.get("id")
    if not run_id:
        tlog("[IG] ❌ Não recebi run_id; abortando.")
        return []
    wanted_sc = {s for s in (_shortcode(u) for u in posts) if s}
    wanted_set = set(posts)
    ds = None
    ds_id: Optional[str] = None
    offset = 0
    out: List[Dict[str, Optional[str]]] = []
    no_new_ticks = 0

    while True:
        info = apify_client.run(run_id).get()
        status = info.get("status")
        if not ds_id:
            ds_id = info.get("defaultDatasetId")
            if ds_id:
                ds = apify_client.dataset(ds_id)
                tlog(f"[IG] 🟡 status={status}, dataset={ds_id}")
        if ds_id:
            page = ds.list_items(limit=200, offset=offset, clean=True)
            items = _page_items(page)
            if items:
                tlog(f"[IG] 📥 +{len(items)} novos item(ns) (offset {offset}→{offset+len(items)})")
                offset += len(items)

                def _match(i: dict) -> bool:
                    iu = i.get("inputUrl") or i.get("url") or ""
                    if iu in wanted_set:
                        return True
                    sc_i = i.get("shortCode") or _shortcode(iu)
                    return bool(sc_i and sc_i in wanted_sc)

                items = [i for i in items if _match(i)]

                def _ensure_list(x):
                    if not x: return []
                    if isinstance(x, (list, tuple)): return list(x)
                    return [x]

                for it in items:
                    try:
                        media_candidates = it.get("videoUrl") or it.get("displayUrl") or it.get("images")
                        m_urls = _ensure_list(media_candidates)
                        local_paths: List[str] = []
                        for mu in m_urls:
                            try:
                                lp = _download_file(mu, pasta_destino=media)
                                if lp:
                                    local_paths.append(lp)
                            except Exception as e:
                                tlog(f"[IG] ⚠️ Falha ao baixar {mu}: {e}")
                        out.append({
                            "url": it.get("inputUrl") or it.get("url"),
                            "ownerUsername": it.get("ownerUsername"),
                            "likesCount": it.get("likesCount"),
                            "commentsCount": it.get("commentsCount"),
                            "caption": it.get("caption"),
                            "type": it.get("type"),
                            "timestamp": it.get("timestamp"),
                            "videoPlayCount": it.get("videoPlayCount"),
                            "videoViewCount": it.get("videoViewCount"),
                            "mediaUrl": m_urls if len(m_urls) > 1 else (m_urls[0] if m_urls else None),
                            "mediaLocalPaths": local_paths if len(local_paths) > 1 else None,
                            "mediaLocalPath": local_paths[0] if local_paths else None,
                        })
                        if len(out) % 25 == 0:
                            tlog(f"[IG] ✅ {len(out)} item(ns) processado(s)")
                        if len(out) >= max_results:
                            tlog(f"[IG] ⛔ Atingiu max_results={max_results}")
                            return out[:max_results]
                    except Exception as e:
                        tlog(f"[IG] ❌ Erro processando item: {e}")
                no_new_ticks = 0
            else:
                no_new_ticks += 1

        if status in {"RUNNING", "READY"}:
            tlog(f"[IG] ⏳ status={status}, recebidos={len(out)}")
        else:
            tlog(f"[IG] 🧭 status final={status}, recebidos={len(out)}")
            if no_new_ticks >= 2:
                break
        time.sleep(POLL_SEC)

    tlog(f"[IG] 🏁 Concluído com {len(out)} resultado(s)")
    return out

def _tt_id(u: str) -> Optional[str]:
    m = re.search(r"/video/(\d+)", u)
    return m.group(1) if m else None

def _fetch_tiktok(urls_tiktok: List[str], api_token: str, max_results: int) -> List[Dict[str, Optional[str]]]:
    apify_client, _, _ = get_clients()
    posts = [u for u in urls_tiktok if _tt_id(u)]
    posts = list(dict.fromkeys(posts))
    if not posts:
        tlog("[TT] Nenhuma URL de post válida encontrada.")
        return []
    tlog(f"[TT] ▶️ Iniciando run único com {len(posts)} URL(s) | actor={TIKTOK_ACTOR}")
    run = apify_client.actor(TIKTOK_ACTOR).start(
        run_input={
            "postURLs": posts,
            "shouldDownloadVideos": True,
            "shouldDownloadSlideshowImages": True,
        }
    )
    run_id = run.get("id")
    if not run_id:
        tlog("[TT] ❌ Não recebi run_id; abortando.")
        return []

    wanted_ids = {vid for vid in (_tt_id(u) for u in posts) if vid}
    wanted_set = set(posts)

    def _any_post_url(i: dict) -> str:
        for k in ("submittedVideoUrl", "webVideoUrl", "url", "shareUrl"):
            v = i.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    def _match(i: dict) -> bool:
        iu = _any_post_url(i)
        if not iu:
            return False
        if iu in wanted_set:
            return True
        vid = _tt_id(iu)
        return bool(vid and vid in wanted_ids)

    ds = None
    ds_id: Optional[str] = None
    offset = 0
    items_total: List[dict] = []
    no_new_ticks = 0

    while True:
        info = apify_client.run(run_id).get()
        status = info.get("status")
        if not ds_id:
            ds_id = info.get("defaultDatasetId")
            if ds_id:
                ds = apify_client.dataset(ds_id)
                tlog(f"[TT] 🟡 status={status}, dataset={ds_id}")

        if ds_id:
            page = ds.list_items(limit=200, offset=offset, clean=True)
            page_items = _page_items(page)
            if page_items:
                tlog(f"[TT] 📥 +{len(page_items)} novos item(ns) (offset {offset}→{offset+len(page_items)})")
                offset += len(page_items)
                items_total.extend(page_items)
                no_new_ticks = 0
            else:
                no_new_ticks += 1

        if status not in {"RUNNING", "READY"}:
            tlog(f"[TT] 🧭 status final={status}, recebidos={len(items_total)}")
            if no_new_ticks >= 2:
                break
        else:
            tlog(f"[TT] ⏳ status={status}, recebidos parciais={len(items_total)}")
        time.sleep(POLL_SEC)

    if not items_total:
        tlog("[TT] ⚠️ Nenhum item retornado pelo actor.")
        return []

    tlog("[TT] ⏱ aguardando 2s antes de iniciar downloads…")
    time.sleep(2)

    out: List[Dict[str, Optional[str]]] = []
    for it in (i for i in items_total if _match(i)):
        try:
            def _ensure_list(x):
                if not x: return []
                if isinstance(x, (list, tuple)): return list(x)
                return [x]

            media_candidates: List[str] = []
            media_candidates += [u for u in _ensure_list(it.get("mediaUrls")) if isinstance(u, str) and u.strip()]
            dl_root = it.get("downloadAddr")
            if isinstance(dl_root, str) and dl_root.strip():
                media_candidates.append(dl_root.strip())
            dl_vm = (it.get("videoMeta") or {}).get("downloadAddr")
            if isinstance(dl_vm, str) and dl_vm.strip():
                media_candidates.append(dl_vm.strip())

            seen = set()
            media_candidates = [u for u in media_candidates if not (u in seen or seen.add(u))]

            local_paths: List[str] = []
            first_ok_path: Optional[str] = None
            is_slideshow = bool(it.get("isSlideshow"))

            for mu in media_candidates:
                try:
                    lp = _download_file(mu, pasta_destino=media)
                    if lp and os.path.isfile(lp):
                        local_paths.append(lp)
                        if not first_ok_path:
                            first_ok_path = lp
                        tlog(f"[TT][DL] ✅ ok → {os.path.basename(lp)}")
                        if not is_slideshow:
                            break
                    else:
                        tlog(f"[TT][DL] ⚠️ sem arquivo após download: {mu.split('?')[0]}")
                except Exception as e:
                    tlog(f"[TT][DL] ⚠️ falhou: {mu.split('?')[0]} | {e}")

            _type = "Slideshow" if is_slideshow or len(media_candidates) > 1 else "Video"

            out.append({
                "url": _any_post_url(it) or it.get("webVideoUrl") or it.get("url"),
                "ownerUsername": ((it.get("authorMeta") or {}).get("name")) or it.get("authorUniqueId"),
                "likesCount": it.get("diggCount"),
                "commentsCount": it.get("commentCount"),
                "caption": it.get("text") or it.get("description"),
                "type": _type,
                "timestamp": it.get("createTimeISO") or it.get("createTime"),
                "videoPlayCount": it.get("playCount"),
                "videoViewCount": it.get("playCount"),
                "mediaUrl": media_candidates if len(media_candidates) > 1 else (media_candidates[0] if media_candidates else None),
                "mediaLocalPaths": local_paths if len(local_paths) > 1 else None,
                "mediaLocalPath": first_ok_path,
            })

            if media_candidates and not first_ok_path:
                tlog(f"[TT] ⚠️ Nenhum arquivo salvo — candidatos={len(media_candidates)}; ex={media_candidates[0].split('?')[0]}")

            if len(out) >= max_results:
                tlog(f"[TT] ⛔ Atingiu max_results={max_results}")
                return out[:max_results]
        except Exception as e:
            keys = sorted(list(it.keys()))
            tlog(f"[TT] ❌ Erro processando item: {e} | keys={keys}")

    tlog(f"[TT] 🏁 Concluído com {len(out)} resultado(s)")
    return out

# ─────────────────────────────────────────────────────────────
# Pipeline principal (busca → transcrição → limpeza)
# ─────────────────────────────────────────────────────────────
def _partition_known_unknown_by_mongo(all_urls: List[str]) -> Tuple[Dict[str, dict], Set[str]]:
    known_map = _load_docs_by_urls(all_urls, MONGO_FIELDS)
    known_urls = set(known_map.keys())
    unknown_set = set(u for u in all_urls if u not in known_urls)
    return known_map, unknown_set

def _merge_results_in_input_order(
    input_urls: List[str],
    known_map: Dict[str, dict],
    fetched_new_items: List[Dict[str, Optional[str]]]
) -> List[Dict[str, Optional[str]]]:
    fetched_map = {item.get("url"): item for item in fetched_new_items or [] if item.get("url")}
    results: List[Dict[str, Optional[str]]] = []
    for u in input_urls:
        if u in known_map:
            doc = {k: known_map[u].get(k) for k in MONGO_FIELDS}
            doc["_from_mongo"] = True
            results.append(doc)
        else:
            it = fetched_map.get(u)
            if it:
                row = {k: it.get(k) for k in OUTPUT_FIELDS}
                row["_from_mongo"] = False
                results.append(row)
            else:
                results.append({"url": u, **{k: None for k in OUTPUT_FIELDS if k != "url"}, "_from_mongo": False})
    return results

def _tipo_midia_url(url: Optional[str]) -> str:
    if not url:
        return "desconhecido"
    p = urlparse(url)
    clean = urlunparse((p.scheme, p.netloc, p.path, "", "", ""))
    tipo = _tipo_midia(clean)
    if tipo == "desconhecido" and "dst-mp4" in url.lower():
        return "video"
    return tipo

def _pick_media_url(media: Any, media_local_path: Any = None, media_local_paths: Any = None) -> Optional[str]:
    if media:
        if isinstance(media, str) and media:
            return media
        if isinstance(media, list):
            for m in media:
                if isinstance(m, str) and m:
                    return m
        if isinstance(media, dict):
            for v in media.values():
                if isinstance(v, str) and v:
                    return v
                if isinstance(v, list):
                    for m in v:
                        if isinstance(m, str) and m:
                            return m
    if isinstance(media_local_path, str) and media_local_path:
        return media_local_path
    if isinstance(media_local_paths, (list, tuple)):
        for p in media_local_paths:
            if isinstance(p, str) and p:
                return p
    return None

def _descrever_frames(frames_b64: List[str], max_imgs: int = 5, idioma: str = "pt-BR") -> str:
    if not frames_b64:
        return ""
    try:
        imagens = frames_b64[:max_imgs]
        mensagens = [
            {
                "role": "user",
                "content": (
                    [{"type": "text", "text": f"Descreva em {idioma} o que aparece nestas imagens/frame de vídeo. "
                                               f"Foque em quem/que objetos aparecem, ações e contexto. Seja conciso. "
                                               f"Não descreva cada frame individualmente; faça um resumo da temática. "
                                               f"Mencione marcas/produtos específicos se aparecerem."}]
                    + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}} for b64 in imagens]
                ),
            }
        ]
        s = get_settings()
        _, client, _ = get_clients()
        resp = client.chat.completions.create(model=s.OPENAI_CHAT_MODEL, messages=mensagens)
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[ERRO descrição frames] {e}"

# ─────────────────────────────────────────────────────────────
# Transcrição em paralelo (threads) + descrição de frames
# ─────────────────────────────────────────────────────────────
def _thread_worker(idx: int, media_url: Optional[str], sem: threading.Semaphore):
    """
    Transcreve vídeo/imag. usando arquivo local (em ./media) e retorna:
      (idx, transcricao:str|None, base64Frames:list[str], framesDescricao:str|None, erro:str|None)
    """
    if not media_url:
        return idx, None, [], None, "sem_media_url"

    try:
        base_dir = Path(media)

        def _is_url(s: str) -> bool:
            return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))

        def _resolve_local_path(s: str) -> Path:
            if _is_url(s):
                path = urlsplit(s).path
                name = unquote(os.path.basename(path)) or "mediafile"
            else:
                name = s
            p = Path(name)
            return p if p.is_absolute() else (base_dir / name)

        local_path = _resolve_local_path(media_url)
        if not local_path.exists():
            return idx, None, [], None, f"FileNotFoundError: {local_path}"

        ext = local_path.suffix.lower()
        is_video = ext in {".mp4", ".mov", ".m4v", ".webm", ".ts", ".mkv", ".avi"}
        is_image = ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

        texto: Optional[str] = None
        base64_frames: List[str] = []
        frames_descricao: Optional[str] = None

        if is_video:
            tlog(f"[TR] 🎙️ idx={idx}: transcrição {local_path.name}")
            # 1) transcrição de áudio (serializada pelo semáforo)
            with sem:
                try:
                    _t0 = time.time()
                    texto = transcrever_video_em_speed(str(local_path))
                    tlog(f"[TR] ✅ idx={idx}: {local_path.name} em {time.time()-_t0:.1f}s")
                except Exception as e:
                    tlog(f"[TR] ❌ idx={idx}: {local_path.name}: {e}")
                    raise
            # 2) frames
            try:
                base64_frames = get_video_frames(str(local_path), every_nth=60)
            except Exception as fe:
                tlog(f"[frames] erro extraindo frames de {local_path}: {fe}")
                base64_frames = []
            # 3) descrição dos frames
            if base64_frames:
                frames_descricao = _descrever_frames(base64_frames)

        elif is_image:
            with open(local_path, "rb") as img:
                base64_frames = [base64.b64encode(img.read()).decode("utf-8")]
            frames_descricao = _descrever_frames(base64_frames)

        return idx, texto, base64_frames, frames_descricao, None

    except Exception as e:
        return idx, None, [], None, f"{type(e).__name__}: {e}"

def anexar_transcricoes_threaded(
    resultados: List[Dict[str, Any]],
    max_workers: int = 4,
    gpu_singleton: bool = True,
) -> List[Dict[str, Any]]:
    """
    Para cada item com mídia local, transcreve (vídeo) e descreve frames em paralelo.
    Mantém campos existentes vindos do Mongo; só preenche se houver conteúdo novo.
    """
    if not resultados:
        return resultados

    sem = threading.Semaphore(1 if gpu_singleton else max_workers)

    jobs: List[Tuple[int, Optional[str]]] = []
    for idx, item in enumerate(resultados):
        veio_mongo = bool(item.get("_from_mongo"))

        # Se já tem transcrição/frames do Mongo, não reprocessa
        if veio_mongo and (item.get("transcricao") is not None or item.get("framesDescricao") is not None):
            item.setdefault("base64Frames", [])
            continue

        url_media = _pick_media_url(
            item.get("mediaUrl"),
            item.get("mediaLocalPath"),
            item.get("mediaLocalPaths"),
        )
        if url_media:
            jobs.append((idx, url_media))
        else:
            # Sem media → marque erro mínimo somente se não houver nada preenchido
            if item.get("transcricao") is None and item.get("framesDescricao") is None:
                item.setdefault("base64Frames", [])
                item["transcricao"] = None
                item["transcricao_erro"] = "sem_media_url"
                item["framesDescricao"] = None

    if not jobs:
        return resultados

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_thread_worker, idx, url, sem) for idx, url in jobs]
        for fut in as_completed(futs):
            idx, transcricao, frames, frames_desc, erro = fut.result()

            if transcricao is not None:
                resultados[idx]["transcricao"] = transcricao
            if frames is not None:
                resultados[idx]["base64Frames"] = frames
            if frames_desc is not None:
                resultados[idx]["framesDescricao"] = frames_desc

            if erro:
                resultados[idx]["transcricao_erro"] = erro
            else:
                resultados[idx].pop("transcricao_erro", None)

    return resultados

def _coletar_caminhos_midia(resultados: List[dict]) -> Set[Path]:
    paths: Set[Path] = set()
    def _add(p):
        if p:
            pp = Path(p)
            if pp.exists() and pp.is_file():
                paths.add(pp)
    for item in resultados:
        _add(item.get("mediaLocalPath"))
        mlist = item.get("mediaLocalPaths")
        if isinstance(mlist, (list, tuple)):
            for p in mlist:
                _add(p)
    return paths

def _deletar_arquivos(paths: Iterable[Path]) -> None:
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except Exception as e:
            print(f"[CLEANUP] Falha ao deletar {p}: {e}")

def _deletar_pasta_se_vazia(pasta: Path) -> None:
    try:
        if pasta.exists() and pasta.is_dir() and not any(pasta.iterdir()):
            pasta.rmdir()
    except Exception as e:
        print(f"[CLEANUP] Falha ao remover pasta {pasta}: {e}")

async def fetch_social_post_summary_async(
    post_url: List[str],
    api_token: Optional[str] = None,
    max_results: int = 1000,
) -> List[Dict[str, Optional[str]]]:
    s = get_settings()
    api_token = api_token or s.APIFY_KEY  # type: ignore[attr-defined]

    input_urls = [u.strip() for u in post_url if u and u.strip()]
    if not input_urls:
        tlog("[FETCH] Nenhuma URL válida fornecida.")
        return []

    known_map, unknown_set = _partition_known_unknown_by_mongo(input_urls)

    urls_instagram = [u for u in input_urls if "instagram.com" in u and u in unknown_set]
    urls_tiktok    = [u for u in input_urls if "tiktok.com"   in u and u in unknown_set]
    urls_static    = [u for u in input_urls if "https://static-resources" in u and u in unknown_set]

    tlog(
        f"[FETCH] Instagram URLs novas: {len(urls_instagram)} | "
        f"TikTok URLs novas: {len(urls_tiktok)} | "
        f"Static URLs novas: {len(urls_static)} | "
        f"Já no Mongo: {len(known_map)}"
    )

    tasks = []
    if urls_instagram:
        tlog("[FETCH] Agendando fetch do Instagram…")
        tasks.append(asyncio.to_thread(_fetch_instagram, urls_instagram, api_token, max_results))
    if urls_tiktok:
        tlog("[FETCH] Agendando fetch do TikTok…")
        tasks.append(asyncio.to_thread(_fetch_tiktok, urls_tiktok, api_token, max_results))
    if urls_static:
        tlog("[FETCH] Agendando fetch de Static Resources (download direto)…")
        tasks.append(asyncio.to_thread(_fetch_static_resources, urls_static, max_results))

    fetched_new_items: List[Dict[str, Optional[str]]] = []
    if tasks:
        parts = await asyncio.gather(*tasks)
        for p in parts:
            fetched_new_items.extend(p)

    results = _merge_results_in_input_order(input_urls, known_map, fetched_new_items)
    if len(results) > max_results:
        results = results[:max_results]
    tlog(f"[FETCH] Finalizado: {len(results)} item(ns) total. (reutilizados={len(known_map)}, novos={len(fetched_new_items)})")
    return results

async def rodar_pipeline(urls: List[str]) -> List[dict]:
    """
    1) Busca/baixa os posts (TikTok/Instagram) salvando mídia em ./media
    2) Transcreve em paralelo (e extrai frames); anexa em 'transcricao', 'base64Frames'
    3) Deleta os arquivos de mídia locais
    4) Retorna a lista de resultados enriquecida
    """
    if urls is None or (isinstance(urls, (list, tuple, set)) and len(urls) == 0) \
       or (isinstance(urls, pd.Series) and urls.empty) \
       or (hasattr(urls, "empty") and urls.empty):
        print("Nenhuma URL fornecida.")
        return []

    resultados = await fetch_social_post_summary_async(urls, api_token=None, max_results=1000)
    if not resultados:
        print("Nenhum resultado retornado pelos scrapers.")
        return []

    anexar_transcricoes_threaded(resultados, max_workers=4, gpu_singleton=True)

    caminhos = _coletar_caminhos_midia(resultados)
    _deletar_arquivos(caminhos)
    _deletar_pasta_se_vazia(Path(media))

    return resultados

