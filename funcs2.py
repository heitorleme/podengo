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
from typing import Any, Iterable, Optional, List, Tuple, Set, Dict, Callable
from urllib.parse import urlparse, urlunparse, urlsplit, unquote
from tempfile import TemporaryDirectory, mkdtemp
import numpy as np

import pydantic; print(pydantic.__version__)  # debug: mostra versão do pydantic
import pandas as pd
import requests
import tiktoken
from apify_client import ApifyClient
from filetype import guess
from instaloader import Post
import instaloader
import http.cookiejar
from pymongo import MongoClient, UpdateOne
from datetime import datetime
from pymongo.errors import BulkWriteError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from loguru import logger
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import logging

# OpenAI (cliente moderno)
from openai import OpenAI, AsyncOpenAI, RateLimitError, APIConnectionError, APITimeoutError

# Whisper local é opcional; não exigimos para rodar (você usa a API do OpenAI)
try:
    import whisper  # opcional
except Exception:
    whisper = None

warnings.filterwarnings("ignore")

# Max workers
max_workers=os.cpu_count() or 8

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
    "MONGO_URI"
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
    OPENAI_CHAT_MODEL: str = "gpt-5-nano"
    OPENAI_TRANSCRIBE_MODEL: str = "whisper-1"

    MONGO_HOST: str
    MONGO_PORT: int
    MONGO_USER: str
    MONGO_PASSWORD: str
    MONGO_DB_NAME: str
    MONGO_URI: str

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

    REQUIRED_KEYS_BASE = ["APIFY_KEY", "OPENAI_API_KEY", "MONGO_URI"]
    missing = [k for k in REQUIRED_KEYS_BASE if not env.get(k)]

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
MONGO_FIELDS  = ["url", "ownerUsername", "caption", "type", "post_timestamp", "transcricao", "framesDescricao", "embedding", "categoria_principal", "subcategoria", "comunidade_predita", "comunidades_proporcoes"]
MEDIA_FIELDS  = ["mediaUrl", "mediaLocalPath", "mediaLocalPaths"]
OTHER_FIELDS  = [
    "likesCount", "commentsCount", "videoPlayCount", "videoViewCount",
    "audio_id", "audio_snapshot", "hashtags", "mentions",
    "ai_model_data"
]

MONGO_FETCH_FIELDS = MONGO_FIELDS + OTHER_FIELDS
OUTPUT_FIELDS = MONGO_FIELDS + MEDIA_FIELDS + OTHER_FIELDS

logger = logging.getLogger(__name__)

def _connect_to_mongo(
    HOST: Optional[str] = None,
    PORT: Optional[int] = None,
    USERNAME: Optional[str] = None,
    PASSWORD: Optional[str] = None,
    DATABASE: Optional[str] = None,
):
    """
    Conecta ao MongoDB usando MONGO_URI da configuração.
    Os parâmetros individuais (HOST, PORT, etc.) são mantidos apenas para compatibilidade,
    mas são ignorados se MONGO_URI estiver configurado.
    """
    try:
        s = get_settings()

        # 🔹 Prioriza MONGO_URI (nova lógica)
        if hasattr(s, "MONGO_URI") and s.MONGO_URI:
            client_mongo = MongoClient(s.MONGO_URI)
            db_name = s.MONGO_DB_NAME
            db = client_mongo[db_name]
            logger.info(f"Connected to MongoDB via URI: {db_name}")
            return db

        # 🔸 Fallback (modo legado)
        host = HOST or s.MONGO_HOST
        port = PORT or s.MONGO_PORT
        user = USERNAME or s.MONGO_USER
        pwd  = PASSWORD or s.MONGO_PASSWORD
        dbname = DATABASE or s.MONGO_DB_NAME

        client_mongo = MongoClient(host, port, username=user, password=pwd)
        db = client_mongo[dbname]
        logger.info(f"Connected to MongoDB (legacy mode): {dbname}")
        return db

    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {e}")
        return None

def _mongo_collection():
    db = _connect_to_mongo()
    if db is None:
        raise RuntimeError("Não foi possível conectar ao MongoDB (verifique segredos).")
    return db["video-transcript"]

def _load_docs_by_urls(urls: List[str], fields: List[str] = MONGO_FETCH_FIELDS) -> Dict[str, dict]:
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
            # garante que todas as chaves existam (None quando ausentes)
            clean = {f: doc.get(f) for f in fields}
            clean["url"] = u
            out[u] = clean
    return out

@retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
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


regex_hashtags  = re.compile(r"#\w+", re.UNICODE)
regex_mentions  = re.compile(r"@\w+", re.UNICODE)

def _dedup_preservando_ordem(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _extract_tags_and_mentions(caption: str):
    caption = caption or ""
    hashtags = _dedup_preservando_ordem([h.lstrip("#") for h in regex_hashtags.findall(caption)])
    mentions = _dedup_preservando_ordem([m.lstrip("@") for m in regex_mentions.findall(caption)])
    return hashtags, mentions

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

def get_video_frames(path: str, every_nth: Optional[int] = None) -> List[str]:
    """
    Extrai entre 5 e 15 frames de um vídeo usando FFmpeg, sem decodificar tudo com OpenCV.
    Retorna uma lista de strings base64 (imagens JPEG).
    É de 3x a 10x mais rápido que a versão anterior baseada em cv2.VideoCapture.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo de vídeo não encontrado: {path}")

    # Determina a duração e o FPS
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=duration,nb_frames,r_frame_rate",
                "-of", "json", path
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        info = json.loads(proc.stdout.decode("utf-8", errors="ignore"))
        stream = (info.get("streams") or [{}])[0]
        duration = float(stream.get("duration", 0))
    except Exception:
        duration = 0

    # Decide quantos frames pegar
    if duration < 15:
        num_frames_target = 8
    elif duration < 60:
        num_frames_target = 10
    elif duration < 180:
        num_frames_target = 12
    else:
        num_frames_target = 15

    # Intervalo (segundos) entre frames
    if duration and duration > 0:
        interval = max(duration / num_frames_target, 0.5)
    else:
        interval = 1.0

    # Cria diretório temporário
    tmpdir = Path(tempfile.mkdtemp(prefix="frames_ffmpeg_"))
    pattern = str(tmpdir / "frame_%03d.jpg")

    # Extração direta com FFmpeg (1 frame a cada N segundos)
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-y",
        "-i", path,
        "-vf", f"fps=1/{interval}",
        "-q:v", "3",
        pattern
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    # Lê os frames e converte para base64
    frames_b64: List[str] = []
    for jpg_path in sorted(tmpdir.glob("frame_*.jpg")):
        try:
            with open(jpg_path, "rb") as f:
                frames_b64.append(base64.b64encode(f.read()).decode("utf-8"))
            if len(frames_b64) >= num_frames_target:
                break
        except Exception:
            continue

    # Limpeza temporária
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass

    return frames_b64

def _audio_temp_speed_from_video_ffmpeg(
    video_path: str,
    speed: float = 2.0,
    sample_rate: int = 16000,
    mono: bool = True,
) -> Tuple[str, str]:
    """
    Extrai o áudio do vídeo (ajustando velocidade) para um diretório temporário persistente.
    Retorna (tmpdir, caminho_wav).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {video_path}")

    # 🔹 Usa diretório temporário persistente em vez de TemporaryDirectory()
    tmpdir = Path(BASE_DIR) / "tmp"
    tmpdir.mkdir(exist_ok=True)

    # Nome de saída único baseado no hash do vídeo
    hash_part = hashlib.sha1(video_path.encode("utf-8")).hexdigest()[:10]
    out_wav = tmpdir / f"audio_{hash_part}_{speed}x.wav"

    # Se já existir e for recente, reutiliza (opcional)
    if out_wav.exists() and out_wav.stat().st_size > 0:
        return str(tmpdir), str(out_wav)

    atempo_chain = _build_atempo_chain(speed)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path, "-vn",
        "-filter:a", atempo_chain,
        "-ar", str(sample_rate),
    ]
    if mono:
        cmd += ["-ac", "1"]
    cmd += [str(out_wav)]

    _run_ffmpeg(cmd)
    return str(tmpdir), str(out_wav)

def transcrever_video_em_speed(
    video_path: str, speed: float = 2.0, sample_rate: int = 16000,
) -> Tuple[str, Optional[float]]:
    tmpdir, wav_path = _audio_temp_speed_from_video_ffmpeg(
        video_path, speed=speed, sample_rate=sample_rate
    )
    dur_s = None
    try:
        dur_s = _ffprobe_duration_seconds(wav_path)
        resp = AudioModel().transcribe(wav_path)
        texto = resp.get("text", "") if isinstance(resp, dict) else (resp or "")
        return texto, dur_s
    finally:
        # ❌ Não deletamos aqui — limpeza global no final
        pass


def _ffprobe_duration_seconds(path: str) -> Optional[float]:
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        out = proc.stdout.decode("utf-8", errors="ignore").strip()
        return float(out) if out else None
    except Exception:
        return None

# Heurística: tokens "equivalentes" por tile 512x512
TOKENS_PER_TILE = 85

def _estimate_image_tokens_from_b64_list(frames_b64: List[str], tokens_per_tile: int = TOKENS_PER_TILE) -> Tuple[int, int]:
    """
    Retorna (num_images, estimated_image_tokens) calculando tiles de 512x512 para cada imagem.
    Se falhar ao decodificar alguma imagem, assume 1 tile para ela.
    """
    total_tiles = 0
    num_images = 0
    for b64 in frames_b64:
        try:
            data = base64.b64decode(b64)
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if img is None:
                tiles = 1
            else:
                h, w = img.shape[:2]
                tiles = ceil(w / 512) * ceil(h / 512)
            total_tiles += tiles
            num_images += 1
        except Exception:
            total_tiles += 1
            num_images += 1
    return num_images, total_tiles * tokens_per_tile

# ─────────────────────────────────────────────────────────────
# Apify / Scrapers
# ─────────────────────────────────────────────────────────────
INSTAGRAM_ACTOR = "apify/instagram-scraper"
TIKTOK_ACTOR    = "clockworks/tiktok-scraper"
POLL_SEC = 2  # polling em segundos

IG_PROFILE_RESERVED = {"p", "reel", "reels", "stories", "explore", "tv"}

APIFY_KV_COOLDOWN_SEC = 2  # tempo para o Apify materializar o arquivo no KV após o dataset
APIFY_KV_MAX_RETRY = 6      # tentativas (exponenciais)
APIFY_KV_INITIAL_DELAY = 2  # segundos

def _download_file_with_retry(url: str, pasta_destino: str = media,
                              max_retry: int = APIFY_KV_MAX_RETRY,
                              initial_delay: int = APIFY_KV_INITIAL_DELAY) -> str:
    """
    Baixa com retries exponenciais, tratando 404/403 de propagação do KV do Apify.
    Retorna caminho local ou lança a última exceção se falhar em todas as tentativas.
    """
    delay = initial_delay
    last_err = None
    for attempt in range(1, max_retry + 1):
        try:
            return _download_file(url, pasta_destino=pasta_destino)
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            last_err = e
            # 404/403 são os mais comuns durante a propagação; aguarda e tenta de novo
            if status in (403, 404):
                tlog(f"[DL][retry {attempt}/{max_retry}] {status} para {url.split('?')[0]} — aguardando {delay}s…")
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            # outros status → não insiste
            raise
        except Exception as e:
            last_err = e
            tlog(f"[DL][retry {attempt}/{max_retry}] erro '{type(e).__name__}' em {url.split('?')[0]} — aguardando {delay}s…")
            time.sleep(delay)
            delay = min(delay * 2, 30)
    # se chegou aqui, esgotou
    raise last_err if last_err else RuntimeError("Falha desconhecida no download")


def _any_post_url(i: dict) -> str:
    for k in ("submittedVideoUrl", "webVideoUrl", "url", "shareUrl", "inputUrl"):
        v = i.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def _is_ig_profile_url(u: Optional[str]) -> bool:
    if not u or "instagram.com" not in u:
        return False
    try:
        p = urlparse(u)
        segs = [s for s in (p.path or "").split("/") if s]
        # perfil simples: /<username>[/]
        return len(segs) == 1 and segs[0].lower() not in IG_PROFILE_RESERVED
    except Exception:
        return False

def _normalize_ig_profile_url(u: str) -> str:
    # força https e barra final
    if not re.match(r"^https?://", u, flags=re.I):
        u = "https://" + u
    p = urlparse(u.strip())
    path = (p.path or "/").rstrip("/") + "/"
    return urlunparse(("https", p.netloc.lower(), path, "", "", ""))

def _ig_post_url_from_item(i: dict) -> str:
    # tenta campos comuns; se só tiver shortCode, monta URL canônica
    if isinstance(i.get("inputUrl"), str) and i["inputUrl"]:
        return i["inputUrl"]
    if isinstance(i.get("url"), str) and i["url"]:
        return i["url"]
    sc = i.get("shortCode")
    if isinstance(sc, str) and sc:
        return f"https://www.instagram.com/p/{sc}/"
    return ""

def _normalize_url(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    p = urlparse(u)
    # remove query/fragment e barra final
    path = p.path.rstrip("/")
    return urlunparse((p.scheme.lower(), p.netloc.lower(), path, "", "", ""))

def _post_identity(u: Optional[str]) -> Optional[Tuple[str, str]]:
    """
    Retorna uma identidade canônica para merge:
      ("ig", <shortcode>) | ("tt", <video_id>) | ("url", <normalized_url>)
    """
    if not u:
        return None
    if "instagram.com" in u:
        sc = _shortcode(u)
        if sc:
            return ("ig", sc)
    if "tiktok.com" in u:
        vid = _tt_id(u)
        if vid:
            return ("tt", vid)
    nu = _normalize_url(u)
    return ("url", nu) if nu else None

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
    out_audio: List[Dict[str, Optional[str]]] = []
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
                        m_urls = []
                        if it.get("videoUrl"):
                            m_urls.append(it["videoUrl"])
                        if it.get("displayUrl"):
                            m_urls.append(it["displayUrl"])
                        if it.get("images"):
                            m_urls.extend(it["images"] if isinstance(it["images"], list) else [it["images"]])

                        m_urls = _ensure_list(m_urls)
                        local_paths: List[str] = []
                        for mu in m_urls:
                            try:
                                lp = _download_file(mu, pasta_destino=media)
                                if lp:
                                    local_paths.append(lp)
                            except Exception as e:
                                tlog(f"[IG] ⚠️ Falha ao baixar {mu}: {e}")
                        # extrai info de áudio de forma segura
                        _audio = it.get("musicInfo", {}) or {}
                        _raw_audio_id = _audio.get("audio_id")
                        audio_id = f"ig_{_raw_audio_id}" if _raw_audio_id is not None else None
                        artist_name = _audio.get("artist_name")
                        song_name = _audio.get("song_name")

                        audio_snapshot = None

                        if audio_id is not None or artist_name is not None or song_name is not None:
                            audio_snapshot = {
                                "author_name": artist_name,
                                "audio_name": song_name,
                                "plataforma": "Instagram",
                            }

                        caption = it.get("caption")
                        timestamp = it.get("timestamp")
                        hashtags, mentions = _extract_tags_and_mentions(caption)
                        
                        out.append({
                            "url":  _ig_post_url_from_item(it) or _any_post_url(it),
                            "ownerUsername": it.get("ownerUsername"),
                            "likesCount": it.get("likesCount"),
                            "commentsCount": it.get("commentsCount"),
                            "caption": caption,
                            "type": it.get("type"),
                            "post_timestamp": timestamp,
                            "videoPlayCount": it.get("videoPlayCount"),
                            "videoViewCount": it.get("videoViewCount"),
                            "hashtags": hashtags,
                            "mentions": mentions,
                            "audio_id": audio_id,
                            "audio_snapshot": audio_snapshot,
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

def _fetch_instagram_from_profiles(
    profile_urls: List[str],
    api_token: str,
    results_limit: int = 5,
    max_results: int = 1000,
) -> List[Dict[str, Optional[str]]]:
    """
    Recebe URLs de perfil (instagram.com/<user>/), roda 1 run do Apify com resultsType=posts
    e transforma os itens retornados no mesmo formato do _fetch_instagram.
    """

    def _ig_permalink_from_item(it: dict) -> Optional[str]:
        sc = it.get("shortCode") or _shortcode(it.get("url") or it.get("inputUrl") or "")
        if sc:
            return f"https://www.instagram.com/p/{sc}/"
        return it.get("url") or it.get("inputUrl")

    apify_client, _, _ = get_clients()
    profiles = list(dict.fromkeys(_normalize_ig_profile_url(u) for u in profile_urls if _is_ig_profile_url(u)))
    if not profiles:
        tlog("[IG][PROFILES] Nenhum perfil válido.")
        return []

    tlog(f"[IG][PROFILES] ▶️ Run único ({len(profiles)} perfis) | limit={results_limit}")
    run = apify_client.actor(INSTAGRAM_ACTOR).start(
        run_input={
            "addParentData": False,
            "directUrls": profiles,
            "enhanceUserSearchWithFacebookPage": False,
            "isUserReelFeedURL": False,
            "isUserTaggedFeedURL": False,
            "resultsLimit": int(results_limit),
            "resultsType": "posts",
        }
    )
    run_id = run.get("id")
    if not run_id:
        tlog("[IG][PROFILES] ❌ Sem run_id.")
        return []

    ds = None
    ds_id = None
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
                tlog(f"[IG][PROFILES] 🟡 status={status}, dataset={ds_id}")

        if ds_id:
            page = ds.list_items(limit=200, offset=offset, clean=True)
            items = _page_items(page)
            if items:
                tlog(f"[IG][PROFILES] 📥 +{len(items)} (offset {offset}→{offset+len(items)})")
                offset += len(items)
                for it in items:
                    try:
                        # mídia (coleta múltiplas possíveis fontes)
                        def _ensure_list(x):
                            if not x: return []
                            if isinstance(x, (list, tuple)): return list(x)
                            return [x]

                        m_urls: List[str] = []
                        if it.get("videoUrl"):
                            m_urls.append(it["videoUrl"])
                        if it.get("displayUrl"):
                            m_urls.append(it["displayUrl"])
                        if it.get("images"):
                            m_urls.extend(_ensure_list(it["images"]))

                        # deduplicação preservando ordem
                        seen_mu = set()
                        m_urls = [u for u in m_urls if not (u in seen_mu or seen_mu.add(u))]

                        local_paths: List[str] = []
                        for mu in m_urls:
                            try:
                                lp = _download_file(mu, pasta_destino=media)
                                if lp:
                                    local_paths.append(lp)
                            except Exception as e:
                                tlog(f"[IG][PROFILES] ⚠️ Falha download {mu}: {e}")

                        # áudio (se disponível)
                        _audio = it.get("musicInfo", {}) or {}
                        _raw_audio_id = _audio.get("audio_id")
                        audio_id = f"ig_{_raw_audio_id}" if _raw_audio_id is not None else None
                        artist_name = _audio.get("artist_name")
                        song_name = _audio.get("song_name")
                        audio_snapshot = None
                        if audio_id is not None or artist_name is not None or song_name is not None:
                            audio_snapshot = {
                                "author_name": artist_name,
                                "audio_name": song_name,
                                "plataforma": "Instagram",
                            }

                        caption = it.get("caption") or ""
                        hashtags, mentions = _extract_tags_and_mentions(caption)

                        permalink = _ig_permalink_from_item(it)

                        owner = it.get("ownerUsername") or it.get("username")
                        source_profile = (
                            _normalize_ig_profile_url(f"https://www.instagram.com/{owner}/") if owner else None
                        )
                        timestamp = it.get("timestamp")
                        row = {
                            "url": permalink,
                            "ownerUsername": owner,
                            "likesCount": it.get("likesCount"),
                            "commentsCount": it.get("commentsCount"),
                            "caption": caption,
                            "hashtags": hashtags,
                            "mentions": mentions,
                            "type": it.get("type") or ("Video" if it.get("isVideo") else "Image"),
                            "post_timestamp": timestamp,
                            "videoPlayCount": it.get("videoPlayCount") or it.get("videoViewCount"),
                            "videoViewCount": it.get("videoViewCount") or it.get("videoPlayCount"),
                            "audio_id": audio_id,
                            "audio_snapshot": audio_snapshot,
                            "mediaUrl": m_urls if len(m_urls) > 1 else (m_urls[0] if m_urls else None),
                            "mediaLocalPaths": local_paths if len(local_paths) > 1 else None,
                            "mediaLocalPath": local_paths[0] if local_paths else None,
                            "_source_profile": source_profile,
                        }
                        out.append(row)

                        if len(out) >= max_results:
                            tlog(f"[IG][PROFILES] ⛔ max_results atingido.")
                            return out[:max_results]
                    except Exception as e:
                        tlog(f"[IG][PROFILES] ❌ Erro processando item: {e}")
                no_new_ticks = 0
            else:
                no_new_ticks += 1

        if status in {"RUNNING", "READY"}:
            tlog(f"[IG][PROFILES] ⏳ status={status}, recebidos={len(out)}")
        else:
            tlog(f"[IG][PROFILES] 🧭 status final={status}, total={len(out)}")
            if no_new_ticks >= 2:
                break
        time.sleep(POLL_SEC)

    tlog(f"[IG][PROFILES] 🏁 Concluído com {len(out)} resultado(s)")
    return out

# 4) (opcional) De-dup no final de fetch_social_post_summary_async, antes do return:
def _dedup_results(rows: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for r in rows:
        key = (r.get("url"), r.get("ownerUsername"), r.get("timestamp"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

TT_PROFILE_RESERVED = {"tag", "music", "hashtag", "trending", "discover", "live"}

def _is_tt_profile_url(u: Optional[str]) -> bool:
    if not u or "tiktok.com" not in u:
        return False
    try:
        p = urlparse(u)
        segs = [s for s in (p.path or "").split("/") if s]
        # /@username[/]
        return len(segs) == 1 and segs[0].startswith("@") and segs[0].strip("@")
    except Exception:
        return False

def _normalize_tt_profile_url(u: str) -> str:
    if not re.match(r"^https?://", u, flags=re.I):
        u = "https://" + u
    p = urlparse(u.strip())
    path = (p.path or "/").rstrip("/") + "/"
    return urlunparse(("https", p.netloc.lower(), path, "", "", ""))

def _tt_username_from_url(u: str) -> Optional[str]:
    m = re.search(r"tiktok\.com/@([^/?#]+)", u, re.I)
    return m.group(1) if m else None

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

            # --- musicMeta seguro + variáveis corretas
            mm = it.get("musicMeta") or {}
            _raw_music_id = mm.get("musicId")
            audio_id: Optional[str] = None
            author_name: Optional[str] = None
            audio_name: Optional[str] = None
            plataforma = "Tiktok"

            if _raw_music_id is not None:
                audio_id = f"tt_{_raw_music_id}"
            author_name = mm.get("musicAuthor")
            audio_name = mm.get("musicName")

            audio_snapshot = None
            if audio_id is not None or author_name is not None or audio_name is not None:
                audio_snapshot = {
                    "author_name": author_name,
                    "audio_name": audio_name,
                    "plataforma": plataforma
                }

            caption = (it.get("text") or it.get("description") or "")
            timestamp = (it.get("createTimeISO") or it.get("createTime") or "")
            hashtags, mentions = _extract_tags_and_mentions(caption)
            out.append({
                "url": it.get("inputUrl") or _any_post_url(it),
                "ownerUsername": ((it.get("authorMeta") or {}).get("name")) or it.get("authorUniqueId"),
                "likesCount": it.get("diggCount"),
                "commentsCount": it.get("commentCount"),
                "caption": caption,
                "type": _type,
                "post_timestamp": timestamp,
                "videoPlayCount": it.get("playCount"),
                "videoViewCount": it.get("playCount"),
                "hashtags": hashtags,
                "mentions": mentions,
                "audio_id": audio_id,
                "audio_snapshot": audio_snapshot,
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

def _fetch_tiktok_from_profiles(
    profile_urls: List[str],
    api_token: str,
    results_limit: int = 5,
    max_results: int = 1000,
    post_urls_extra: Optional[List[str]] = None,
) -> List[Dict[str, Optional[str]]]:
    apify_client, _, _ = get_clients()

    # 1) Aceita SÓ links completos de perfil e extrai usernames, mapeando para a URL normalizada
    username_to_profile_url: Dict[str, str] = {}
    ignorados: List[str] = []
    for raw in profile_urls or []:
        if not isinstance(raw, str) or not raw.strip():
            ignorados.append(str(raw)); continue
        u = raw.strip()
        if "tiktok.com" not in u:
            ignorados.append(u); continue
        if not re.match(r"^https?://", u, flags=re.I):
            u = "https://" + u
        if not _is_tt_profile_url(u):
            ignorados.append(u); continue
        u_norm = _normalize_tt_profile_url(u)  # sempre com barra final
        un = _tt_username_from_url(u_norm)
        if un:
            username_to_profile_url.setdefault(un, u_norm)
        else:
            ignorados.append(u)

    usernames = list(username_to_profile_url.keys())
    if ignorados:
        tlog(f"[TT][PROFILES] Ignorados (não são links completos): {len(ignorados)}")
    if not usernames:
        tlog("[TT][PROFILES] ❌ Nenhum perfil válido.")
        return []

    run_input = {
        "excludePinnedPosts": False,
        "postURLs": list(dict.fromkeys((post_urls_extra or []))),
        "profiles": usernames,  # apenas usernames
        "proxyCountryCode": "None",
        "resultsPerPage": int(results_limit),
        "scrapeRelatedVideos": False,
        "shouldDownloadAvatars": False,
        "shouldDownloadCovers": False,
        "shouldDownloadMusicCovers": False,
        "shouldDownloadSlideshowImages": False,
        "shouldDownloadSubtitles": False,
        "shouldDownloadVideos": True,
    }

    tlog(f"[TT][PROFILES] ▶️ Run único ({len(usernames)} perfis) | resultsPerPage={results_limit}")
    run = apify_client.actor(TIKTOK_ACTOR).start(run_input=run_input)
    run_id = run.get("id")
    if not run_id:
        tlog("[TT][PROFILES] ❌ Sem run_id.")
        return []

    ds, ds_id, offset = None, None, 0
    out: List[Dict[str, Optional[str]]] = []
    cache_media_candidates: List[Tuple[int, List[str], bool]] = []  # (idx_out, candidates, is_slideshow)
    per_user_count: Dict[str, int] = {}
    no_new_ticks = 0

    while True:
        info = apify_client.run(run_id).get()
        status = info.get("status")
        if not ds_id:
            ds_id = info.get("defaultDatasetId")
            if ds_id:
                ds = apify_client.dataset(ds_id)
                tlog(f"[TT][PROFILES] 🟡 status={status}, dataset={ds_id}")

        if ds_id:
            page = ds.list_items(limit=200, offset=offset, clean=True)
            items = _page_items(page)
            if items:
                tlog(f"[TT][PROFILES] 📥 +{len(items)} (offset {offset}→{offset+len(items)})")
                offset += len(items)

                for it in items:
                    try:
                        # associa ao perfil de origem via 'input' (username enviado)
                        src_username = (it.get("input") or "").strip() or ((it.get("authorMeta") or {}).get("name") or "").strip()
                        src_profile_url = username_to_profile_url.get(src_username) or (_normalize_tt_profile_url(f"https://www.tiktok.com/@{src_username}/") if src_username else None)

                        # candidatos de mídia (prioriza downloadAddr)
                        def _ensure_list(x):
                            if not x: return []
                            if isinstance(x, (list, tuple)): return list(x)
                            return [x]
                        media_candidates: List[str] = []
                        dl_vm = (it.get("videoMeta") or {}).get("downloadAddr")
                        if isinstance(dl_vm, str) and dl_vm.strip():
                            media_candidates.append(dl_vm.strip())
                        media_candidates += [u for u in _ensure_list(it.get("mediaUrls")) if isinstance(u, str) and u.strip()]
                        seen_mu = set()
                        media_candidates = [u for u in media_candidates if not (u in seen_mu or seen_mu.add(u))]
                        is_slideshow = bool(it.get("isSlideshow"))

                        # áudio
                        mm = (it.get("musicMeta") or {})
                        audio_id = None
                        author_name = None
                        audio_name = None
                        if isinstance(mm, dict):
                            _raw_music_id = mm.get("musicId")
                            if _raw_music_id is not None:
                                audio_id = f"tt_{_raw_music_id}"
                            author_name = mm.get("musicAuthor")
                            audio_name = mm.get("musicName")
                        audio_snapshot = None
                        if audio_id is not None or author_name is not None or audio_name is not None:
                            audio_snapshot = {"author_name": author_name, "audio_name": audio_name, "plataforma": "Tiktok"}

                        # texto/hashtags
                        caption = (it.get("text") or it.get("description") or "") or ""
                        hashtags_cap, mentions_cap = _extract_tags_and_mentions(caption)
                        h_objs = it.get("hashtags") or []
                        if isinstance(h_objs, list):
                            h_names = [h.get("name") for h in h_objs if isinstance(h, dict) and h.get("name")]
                        else:
                            h_names = []
                        hashtags = _dedup_preservando_ordem(list(hashtags_cap) + h_names)
                        mentions = mentions_cap

                        owner = ((it.get("authorMeta") or {}).get("name")) or it.get("authorUniqueId")
                        permalink = it.get("webVideoUrl") or it.get("url") or it.get("inputUrl")
                        _type = "Slideshow" if is_slideshow or len(media_candidates) > 1 else "Video"

                        row = {
                            "url": permalink,
                            "ownerUsername": owner,
                            "likesCount": it.get("diggCount"),
                            "commentsCount": it.get("commentCount"),
                            "caption": caption,
                            "hashtags": hashtags,
                            "mentions": mentions,
                            "type": _type,
                            "post_timestamp": it.get("createTimeISO") or it.get("createTime"),
                            "videoPlayCount": it.get("playCount"),
                            "videoViewCount": it.get("playCount"),
                            "audio_id": audio_id,
                            "audio_snapshot": audio_snapshot,
                            "mediaUrl": media_candidates if len(media_candidates) > 1 else (media_candidates[0] if media_candidates else None),
                            "mediaLocalPaths": None,
                            "mediaLocalPath": None,
                            "_source_profile": src_profile_url,
                        }
                        out.append(row)
                        cache_media_candidates.append((len(out) - 1, media_candidates, is_slideshow))

                        if src_username:
                            per_user_count[src_username] = per_user_count.get(src_username, 0) + 1

                        if len(out) >= max_results:
                            tlog(f"[TT][PROFILES] ⛔ max_results atingido.")
                            break
                    except Exception as e:
                        tlog(f"[TT][PROFILES] ❌ Erro processando item: {e}")
                # fim for items
                if len(out) >= max_results:
                    break
                no_new_ticks = 0
            else:
                no_new_ticks += 1

        if status in {"RUNNING", "READY"}:
            tlog(f"[TT][PROFILES] ⏳ status={status}, recebidos={len(out)}")
        else:
            tlog(f"[TT][PROFILES] 🧭 status final={status}, total={len(out)}")
            if per_user_count:
                tlog(f"[TT][PROFILES] Resumo por username: {per_user_count}")
            break
        time.sleep(POLL_SEC)

    # ✅ COOLDOWN global após o run (deixa o KV estabilizar) + retry por URL
    if out:
        tlog(f"[TT][PROFILES] ⏱ cooldown {APIFY_KV_COOLDOWN_SEC}s antes dos downloads…")
        time.sleep(APIFY_KV_COOLDOWN_SEC)
        for idx_out, candidates, is_slideshow in cache_media_candidates:
            if not candidates:
                continue
            local_paths: List[str] = []
            first_ok: Optional[str] = None
            for mu in candidates:
                try:
                    lp = _download_file_with_retry(mu, pasta_destino=media)
                    if lp and os.path.isfile(lp):
                        local_paths.append(lp)
                        if not first_ok:
                            first_ok = lp
                        tlog(f"[TT][PROFILES][DL] ✅ ok → {os.path.basename(lp)}")
                        if not is_slideshow:
                            break
                except Exception as e:
                    tlog(f"[TT][PROFILES][DL] ⚠️ {mu.split('?')[0]} | {e}")
            # atualiza a linha
            if local_paths:
                out[idx_out]["mediaLocalPaths"] = local_paths if len(local_paths) > 1 else None
                out[idx_out]["mediaLocalPath"]  = first_ok

    tlog(f"[TT][PROFILES] 🏁 Concluído com {len(out)} resultado(s)")
    return out

# ─────────────────────────────────────────────────────────────
# Pipeline principal (busca → transcrição → limpeza)
# ─────────────────────────────────────────────────────────────
def _partition_known_unknown_by_mongo(all_urls: List[str]) -> Tuple[Dict[str, dict], Set[str]]:
    known_map = _load_docs_by_urls(all_urls, MONGO_FETCH_FIELDS)
    known_urls = set(known_map.keys())
    unknown_set = set(u for u in all_urls if u not in known_urls)
    return known_map, unknown_set

def _merge_results_in_input_order(
    input_urls: List[str],
    known_map: Dict[str, dict],
    fetched_new_items: List[Dict[str, Optional[str]]]
) -> List[Dict[str, Optional[str]]]:
    # Índices por identidade/URL e também por perfil (quando vier de perfis)
    fetched_by_id: Dict[Tuple[str, str], dict] = {}
    fetched_by_norm_url: Dict[str, dict] = {}
    by_profile: Dict[str, List[dict]] = {}

    def _cand_urls(it: dict) -> List[str]:
        cand = []
        for k in ("url", "inputUrl", "webVideoUrl", "shareUrl"):
            v = it.get(k)
            if isinstance(v, str) and v.strip():
                cand.append(v.strip())
        seen = set()
        return [u for u in cand if not (u in seen or seen.add(u))]

    # indexação dos itens novos
    for it in fetched_new_items or []:
        sp = it.get("_source_profile")
        if isinstance(sp, str) and sp:
            by_profile.setdefault(sp, []).append(it)

        for u in _cand_urls(it):
            pid = _post_identity(u)
            if pid and pid not in fetched_by_id:
                fetched_by_id[pid] = it
            nu = _normalize_url(u)
            if nu and nu not in fetched_by_norm_url:
                fetched_by_norm_url[nu] = it

    results: List[Dict[str, Optional[str]]] = []

    for u in input_urls:
        # 1) reaproveita do Mongo
        if u in known_map:
            doc = {k: known_map[u].get(k) if k in known_map[u] else None for k in MONGO_FETCH_FIELDS}
            doc["url"] = u
            doc["_from_mongo"] = True
            doc.setdefault("hashtags", [])
            doc.setdefault("mentions", [])
            results.append(doc)
            continue

        # 2) fan-out para perfis IG/TT (não criar placeholder quando não houver posts)
        if _is_ig_profile_url(u) or _is_tt_profile_url(u):
            prof_key = _normalize_ig_profile_url(u) if "instagram.com" in u else _normalize_tt_profile_url(u)
            posts = by_profile.get(prof_key, [])
            for it in posts:
                row = {k: it.get(k) if k in it else None for k in OUTPUT_FIELDS}
                if not row.get("url"):
                    row["url"] = it.get("inputUrl") or u
                row["_from_mongo"] = False
                row.setdefault("hashtags", [])
                row.setdefault("mentions", [])
                results.append(row)
            # ⚠️ Não adiciona placeholder para URLs de perfil (servem apenas como semente)
            continue

        # 3) caso normal (post único)
        pid = _post_identity(u)
        it = fetched_by_id.get(pid) if pid else None
        if not it:
            it = fetched_by_norm_url.get(_normalize_url(u))

        if it:
            row = {k: it.get(k) if k in it else None for k in OUTPUT_FIELDS}
            if not row.get("url"):
                row["url"] = it.get("inputUrl") or u
            row["_from_mongo"] = False
            row.setdefault("hashtags", [])
            row.setdefault("mentions", [])
            results.append(row)
        else:
            # para posts (não perfis), mantemos o placeholder
            placeholder = {"url": u, "_from_mongo": False}
            for k in OUTPUT_FIELDS:
                if k != "url":
                    placeholder[k] = None
            placeholder["hashtags"] = []
            placeholder["mentions"] = []
            results.append(placeholder)

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
    """
    Escolhe a melhor fonte de mídia para processamento.
    🔁 Prioriza ARQUIVOS LOCAIS (necessários para transcrição e extração de frames),
    depois cai para URLs remotas como fallback.
    """
    # 1) Prioriza caminhos locais (o worker depende disso)
    if isinstance(media_local_path, str) and media_local_path:
        return media_local_path
    if isinstance(media_local_paths, (list, tuple)):
        for p in media_local_paths:
            if isinstance(p, str) and p:
                return p

    # 2) Fallback: URLs remotas (não usadas pelo worker para abrir arquivo, mas útil p/ logging)
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
    return None


def _descrever_frames(frames_b64: List[str], max_imgs: int = 5, idioma: str = "pt-BR") -> Tuple[str, Optional[int], Optional[int], int, int]:
    """
    Retorna: (descricao_texto, input_tokens_text, output_tokens_text, num_images, estimated_image_tokens).
    - input/output_tokens_text: apenas TEXTO (usage da API).
    - num_images / estimated_image_tokens: estimativa baseada em tiles 512x512 por imagem.
    """
    if not frames_b64:
        return "", None, None, 0, 0
    try:
        imagens = frames_b64[:max_imgs]

        # calcula num_images e tokens estimados pelas imagens (por resolução)
        num_images, estimated_image_tokens = _estimate_image_tokens_from_b64_list(imagens)

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

        texto = (resp.choices[0].message.content or "").strip()

        # tokens de TEXTO (prompt/completion) — imagens não entram no usage
        usage = getattr(resp, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        output_tokens = getattr(usage, "completion_tokens", None) if usage else None

        return texto, input_tokens, output_tokens, num_images, estimated_image_tokens

    except Exception as e:
        return f"[ERRO descrição frames] {e}", None, None, 0, 0

# ─────────────────────────────────────────────────────────────
# Transcrição em paralelo (threads) + descrição de frames
# ─────────────────────────────────────────────────────────────
def _thread_worker(idx: int, media_url: Optional[str], sem: threading.Semaphore):
    """
    Transcreve vídeo/imagem e gera descrição de frames em paralelo.

    Retorna:
      (idx, transcricao:str|None, base64Frames:list[str], framesDescricao:str|None,
       ai_model_data:dict, erro:str|None)
    """

    ai_model_data = {
        "ai_model": "gpt-5-nano",
        "input_tokens": None,
        "output_tokens": None,
        "audio_seconds": None,
        "num_images": 0,
        "estimated_image_tokens": 0,
    }

    if not media_url:
        return idx, None, [], None, ai_model_data, "sem_media_url"

    try:
        base_dir = Path(media)

        def _is_url(s: str) -> bool:
            return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))

        def _resolve_local_path(s: str) -> Path:
            if _is_url(s):
                name = unquote(os.path.basename(urlsplit(s).path)) or "mediafile"
            else:
                name = s
            p = Path(name)
            return p if p.is_absolute() else (base_dir / name)

        local_path = _resolve_local_path(media_url)
        if not local_path.exists():
            return idx, None, [], None, ai_model_data, f"FileNotFoundError: {local_path}"

        ext = local_path.suffix.lower()
        is_video = ext in {".mp4", ".mov", ".m4v", ".webm", ".ts", ".mkv", ".avi"}
        is_image = ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

        texto: Optional[str] = None
        base64_frames: List[str] = []
        frames_descricao: Optional[str] = None

        # ==========================================================
        # 🔹 PROCESSAMENTO DE VÍDEO (transcrição + frames em paralelo)
        # ==========================================================
        if is_video:
            tlog(f"[TR] 🎙️ idx={idx}: processando vídeo {local_path.name}")

            # Rodamos transcrição (com semáforo) e extração de frames em paralelo
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_transcricao = ex.submit(
                    lambda: (lambda: (
                        sem.acquire(),
                        tlog(f"[TR] ▶️ transcrevendo {local_path.name}"),
                        transcrever_video_em_speed(str(local_path))
                    ))()[2]
                    if sem
                    else transcrever_video_em_speed(str(local_path))
                )
                fut_frames = ex.submit(get_video_frames, str(local_path))

                try:
                    texto_resp, dur_s = fut_transcricao.result()
                    texto = texto_resp
                    ai_model_data["audio_seconds"] = dur_s
                finally:
                    try:
                        sem.release()
                    except Exception:
                        pass

                try:
                    base64_frames = fut_frames.result()
                except Exception as e:
                    tlog(f"[frames] erro extraindo frames de {local_path}: {e}")
                    base64_frames = []

            # 🔸 Descrição dos frames
            if base64_frames:
                secs = ai_model_data.get("audio_seconds") or 0
                dyn_max_imgs = max(3, min(15, int(round(secs / 6)) or 1))
                desc, in_tok, out_tok, num_imgs, est_img_tokens = _descrever_frames(
                    base64_frames, max_imgs=dyn_max_imgs
                )
                frames_descricao = desc
                ai_model_data["input_tokens"] = in_tok
                ai_model_data["output_tokens"] = out_tok
                ai_model_data["num_images"] = num_imgs
                ai_model_data["estimated_image_tokens"] = est_img_tokens

            tlog(f"[TR] ✅ idx={idx}: concluído {local_path.name}")

        # ==========================================================
        # 🔹 PROCESSAMENTO DE IMAGEM ESTÁTICA
        # ==========================================================
        elif is_image:
            with open(local_path, "rb") as img:
                base64_frames = [base64.b64encode(img.read()).decode("utf-8")]

            desc, in_tok, out_tok, num_imgs, est_img_tokens = _descrever_frames(
                base64_frames, max_imgs=1
            )
            frames_descricao = desc
            ai_model_data["input_tokens"] = in_tok
            ai_model_data["output_tokens"] = out_tok
            ai_model_data["num_images"] = num_imgs
            ai_model_data["estimated_image_tokens"] = est_img_tokens

        # ==========================================================
        # 🔹 RETORNO PADRÃO
        # ==========================================================
        return idx, texto, base64_frames, frames_descricao, ai_model_data, None

    except Exception as e:
        return idx, None, [], None, ai_model_data, f"{type(e).__name__}: {e}"

    finally:
        # ==========================================================
        # ♻️ LIMPEZA DE ARQUIVOS TEMPORÁRIOS
        # ==========================================================
        try:
            if local_path.exists():
                local_path.unlink()
                tlog(f"[LIMPEZA] 🗑️ {local_path.name} removido com sucesso")
        except Exception as e:
            tlog(f"[WARN] falha ao remover {local_path}: {e}")

        try:
            tmpdir = Path(BASE_DIR) / "tmp"
            for f in tmpdir.glob("audio_*"):
                if time.time() - f.stat().st_mtime > 10:
                    f.unlink()
        except Exception:
            pass

def anexar_transcricoes_threaded(
    resultados: List[Dict[str, Any]],
    max_workers: int = max_workers,
    gpu_singleton: bool = False,
    callback: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Para cada item com mídia local, transcreve (vídeo) e descreve frames em paralelo.
    Mantém campos existentes vindos do Mongo; só preenche se houver conteúdo novo.

    Campos adicionados/atualizados por item:
      - transcricao
      - base64Frames
      - framesDescricao
      - ai_model_data
      - transcricao_erro

    Parâmetro opcional:
      - callback(i, total): chamado a cada item processado (para atualizar progresso).
    """
    if not resultados:
        return resultados

    sem = threading.Semaphore(1 if gpu_singleton else max_workers)

    jobs: List[Tuple[int, Optional[str]]] = []
    for idx, item in enumerate(resultados):
        veio_mongo = bool(item.get("_from_mongo"))

        # Garante que SEMPRE exista ai_model_data (mesmo que não processe nada)
        item.setdefault("ai_model_data", {
            "ai_model": "gpt-5-nano",
            "input_tokens": None,
            "output_tokens": None,
            "audio_seconds": None,
            "num_images": 0,
            "estimated_image_tokens": 0,
        })

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
        # nada para processar; já garantimos ai_model_data acima
        return resultados

    total = len(jobs)
    concluídos = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_thread_worker, idx, url, sem) for idx, url in jobs]
        for fut in as_completed(futs):
            (
                idx,
                transcricao,
                frames,
                frames_desc,
                ai_model_data,
                erro,
            ) = fut.result()

            if transcricao is not None:
                resultados[idx]["transcricao"] = transcricao
            if frames is not None:
                resultados[idx]["base64Frames"] = frames
            if frames_desc is not None:
                resultados[idx]["framesDescricao"] = frames_desc

            # ✅ único campo agregado para dados da IA
            resultados[idx]["ai_model_data"] = ai_model_data or {
                "ai_model": "gpt-5-nano",
                "input_tokens": None,
                "output_tokens": None,
                "audio_seconds": None,
            }

            if erro:
                resultados[idx]["transcricao_erro"] = erro
            else:
                resultados[idx].pop("transcricao_erro", None)

            # ✅ Atualiza progresso
            concluídos += 1
            if callback:
                try:
                    callback(concluídos, total)
                except Exception:
                    pass  # callback nunca deve travar o processamento

    # segurança: garante ai_model_data em todos (inclusive vindos do Mongo ou sem mídia)
    for item in resultados:
        if "ai_model_data" not in item or not isinstance(item["ai_model_data"], dict):
            item["ai_model_data"] = {
                "ai_model": "gpt-5-nano",
                "input_tokens": None,
                "output_tokens": None,
                "audio_seconds": None,
            }
        else:
            # completa chaves eventualmente faltantes
            item["ai_model_data"].setdefault("ai_model", "gpt-5-nano")
            item["ai_model_data"].setdefault("input_tokens", None)
            item["ai_model_data"].setdefault("output_tokens", None)
            item["ai_model_data"].setdefault("audio_seconds", None)
            item["ai_model_data"].setdefault("num_images", 0)
            item["ai_model_data"].setdefault("estimated_image_tokens", 0)

    return resultados

def gerar_embeddings(resultados: List[dict], model: str = "text-embedding-3-small", batch_size: int = 100) -> List[dict]:
    """
    Gera embeddings em BATCHES para cada item combinando caption + transcricao + framesDescricao.
    Retorna embeddings no formato:
      {"embedding": {"vectorized_embedding": [...], "embedding_provider": "openai", "embedding_model": model}}

    batch_size: número máximo de textos por chamada à API (100 recomendado)
    """

    if not resultados:
        return resultados

    _, client, _ = get_clients()

    def limpar_texto(txt: str) -> str:
        """Remove pontuação e múltiplos espaços."""
        txt = re.sub(r"[^A-Za-zÀ-ÿ0-9\s]", " ", str(txt))
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

    # 🔹 Monta a lista de textos a vetorializar
    textos = []
    idx_map = []  # mapeia índice original → posição no batch
    for i, item in enumerate(resultados):
        caption = item.get("caption") or ""
        transcricao = item.get("transcricao") or ""
        frames = item.get("framesDescricao") or ""

        texto = f"caption: {caption} transcricao: {transcricao} frames: {frames}"
        texto = limpar_texto(texto)
        if texto:
            textos.append(texto)
            idx_map.append(i)
        else:
            item["embedding"] = None  # sem texto para vetorializar

    if not textos:
        return resultados

    # 🔹 Envia em batches (melhor desempenho e custo)
    for start in range(0, len(textos), batch_size):
        end = min(start + batch_size, len(textos))
        batch = textos[start:end]
        idxs = idx_map[start:end]

        try:
            resp = client.embeddings.create(model=model, input=batch)
            embeddings = [d.embedding for d in resp.data]

            for i, emb in zip(idxs, embeddings):
                resultados[i]["embedding"] = {
                    "vectorized_embedding": emb,
                    "embedding_provider": "openai",
                    "embedding_model": model,
                }

        except Exception as e:
            # Em caso de erro no batch, marca todos os itens afetados
            for i in idxs:
                resultados[i]["embedding"] = {
                    "vectorized_embedding": None,
                    "embedding_provider": "openai",
                    "embedding_model": model,
                    "error": str(e),
                }
            tlog(f"[EMBEDDINGS] ⚠️ Erro no batch {start}-{end}: {e}")

    tlog(f"[EMBEDDINGS] ✅ Concluído: {len(textos)} textos vetorizados em batches de {batch_size}.")
    return resultados

# Nome do índice vetorial no Atlas
VECTOR_INDEX_NAME = "comm_ref_vector_index"

# Função para buscar os vizinhos no Mongo
def _buscar_vizinhos_mongo(embedding_vector, k=5):
    """Busca os k vizinhos mais próximos no MongoDB Atlas Vector Search."""
    if embedding_vector is None:
        return []

    # Converte para lista se vier como np.ndarray
    if isinstance(embedding_vector, np.ndarray):
        embedding_vector = embedding_vector.tolist()
    elif not isinstance(embedding_vector, list):
        try:
            embedding_vector = list(embedding_vector)
        except Exception:
            print("[CLUSTER] Embedding inválido — não iterável.")
            return []

    if len(embedding_vector) == 0:
        return []

    db = _connect_to_mongo()
    collection_ref = db["comunidades-ref"]

    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "embedding",  # campo vetorial na coleção de referência
                "queryVector": embedding_vector,
                "numCandidates": 100,
                "limit": k,
            }
        },
        {
            "$project": {
                "_id": 1,
                "score": {"$meta": "vectorSearchScore"},
                "comunidade": 1,
                "categoria_principal": 1,
                "subcategoria": 1,
            }
        },
    ]

    try:
        resultados = list(collection_ref.aggregate(pipeline))
        return resultados
    except Exception as e:
        print(f"[CLUSTER] Erro ao buscar vizinhos no MongoDB: {e}")
        return []

# Função para inferir categoria
def _inferir_categoria_knn(vizinhos):
    """Determina comunidade/categoria/subcategoria predominantes ponderando pelos scores."""
    if not vizinhos:
        return None

    comunidades = [v.get("comunidade") for v in vizinhos if v.get("comunidade")]
    categorias = [v.get("categoria_principal") for v in vizinhos if v.get("categoria_principal")]
    subcategorias = [v.get("subcategoria") for v in vizinhos if v.get("subcategoria")]
    scores = np.array([v.get("score", 0) for v in vizinhos], dtype=float)

    def ponderar(valores, distancias, epsilon: float = 1e-9):
        """
        Calcula pesos normalizados proporcionalmente à proximidade (1 / distância).
        Evita divisões por zero e problemas de tipo com arrays.
        """
        if (
            valores is None
            or len(valores) == 0
            or distancias is None
            or len(distancias) == 0
            or len(valores) != len(distancias)
        ):
            return None, {}

        dist = np.array(distancias, dtype=float)

        # substitui distâncias inválidas (0, NaN, inf)
        dist[~np.isfinite(dist)] = 1.0

        # evita divisão por zero
        inv_dist = 1.0 / (dist + epsilon)

        # normaliza pesos para somarem 1
        pesos_norm = inv_dist / inv_dist.sum()

        pesos_dict = {}
        for val, peso in zip(valores, pesos_norm):
            if val:
                pesos_dict[val] = pesos_dict.get(val, 0.0) + float(peso)

        if not pesos_dict:
            return None, {}

        proporcoes = {k: round(v * 100, 1) for k, v in pesos_dict.items()}
        valor_predito = max(proporcoes, key=proporcoes.get)
        return valor_predito, proporcoes

    comunidade_predita, comunidades_prop = ponderar(comunidades, scores)
    categoria_predita, _ = ponderar(categorias, scores)
    subcategoria_predita, _ = ponderar(subcategorias, scores)

    return {
        "comunidade_predita": comunidade_predita,
        "categoria_principal": categoria_predita,
        "subcategoria": subcategoria_predita,
        "comunidades_proporcoes": comunidades_prop,
    }

def _hash_embedding(embedding):
    """Cria uma chave hash única para logging/debug (não usada em cache)."""
    try:
        return hashlib.md5(np.array(embedding, dtype=float).tobytes()).hexdigest()
    except Exception:
        return "invalid_hash"

def _buscar_vizinhos_mongo_wrapper(embedding, k=5):
    """
    Wrapper simples para validar e converter o embedding antes da busca.
    (Sem cache, totalmente seguro para arrays numpy.)
    """
    if embedding is None:
        return []

    if isinstance(embedding, np.ndarray):
        embedding = embedding.tolist()
    elif not isinstance(embedding, list):
        try:
            embedding = list(embedding)
        except Exception:
            print("[CLUSTER] Embedding inválido recebido no wrapper.")
            return []

    if len(embedding) == 0:
        return []

    try:
        return _buscar_vizinhos_mongo(embedding, k=k)
    except Exception as e:
        print(f"[CLUSTER] Erro em _buscar_vizinhos_mongo_wrapper: {e}")
        return []

def classificar_via_mongo_vector_search(resultados, k=5, max_workers=12):
    """
    Classifica cada item via MongoDB Atlas Vector Search (sem cache),
    garantindo segurança de tipos e compatibilidade com numpy arrays.
    """

    def processar_item(item):
        emb = None
        try:
            emb = (item.get("embedding") or {}).get("vectorized_embedding")
        except Exception:
            pass

        # Converte para lista se necessário
        if emb is None:
            return item
        if isinstance(emb, np.ndarray):
            emb = emb.tolist()
        elif not isinstance(emb, list):
            try:
                emb = list(emb)
            except Exception:
                print(f"[WARN] embedding inválido no item: {item.get('url')}")
                return item

        if len(emb) == 0:
            return item

        try:
            vizinhos = _buscar_vizinhos_mongo_wrapper(emb, k=k)
            resultado = _inferir_categoria_knn(vizinhos)
            if resultado:
                item.update(resultado)
        except Exception as e:
            print(f"[CLUSTER] Erro ao processar item {item.get('url')}: {e}")

        return item

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futuros = [executor.submit(processar_item, item) for item in resultados]
        resultados_final = [f.result() for f in as_completed(futuros)]

    return resultados_final

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

    urls_instagram_profiles = [u for u in input_urls if "instagram.com" in u and _is_ig_profile_url(u)]
    urls_instagram_posts    = [u for u in input_urls if "instagram.com" in u and _shortcode(u)]
    
    urls_tiktok_profiles    = [u for u in input_urls if "tiktok.com" in u and _is_tt_profile_url(u)]
    urls_tiktok_posts       = [u for u in input_urls if "tiktok.com" in u and _tt_id(u)]
    
    urls_static             = [u for u in input_urls if "https://static-resources" in u]
    
    # filtrar o que já está no Mongo
    urls_instagram_posts = [u for u in urls_instagram_posts if u in unknown_set]
    urls_tiktok_posts    = [u for u in urls_tiktok_posts if u in unknown_set]
    urls_tiktok_profiles = [u for u in urls_tiktok_profiles if u in unknown_set]
    urls_static          = [u for u in urls_static if u in unknown_set]

    tlog(
    f"[FETCH] IG posts novos: {len(urls_instagram_posts)} | "
    f"IG perfis: {len(urls_instagram_profiles)} | "
    f"TT posts novos: {len(urls_tiktok_posts)} | "
    f"TT perfis: {len(urls_tiktok_profiles)} | "
    f"Static novos: {len(urls_static)} | Já no Mongo: {len(known_map)}"
    )

    tasks = []
    if urls_instagram_posts:
        tlog("[FETCH] Agendando fetch IG (posts)…")
        tasks.append(asyncio.to_thread(_fetch_instagram, urls_instagram_posts, api_token, max_results))
    if urls_instagram_profiles:
        tlog("[FETCH] Agendando fetch IG (perfis→posts)…")
        tasks.append(asyncio.to_thread(_fetch_instagram_from_profiles, urls_instagram_profiles, api_token, 5, max_results))
    if urls_tiktok_posts:
        tlog("[FETCH] Agendando fetch TikTok (posts)…")
        tasks.append(asyncio.to_thread(_fetch_tiktok, urls_tiktok_posts, api_token, max_results))
    if urls_tiktok_profiles:
        tlog("[FETCH] Agendando fetch TikTok (perfis→posts)…")
        tasks.append(asyncio.to_thread(_fetch_tiktok_from_profiles, urls_tiktok_profiles, api_token, 5, max_results))
    if urls_static:
        tlog("[FETCH] Agendando fetch Static…")
        tasks.append(asyncio.to_thread(_fetch_static_resources, urls_static, max_results))

    fetched_new_items: List[Dict[str, Optional[str]]] = []
    if tasks:
        parts = await asyncio.gather(*tasks, return_exceptions=True)
        for p in parts:
            if isinstance(p, Exception):
                tlog(f"[FETCH] ⚠️ Tarefa paralela falhou: {p}")
                continue
            fetched_new_items.extend(p or [])


    results = _merge_results_in_input_order(input_urls, known_map, fetched_new_items)

    # remove placeholders vazios de perfis
    profile_set = set(_normalize_ig_profile_url(u) for u in urls_instagram_profiles) | \
                  set(_normalize_tt_profile_url(u) for u in urls_tiktok_profiles)

    def _is_placeholder_for_profile(row: dict) -> bool:
        u = row.get("url") or ""
        try:
            if "instagram.com" in u:
                nu = _normalize_ig_profile_url(u)
            elif "tiktok.com" in u:
                nu = _normalize_tt_profile_url(u)
            else:
                nu = u
        except Exception:
            nu = u
        if nu not in profile_set:
            return False
        return all(row.get(k) is None for k in OUTPUT_FIELDS if k != "url")

    results = [r for r in results if not _is_placeholder_for_profile(r)]

    # ✅ garante inclusão de posts que não foram mapeados — mas IGNORA perfis
    seen_urls = set(r.get("url") for r in results if r.get("url"))
    for it in fetched_new_items:
        u = it.get("url")
        if not u or u in seen_urls:
            continue
        # pula URLs de perfil para não criar "registro do perfil"
        try:
            if ("tiktok.com" in u and _is_tt_profile_url(u)) or ("instagram.com" in u and _is_ig_profile_url(u)):
                continue
        except Exception:
            pass

        row = {k: it.get(k) if k in it else None for k in OUTPUT_FIELDS}
        row["url"] = u
        row["_from_mongo"] = False
        row.setdefault("hashtags", [])
        row.setdefault("mentions", [])
        results.append(row)
        seen_urls.add(u)

    # Deduplicação final (url + ownerUsername + timestamp)
    unique: Dict[Tuple[str, Optional[str], Optional[str]], dict] = {}
    for r in results:
        key = (
            str(r.get("url") or "").rstrip("/"),
            r.get("ownerUsername"),
            str(r.get("timestamp") or r.get("post_timestamp") or ""),
        )
        if key not in unique:
            unique[key] = r
    results = list(unique.values())

    if len(results) > max_results:
        results = results[:max_results]

    tlog(f"[FETCH] Finalizado: {len(results)} item(ns) total.")
    return results

async def rodar_pipeline(urls: List[str], progress_callback=None) -> List[dict]:
    """
    Executa o pipeline completo:
    1) Busca posts
    2) Transcreve e extrai frames
    3) Gera embeddings
    4) Classifica via Mongo Vector Search
    5) Limpa mídia temporária
    """

    if not urls:
        print("Nenhuma URL fornecida.")
        return []

    total_steps = 5
    step = 0

    def update_step(msg):
        nonlocal step
        step += 1
        if progress_callback:
            progress_callback(step / total_steps, msg)

    # ----------------------------
    # 1️⃣ Buscar/baixar posts
    # ----------------------------
    update_step("🔍 Buscando e baixando posts...")
    resultados = await fetch_social_post_summary_async(urls, api_token=None, max_results=1000)
    if not resultados:
        print("Nenhum resultado retornado pelos scrapers.")
        return []

    # ----------------------------
    # 2️⃣ Transcrever e extrair frames
    # ----------------------------
    update_step("🎙️ Transcrevendo e extraindo frames...")

    total_videos = len(resultados)
    progresso_local = 0

    def local_progress():
        nonlocal progresso_local
        progresso_local += 1
        if progress_callback:
            # Progresso da etapa 2 representando de 20% a 60% do total
            progresso_total = (1 + (progresso_local / total_videos) * 2) / total_steps
            progress_callback(progresso_total, f"🎧 Transcrevendo vídeos ({progresso_local}/{total_videos})...")

    anexar_transcricoes_threaded(resultados, max_workers=max_workers, gpu_singleton=True, callback=local_progress)

    # ----------------------------
    # 3️⃣ Gerar embeddings
    # ----------------------------
    update_step("🧠 Gerando embeddings...")
    resultados = gerar_embeddings(resultados)

    # ----------------------------
    # 4️⃣ Classificar via Mongo Vector Search
    # ----------------------------
    update_step("🏷️ Classificando resultados...")
    resultados = classificar_via_mongo_vector_search(resultados, k=5)

    # ----------------------------
    # 5️⃣ Limpar arquivos temporários
    # ----------------------------
    update_step("🧹 Limpando diretórios temporários...")
    caminhos = _coletar_caminhos_midia(resultados)
    _deletar_arquivos(caminhos)
    _deletar_pasta_se_vazia(Path(media))

    tmpdir = Path(BASE_DIR) / "tmp"
    if tmpdir.exists():
        for f in tmpdir.iterdir():
            try:
                f.unlink()
            except Exception:
                pass
        _deletar_pasta_se_vazia(tmpdir)

    update_step("✅ Finalizado com sucesso!")
    return resultados

