import asyncio
import io
import json
import re
from datetime import datetime
from typing import Iterable, List, Tuple, Union
from urllib.parse import urlparse
import os

import pandas as pd
import numpy as np

from funcs2 import rodar_pipeline, _upload_para_mongo, upload_muitos_para_mongo  # suas funções

# ----------------------------
# Configurações de domínios válidos
# ----------------------------
VALID_DOMAINS = {
    "instagram.com", "www.instagram.com", "m.instagram.com",
    "tiktok.com", "www.tiktok.com", "vm.tiktok.com", "vt.tiktok.com",
    "static-resources"  # mantido do seu filtro original
}

# ----------------------------
# Debug (MONGO_URI)
# ----------------------------

if not os.getenv("MONGO_URI"):
    raise RuntimeError("Variável MONGO_URI não encontrada no ambiente! Verifique as variáveis no Railway.")

# ----------------------------
# Funções auxiliares de URL
# ----------------------------
def _is_supported_url(u: str) -> bool:
    try:
        if not u:
            return False
        if not u.startswith(("http://", "https://")):
            u = "https://" + u
        p = urlparse(u.strip())
        if p.scheme not in {"http", "https"}:
            return False
        host = (p.netloc or "").lower().split(":")[0]
        return any(d in host for d in VALID_DOMAINS)
    except Exception:
        return False


def _normalize_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return ""
    return u if re.match(r"^https?://", u, flags=re.I) else f"https://{u}"


def _parse_urls_from_text(raw_text: str) -> List[str]:
    lines = [line.strip() for line in (raw_text or "").splitlines()]
    seen, urls = set(), []
    for line in lines:
        n = _normalize_url(line)
        if n and _is_supported_url(n) and n not in seen:
            urls.append(n)
            seen.add(n)
    return urls


def _ensure_urls(obj: Union[str, Iterable[str]]) -> List[str]:
    if isinstance(obj, str):
        return _parse_urls_from_text(obj)
    urls = [_normalize_url(x) for x in obj if x and str(x).strip()]
    urls = [u for u in urls if _is_supported_url(u)]
    unique, seen = [], set()
    for u in urls:
        if u not in seen:
            unique.append(u)
            seen.add(u)
    return unique


# ----------------------------
# Execução assíncrona
# ----------------------------
def _run_async_pipeline(urls: List[str]):
    try:
        import platform
        if platform.system() == "Windows":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore
    except Exception:
        pass
    return asyncio.run(rodar_pipeline(urls))


# ----------------------------
# Sanitização dos resultados (apenas para upload)
# ----------------------------
def _sanitize_items(resultados):
    campos_excluir = {"mediaLocalPaths", "mediaLocalPath", "base64Frames", "mediaUrl"}
    itens_sanitizados = []
    itens_para_upload = []

    for r in resultados:
        item = {k: v for k, v in r.items() if k not in campos_excluir}
        itens_sanitizados.append(item)

        if "transcricao_erro" not in r:
            item_para_upload = {
                k: r.get(k)
                for k in [
                    "url",
                    "ownerUsername",
                    "caption",
                    "type",
                    "likesCount",
                    "commentsCount",
                    "videoPlayCount",
                    "videoViewCount",
                    "transcricao",
                    "framesDescricao",
                    "audio_id",
                    "audio_snapshot",
                    "hashtags",
                    "mentions",
                    "ai_model_data",
                    "embedding",
                    "categoria_principal",
                    "subcategoria",
                    "comunidade_predita",
                    "comunidades_proporcoes",
                ]
            }

            item_para_upload["post_timestamp"] = r.get("timestamp") or r.get("post_timestamp")
            item_para_upload["upload_timestamp"] = datetime.now()
            itens_para_upload.append(item_para_upload)

        # Impressão JSON (debug)
        try:
            print(json.dumps(item, ensure_ascii=False, indent=2))
        except Exception:
            print(str(item))

    return itens_sanitizados, itens_para_upload
    
# ----------------------------
# Exportação para XLSX
# ----------------------------
def _to_xlsx_bytes(df: pd.DataFrame) -> Tuple[bytes, str]:
    """Converte DataFrame em bytes XLSX."""
    from io import BytesIO
    agora = datetime.now()
    data_hora_formatada = agora.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"posts_analisados_{data_hora_formatada}.xlsx"

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="posts_analisados")
    return buf.getvalue(), filename


# ----------------------------
# Função principal
# ----------------------------
def main(urls_or_text: Union[str, Iterable[str]]):
    """
    Aceita:
      - string com 'uma URL por linha'
      - lista/iterável de URLs

    Retorna:
      - dict com chaves:
          'df'            -> pandas.DataFrame com resultados
          'xlsx_bytes'    -> bytes do arquivo .xlsx
          'xlsx_name'     -> nome sugerido do arquivo
          'n_urls'        -> int
          'n_items_upload'-> int
    """
    urls = _ensure_urls(urls_or_text)
    if not urls:
        raise ValueError("Nenhuma URL válida foi fornecida (Instagram/TikTok).")

    print(f"type(urls)={type(urls).__name__}, n_urls={len(urls)}")
    print("Exemplos:", urls[:3])

    try:
        resultados = _run_async_pipeline(urls)
    except Exception as e:
        print(f"[ERRO] Falha ao executar pipeline: {type(e).__name__}: {e}")
        return {
            "df": pd.DataFrame(),
            "xlsx_bytes": b"",
            "xlsx_name": f"erro_{datetime.now():%Y-%m-%d_%H-%M-%S}.xlsx",
            "n_urls": len(urls),
            "n_items_upload": 0,
        }

    if not resultados:
        print("[Aviso] Pipeline não retornou resultados válidos.")
        return {
            "df": pd.DataFrame(),
            "xlsx_bytes": b"",
            "xlsx_name": f"sem_resultados_{datetime.now():%Y-%m-%d_%H-%M-%S}.xlsx",
            "n_urls": len(urls),
            "n_items_upload": 0,
        }

    itens_sanitizados, itens_para_upload = _sanitize_items(resultados)

    try:
        if len(itens_para_upload) > 1:
            upload_muitos_para_mongo(itens_para_upload)
        elif len(itens_para_upload) == 1:
            _upload_para_mongo(itens_para_upload[0])
    except Exception as e:
        print(f"[Aviso] Falha no upload para Mongo: {e}")

    # Remove campos grandes e não necessários no Excel
    for r in itens_para_upload:
        r.pop("audio_id", None)
        r.pop("audio_snapshot", None)

            # 🔧 Remove o vetor de embedding (muito grande para Excel)
        if isinstance(r.get("embedding"), dict):
            r["embedding"].pop("vectorized_embedding", None)

    # DataFrame baseado em itens_para_upload (mesmos dados enviados ao Mongo)
    df = pd.DataFrame(itens_para_upload)
    if not df.empty:
        df = df.drop(columns=["_from_mongo"], errors="ignore")

    # Exporta para XLSX
    xlsx_bytes, xlsx_name = _to_xlsx_bytes(df)

    return {
        "df": df,
        "xlsx_bytes": xlsx_bytes,
        "xlsx_name": xlsx_name,
        "n_urls": len(urls),
        "n_items_upload": len(itens_para_upload),
    }

# ----------------------------
# Execução direta (opcional)
# ----------------------------
if __name__ == "__main__":
    print("Este módulo agora gera um arquivo .xlsx e envia os dados para o MongoDB.")



