# main.py
import asyncio
import io
import json
import re
from datetime import datetime
from typing import Iterable, List, Tuple, Union
from urllib.parse import urlparse

import pandas as pd

from funcs2 import rodar_pipeline, _upload_para_mongo, upload_muitos_para_mongo  # suas funções

VALID_DOMAINS = {
    "instagram.com", "www.instagram.com", "m.instagram.com",
    "tiktok.com", "www.tiktok.com", "vm.tiktok.com", "vt.tiktok.com",
    "static-resources"  # mantido do seu filtro original
}

def _is_supported_url(u: str) -> bool:
    try:
        if not u:
            return False
        # garante esquema p/ urlparse funcionar bem:
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
    # Iterable de URLs
    urls = [_normalize_url(x) for x in obj if x and str(x).strip()]
    urls = [u for u in urls if _is_supported_url(u)]
    # dedup preservando ordem
    unique, seen = [], set()
    for u in urls:
        if u not in seen:
            unique.append(u)
            seen.add(u)
    return unique

def _run_async_pipeline(urls: List[str]):
    # Ajuste do loop no Windows
    try:
        import platform
        if platform.system() == "Windows":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore
    except Exception:
        pass
    return asyncio.run(rodar_pipeline(urls))

def _sanitize_items(resultados):
    campos_excluir = {"mediaLocalPaths", "mediaLocalPath", "base64Frames"}
    itens_sanitizados = []
    itens_para_upload = []

    for r in resultados:
        item = {k: v for k, v in r.items() if k not in campos_excluir}
        itens_sanitizados.append(item)

        if "transcricao_erro" not in r:
            item_para_upload = {
                k: r.get(k)
                for k in ["url", "ownerUsername", "caption", "type",
                          "likesCount", "commentsCount", "videoPlayCount",
                          "transcricao", "framesDescricao"]
            }
            item_para_upload["post_timestamp"] = r.get("timestamp")
            item_para_upload["upload_timestamp"] = datetime.now()
            itens_para_upload.append(item_para_upload)

        # log bonito (stdout do streamlit mostrará)
        try:
            print(json.dumps(item, ensure_ascii=False, indent=2))
        except Exception:
            print(str(item))

    return itens_sanitizados, itens_para_upload

def _to_excel_bytes(df: pd.DataFrame) -> Tuple[bytes, str]:
    agora = datetime.now()
    data_hora_formatada = agora.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"posts_analisados_{data_hora_formatada}.xlsx"

    buf = io.BytesIO()
    # openpyxl é o engine padrão do pandas para .xlsx, funciona bem em memória
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    buf.seek(0)
    return buf.read(), filename

def main(urls_or_text: Union[str, Iterable[str]]):
    """
    Aceita:
      - string com 'uma URL por linha' (ex.: conteúdo da textarea)
      - lista/iterável de URLs

    Retorna:
      - dict com chaves:
          'df'            -> pandas.DataFrame com resultados sanitizados
          'excel_bytes'   -> bytes do arquivo .xlsx
          'excel_name'    -> nome sugerido do arquivo
          'n_urls'        -> int
          'n_items_upload'-> int
    Levanta ValueError se não houver URLs válidas.
    """
    urls = _ensure_urls(urls_or_text)
    if not urls:
        raise ValueError("Nenhuma URL válida foi fornecida (Instagram/TikTok).")

    print(f"type(urls)={type(urls).__name__}, n_urls={len(urls)}")
    print("Exemplos:", urls[:3])

    resultados = _run_async_pipeline(urls)
    itens_sanitizados, itens_para_upload = _sanitize_items(resultados)

    # upload para Mongo, como no código original
    try:
        if len(itens_para_upload) > 1:
            upload_muitos_para_mongo(itens_para_upload)
        elif len(itens_para_upload) == 1:
            _upload_para_mongo(itens_para_upload)
    except Exception as e:
        print(f"[Aviso] Falha no upload para Mongo: {e}")

    for r in itens_sanitizados:
        r.pop("audio_id", None)
        r.pop("audio_snapshot", None)

    df = pd.DataFrame(itens_sanitizados)
    excel_bytes, excel_name = _to_excel_bytes(df)

    return {
        "df": df,
        "excel_bytes": excel_bytes,
        "excel_name": excel_name,
        "n_urls": len(urls),
        "n_items_upload": len(itens_para_upload),
    }

# Mantém compatibilidade com execução direta (opcional)
if __name__ == "__main__":
    print("Este módulo agora espera ser chamado via main(urls_or_text).")

