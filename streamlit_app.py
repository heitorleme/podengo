# streamlit_app.py
import io
from contextlib import redirect_stdout, redirect_stderr
from importlib import import_module
from urllib.parse import urlparse
from main import main
import streamlit as st

VALID_DOMAINS = {
    "instagram.com", "www.instagram.com", "m.instagram.com",
    "tiktok.com", "www.tiktok.com", "vm.tiktok.com", "vt.tiktok.com", "static-resources"
}

def is_supported_url(u: str) -> bool:
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

def normalize_url(u: str) -> str:
    if not u:
        return ""
    return u if u.startswith(("http://", "https://")) else "https://" + u.strip()

def parse_urls(raw_text: str):
    lines = [normalize_url(line.strip()) for line in (raw_text or "").splitlines()]
    urls, seen = [], set()
    for line in lines:
        if line and is_supported_url(line) and line not in seen:
            urls.append(line)
            seen.add(line)
    return urls

st.set_page_config(page_title="Fetcher IG/TikTok", page_icon="🔗", layout="centered")
st.title("🔗 Processar publicações do Instagram e TikTok")
st.caption("Cole **uma URL por linha** ou envie um arquivo `.txt`. Geraremos um Excel para download.")

with st.expander("Como usar", expanded=False):
    st.markdown(
        "- Cole **uma URL por linha**.\n"
        "- Suportados: **instagram.com**, **tiktok.com** (inclui `vm.tiktok.com`).\n"
        "- Ao finalizar, um **Excel** é disponibilizado para download."
    )

col1, col2 = st.columns(2, vertical_alignment="top")

with col1:
    raw = st.text_area(
        "Cole aqui (uma URL por linha):",
        height=180,
        placeholder="https://www.instagram.com/p/...\nhttps://www.tiktok.com/@user/video/...\nhttps://vm.tiktok.com/...",
    )

with col2:
    uploaded = st.file_uploader("Ou envie um .txt com URLs", type=["txt"])
    if uploaded is not None:
        try:
            txt = uploaded.read().decode("utf-8", errors="ignore")
            raw = (raw + "\n" + txt) if raw else txt
        except Exception as e:
            st.error(f"Erro ao ler o arquivo: {e}")

urls = parse_urls(raw or "")

# Feedback
if raw:
    st.write("### URLs detectadas")
    if urls:
        st.success(f"{len(urls)} URL(s) válida(s).")
        for u in urls:
            st.write(f"• {u}")
    else:
        st.warning("Nenhuma URL válida detectada.")

st.divider()

# Execução
if st.button("▶️ Executar pipeline e gerar Excel", type="primary", disabled=not urls):
    out_buf, err_buf = io.StringIO(), io.StringIO()
    try:
        mod = import_module("main")
        if not hasattr(mod, "main"):
            st.error("`main.py` não contém uma função `main`.")
        else:
            fn = getattr(mod, "main")
            with st.spinner("Executando..."):
                with redirect_stdout(out_buf), redirect_stderr(err_buf):
                    # passa a lista diretamente
                    result = fn(urls)

            stdout_txt = out_buf.getvalue().strip()
            stderr_txt = err_buf.getvalue().strip()

            # Logs
            with st.expander("📄 Saída (stdout)", expanded=bool(stdout_txt)):
                st.code(stdout_txt or "(sem saída)")
            with st.expander("⚠️ Erros/alertas (stderr)", expanded=bool(stderr_txt)):
                st.code(stderr_txt or "(sem erros)")

            # Download Excel
            if result and "excel_bytes" in result and "excel_name" in result:
                st.success("✅ Execução concluída. Baixe o Excel abaixo.")
                st.download_button(
                    label="⬇️ Baixar Excel",
                    data=result["excel_bytes"],
                    file_name=result["excel_name"],
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    use_container_width=True,
                )

            # Mostra um resumo opcional
            st.caption(
                f"Processadas {result.get('n_urls', 0)} URL(s). "
                f"Itens enviados ao Mongo: {result.get('n_items_upload', 0)}."
            )

    except ModuleNotFoundError:
        st.error("Arquivo `main.py` não encontrado na pasta atual.")
    except Exception as e:
        st.error(f"Falha ao executar: {e}")
        # ainda mostramos os buffers se houver algo
        stdout_txt = out_buf.getvalue().strip()
        stderr_txt = err_buf.getvalue().strip()
        if stdout_txt:
            with st.expander("📄 Saída (stdout)"):
                st.code(stdout_txt)
        if stderr_txt:
            with st.expander("⚠️ Erros/alertas (stderr)"):
                st.code(stderr_txt)

st.caption("Dica: `streamlit run streamlit_app.py`")