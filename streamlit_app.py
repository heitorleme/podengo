import io 
import time 
from contextlib import redirect_stdout, redirect_stderr 
from importlib import import_module 
from urllib.parse import urlparse 
from datetime import datetime 
import streamlit as st 

from main import main  # sua função principal

# ----------------------------
# Configurações
# ----------------------------
VALID_DOMAINS = {
    "instagram.com", "www.instagram.com", "m.instagram.com",
    "tiktok.com", "www.tiktok.com", "vm.tiktok.com",
    "vt.tiktok.com", "static-resources",
}

# ----------------------------
# Helpers de URL
# ----------------------------
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


# ----------------------------
# Session state seguro
# ----------------------------
defaults = {
    "running": False,
    "result": None,
    "stdout": "",
    "stderr": "",
    "start_time": None,
    "progress_ratio": 0.0,
    "progress_message": "---",
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ----------------------------
# Layout principal
# ----------------------------
st.set_page_config(page_title="Fetcher IG/TikTok", page_icon="🔗", layout="centered")
st.title("🔗 Processar publicações do Instagram e TikTok")

st.caption(
    "Cole **uma URL por linha** ou envie um arquivo .txt. "
    "Geraremos um arquivo Excel (.xlsx) para download."
)

with st.expander("Como usar", expanded=False):
    st.markdown("""
    - Cole **uma URL por linha**.  
    - Suportados: **instagram.com**, **tiktok.com**.  
    - Ao finalizar, um **arquivo Excel** estará disponível.
    """)

# Inputs
col1, col2 = st.columns(2, vertical_alignment="top")
with col1:
    raw = st.text_area(
        "Cole aqui:",
        height=180,
        placeholder="https://www.instagram.com/p/...\nhttps://www.tiktok.com/@user/video/..."
    )

with col2:
    uploaded = st.file_uploader("Ou envie um arquivo .txt", type=["txt"])
    if uploaded:
        txt = uploaded.read().decode("utf-8", errors="ignore")
        raw = (raw + "\n" + txt) if raw else txt

urls = parse_urls(raw or "")

if raw:
    st.write("### URLs detectadas")
    if urls:
        st.success(f"{len(urls)} URL(s) válidas.")
        with st.expander("Ver URLs", expanded=False):
            st.text("\n".join(urls))
    else:
        st.warning("Nenhuma URL válida detectada.")

st.divider()

# ----------------------------
# Botão para iniciar pipeline
# ----------------------------
run_analysis = st.checkbox("Executar Análise GPT (custo extra)", value=True, key="run_analysis")

if st.button("▶️ Executar pipeline e gerar Excel", disabled=not urls, type="primary"):
    st.session_state.running = True
    st.session_state.result = None
    st.session_state.stdout = ""
    st.session_state.stderr = ""
    st.session_state.progress_ratio = 0.0
    st.session_state.progress_message = "---"
    st.session_state.start_time = time.time()
    st.rerun()


# ============================================================
# EXECUÇÃO DO PIPELINE (SEGURO)
# ============================================================
if st.session_state.running:

    st.info("⏳ Pipeline iniciado. Não feche ou recarregue a página.")

    # Barra de progresso
    progress_bar = st.progress(0)
    progress_text = st.empty()
    eta_text = st.empty()

    # LOGS em tempo real
    live_logs = st.container()
    live_logs.markdown("### 🪵 Logs em tempo real")
    live_log_box = live_logs.empty()

    out_buf = io.StringIO()
    err_buf = io.StringIO()

    # --------------------------------------
    # CALLBACK: Atualiza barra + ETA
    # --------------------------------------
    def update_progress(ratio, message):
        st.session_state.progress_ratio = max(0.0, min(1.0, float(ratio)))
        st.session_state.progress_message = message

        percent = int(st.session_state.progress_ratio * 100)
        progress_bar.progress(percent)
        progress_text.write(f"**{percent}% — {message}**")

        elapsed = time.time() - st.session_state.start_time
        if ratio > 0:
            total_est = elapsed / ratio
            remaining = total_est - elapsed
            if remaining > 0:
                eta_text.write(
                    f"⏱️ ETA: **{int(remaining//60)} min {int(remaining%60)} s**"
                )

        # 🔥 Atualizar logs ao vivo
        live_log_box.code(out_buf.getvalue() + err_buf.getvalue(), language="bash")

    # --------------------------------------
    # Execução principal
    # --------------------------------------
    # --------------------------------------
    # Execução principal
    # --------------------------------------
    try:
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            result = main(urls, run_analysis=st.session_state.get("run_analysis", True), progress_callback=update_progress)

        st.session_state.result = result
        st.session_state.stdout = out_buf.getvalue()
        st.session_state.stderr = err_buf.getvalue()
        st.session_state.running = False
        st.rerun()

    except Exception as e:
        st.session_state.stderr = f"[FATAL] {e}\n\n{err_buf.getvalue()}"
        st.session_state.stdout = out_buf.getvalue()
        st.session_state.running = False
        st.rerun()


# ============================================================
# EXIBIÇÃO DOS RESULTADOS
# ============================================================
if st.session_state.result:

    result = st.session_state.result
    st.success("🎉 Pipeline concluído!")

    # DOWNLOAD EXCEL
    if "xlsx_bytes" in result:
        st.download_button(
            label="⬇️ Baixar Excel (.xlsx)",
            data=result["xlsx_bytes"],
            file_name=result["xlsx_name"],
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True,
        )

    # JSON result
    st.subheader("📄 Dados estruturados")
    with st.expander("Ver JSON de exemplo", expanded=False):
        if "df" in result:
            records = result["df"].to_dict(orient="records")
            st.json(records[0] if records else {})
        else:
            st.json(result)

    st.caption(
        f"Processadas {result.get('n_urls', 0)} URL(s). "
        f"Itens enviados ao Mongo: {result.get('n_items_upload', 0)}."
    )
