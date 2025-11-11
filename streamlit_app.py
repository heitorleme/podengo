import io
from contextlib import redirect_stdout, redirect_stderr
from importlib import import_module
from urllib.parse import urlparse
import streamlit as st
from main import main  # importa sua função principal

# ----------------------------
# Configurações de domínios válidos
# ----------------------------
VALID_DOMAINS = {
    "instagram.com", "www.instagram.com", "m.instagram.com",
    "tiktok.com", "www.tiktok.com", "vm.tiktok.com", "vt.tiktok.com", "static-resources"
}

# ----------------------------
# Funções auxiliares de URL
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
# Interface Streamlit
# ----------------------------
st.set_page_config(page_title="Fetcher IG/TikTok", page_icon="🔗", layout="centered")
st.title("🔗 Processar publicações do Instagram e TikTok")
st.caption("Cole **uma URL por linha** ou envie um arquivo `.txt`. Geraremos um arquivo Excel (.xlsx) para download.")

with st.expander("Como usar", expanded=False):
    st.markdown(
        "- Cole **uma URL por linha**.\n"
        "- Suportados: **instagram.com**, **tiktok.com** (inclui `vm.tiktok.com`).\n"
        "- Ao finalizar, um **arquivo Excel (.xlsx)** é disponibilizado para download."
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

# ----------------------------
# Feedback de URLs
# ----------------------------
if raw:
    st.write("### URLs detectadas")
    if urls:
        st.success(f"{len(urls)} URL(s) válida(s).")
        with st.expander("Ver URLs válidas", expanded=False):
            st.text("\n".join(urls))
    else:
        st.warning("Nenhuma URL válida detectada.")

st.divider()

# ----------------------------
# Execução do pipeline
# ----------------------------
if st.button("▶️ Executar pipeline e gerar Excel (.xlsx)", type="primary", disabled=not urls):
    out_buf, err_buf = io.StringIO(), io.StringIO()
    try:
        mod = import_module("main")
        if not hasattr(mod, "main"):
            st.error("`main.py` não contém uma função `main`.")
        else:
            fn = getattr(mod, "main")
            with st.spinner("Executando..."):
                with redirect_stdout(out_buf), redirect_stderr(err_buf):
                    result = fn(urls)

            stdout_txt = out_buf.getvalue().strip()
            stderr_txt = err_buf.getvalue().strip()

            # ----------------------------
            # Saída estruturada em JSON
            # ----------------------------
            st.subheader("📦 Resultados estruturados")

            structured_data = []
            if result and isinstance(result, dict):
                # ✅ Se o main retornou um DataFrame, usa ele para mostrar os resultados
                if "df" in result and hasattr(result["df"], "to_dict"):
                    try:
                        structured_data = result["df"].to_dict(orient="records")
                    except Exception as e:
                        st.warning(f"Não foi possível converter o DataFrame: {e}")
                        structured_data = []
                else:
                    # tenta pegar possíveis saídas por URL
                    for u in urls:
                        structured_data.append({
                            "url": u,
                            "resultado": result.get("outputs", {}).get(u, "sem saída"),
                            "embedding": result.get("embeddings", {}).get(u, []),
                        })
            else:
                # fallback: tenta parsear stdout como JSON
                import json
                try:
                    parsed = json.loads(stdout_txt)
                    if isinstance(parsed, list):
                        structured_data = parsed
                    elif isinstance(parsed, dict):
                        structured_data = [parsed]
                except Exception:
                    structured_data = [{"raw_output": stdout_txt or "(sem saída)"}]

            with st.expander("📄 Visualizar estrutura JSON", expanded=False):
                st.json(structured_data)

            # ----------------------------
            # Erros / alertas
            # ----------------------------
            if stderr_txt:
                with st.expander("⚠️ Erros/alertas (stderr)", expanded=False):
                    st.code(stderr_txt)

            # ----------------------------
            # Download XLSX
            # ----------------------------
            if result and "xlsx_bytes" in result and "xlsx_name" in result:
                st.success("✅ Execução concluída. Baixe o arquivo Excel abaixo.")
                st.download_button(
                    label="⬇️ Baixar Excel (.xlsx)",
                    data=result["xlsx_bytes"],
                    file_name=result["xlsx_name"],
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    use_container_width=True,
                )

            # Resumo
            st.caption(
                f"Processadas {result.get('n_urls', 0)} URL(s). "
                f"Itens enviados ao Mongo: {result.get('n_items_upload', 0)}."
            )

    except ModuleNotFoundError:
        st.error("Arquivo `main.py` não encontrado na pasta atual.")
    except Exception as e:
        st.error(f"Falha ao executar: {e}")
        stdout_txt = out_buf.getvalue().strip()
        stderr_txt = err_buf.getvalue().strip()
        if stdout_txt:
            with st.expander("📄 Saída (stdout)"):
                st.code(stdout_txt)
        if stderr_txt:
            with st.expander("⚠️ Erros/alertas (stderr)"):
                st.code(stderr_txt)
