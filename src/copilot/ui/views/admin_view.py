"""Admin page: document management, audit lookup and the evaluation dashboard."""
from __future__ import annotations

import streamlit as st

from copilot.config import Settings
from copilot.ui.api_client import ApiError, CopilotClient
from copilot.ui.formatting import documents_table, format_ms, version_status_text
from copilot.ui.views.evaluation_view import render_evaluation

_ACCESS = ["internal", "public", "restricted"]
# Version status choices for the upload form. "Use the document header" sends nothing, so a `Status:` line in the file applies.
_VERSION_STATUS = {"Use the document header": None, "Current": "current", "Superseded (replaced by a newer document)": "superseded"}


def _documents_tab(client: CopilotClient, settings: Settings) -> None:
    exts = ", ".join(e.lstrip(".").upper() for e in settings.allowed_extensions)
    st.subheader("Upload a document")
    st.caption(f"Supported formats: {exts} · max {settings.max_upload_mb} MB. Demo knowledge-base files are synthetic, not real company policy.")
    with st.form("upload_form", clear_on_submit=True):
        file = st.file_uploader("File", type=[e.lstrip(".") for e in settings.allowed_extensions], key="upload_file")
        c1, c2, c3 = st.columns(3)
        category = c1.text_input("Category", placeholder="e.g. vpn", max_chars=64)
        department = c2.text_input("Department", placeholder="IT", max_chars=64)
        access = c3.selectbox("Access level", _ACCESS)
        c4, c5 = st.columns(2)
        version_status = c4.selectbox("Version status", list(_VERSION_STATUS), help="Superseded documents stay searchable, but answers label them as outdated.")
        superseded_by = c5.text_input("Replaced by", placeholder="e.g. Password_Policy.txt", max_chars=120, help="Used only when the version status is Superseded.")
        submitted = st.form_submit_button("Upload and ingest")
    if submitted:
        if file is None:
            st.warning("Choose a file first.")
        else:
            try:
                res = client.upload_document(file.name, file.getvalue(), category=category.strip() or None, department=department.strip() or None, access_level=access,
                                         lifecycle=_VERSION_STATUS[version_status], superseded_by=superseded_by.strip() or None)
                {"indexed": st.success, "skipped": st.info}.get(res["status"], st.error)(f"{res['document_name']}: {res['message']}")
            except ApiError as exc:
                st.error(exc.detail)

    st.subheader("Documents")
    if st.button("Refresh", key="refresh_docs"):
        st.rerun()
    try:
        docs = client.list_documents()["documents"]
    except ApiError as exc:
        st.error(exc.detail)
        return
    if not docs:
        st.info("No documents yet. Upload one above, or run `python scripts/ingest_kb.py`.")
        return
    st.dataframe(documents_table(docs), width="stretch", hide_index=True)

    st.subheader("Manage a document")
    chosen = st.selectbox("Document", docs, format_func=lambda d: f"{d['document_name']} (v{d['version']}, {d['access_level']}, {version_status_text(d).lower()})", key="manage_doc")
    c1, c2 = st.columns(2)
    if c1.button("Re-index", key="reindex"):
        try:
            res = client.reindex_document(chosen["document_id"])
            (st.success if res["status"] != "failed" else st.error)(res["message"])
        except ApiError as exc:
            st.error(exc.detail)
    confirm = c2.checkbox("I understand this permanently deletes the document and its vectors", key="confirm_delete")
    if c2.button("Delete", key="delete", disabled=not confirm):
        try:
            client.delete_document(chosen["document_id"])
            st.success(f"Deleted {chosen['document_name']}.")
            st.rerun()
        except ApiError as exc:
            st.error(exc.detail)


def _audit_tab(client: CopilotClient) -> None:
    st.subheader("Audit lookup")
    rid = st.text_input("Request ID", key="audit_rid", max_chars=64)
    if st.button("Look up", key="audit_go") and rid.strip():
        try:
            rec = client.audit(rid.strip())
        except ApiError as exc:
            st.error(exc.detail)
            return
        c = st.columns(4)
        c[0].metric("Intent", rec.get("intent") or "-")
        c[1].metric("Abstained", "yes" if rec["abstained"] else "no")
        c[2].metric("External search", "yes" if rec["external_search_triggered"] else "no")
        c[3].metric("Time", format_ms(rec.get("execution_time_ms")))
        st.json(rec, expanded=False)


def render_admin(client: CopilotClient, settings: Settings) -> None:
    """Draw the admin page."""
    st.header("Administration")
    docs_tab, audit_tab, eval_tab = st.tabs(["Documents", "Audit", "Evaluation"])
    with docs_tab:
        _documents_tab(client, settings)
    with audit_tab:
        _audit_tab(client)
    with eval_tab:
        render_evaluation(client, settings)
