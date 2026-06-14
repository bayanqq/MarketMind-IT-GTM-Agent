"""
app.py  —  MarketMind AI  (Saudi Arabia GTM Intelligence Platform)
------------------------------------------------------------------
Pipeline (Sequential):
  Agent 1: Research Agent         — Saudi Arabia market & competitor analysis
  Agent 2: Strategy Agent         — ICP, budget-aware GTM, 30-day calendar
  Agent 3: Content Agent          — posts, email, ads, video script
  Agent 4: Brand Alignment Agent  — FINAL quality gate, refines all content

UI layout:
  • Results shown at the TOP when available
  • Controls / inputs in sidebar or below results
  • Background image (uploaded) with white overlay
  • Per-agent PDF download buttons
  • Marketing calendar with print/download button

Run:  streamlit run app.py
"""

from __future__ import annotations

import base64
import os
import sys
import tempfile
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

from agents.brand_alignment_agent import BrandAlignmentAgent, RefinedContent
from agents.content_agent import ContentAgent, ContentPackage
from agents.research_agent import CompanyProfile, ResearchAgent
from agents.strategy_agent import StrategyAgent
from utils.pdf_export import build_pdf
from utils.pdf_handler import load_and_validate_pdf
from utils.security import validate_input_text

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MarketMind AI — Saudi Arabia GTM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Background image encoding
# ---------------------------------------------------------------------------
def _bg_css() -> str:
    bg_path = _PROJECT_ROOT / "assets" / "background.jpg"
    whitening_level = "0.55"
    if bg_path.exists():
        with open(bg_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return (
            f"background-image: url('data:image/jpeg;base64,{b64}');"
            f"background-color: rgba(255, 255, 255, {whitening_level});"
            "background-blend-mode: overlay;"
            "background-size: cover; "
            "background-attachment: fixed; "
            "background-position: center;"
        )
    return "background: linear-gradient(135deg,#0f172a 0%,#1e3a5f 50%,#0f172a 100%);"

BG_CSS = _bg_css()

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
:root {{
    --mm-primary:#1B4FE4; --mm-accent:#00C2A8;
    --mm-dark:#0D1B2A;    --mm-surface:#ffffffee;
    --mm-border:#D6E0F5;  --mm-success:#16A34A;
    --mm-text:#0D1B2A;
}}
.stApp {{
    {BG_CSS}
}}
.stApp::before {{
    content:'';
    position:fixed; inset:0;
    background:rgba(255,255,255,0.82);
    z-index:0; pointer-events:none;
}}
.block-container {{
    position:relative; z-index:1;
    padding-top:1.5rem !important;
    max-width:1200px;
}}
[data-testid="stSidebar"] {{
    background:var(--mm-dark) !important;
    z-index:10;
}}
[data-testid="stSidebar"] * {{ color:#E2E8F0 !important; }}
[data-testid="stSidebar"] .stMarkdown h2 {{
    color:#00C2A8 !important; font-size:.82rem;
    letter-spacing:.12em; text-transform:uppercase;
}}
[data-testid="stSidebar"] .stTextInput input {{
    background:#1e2d3d !important; color:#e2e8f0 !important;
    border:1px solid #334155 !important;
}}
[data-testid="stSidebar"] .stSlider {{filter:brightness(1.4);}}
.stage-pill {{
    display:inline-block; padding:3px 12px; border-radius:20px;
    font-size:.72rem; font-weight:700; letter-spacing:.06em;
    text-transform:uppercase; margin-bottom:5px;
}}
.pill-pending {{ background:#E2E8F0; color:#64748B; }}
.pill-active  {{ background:#EFF6FF; color:var(--mm-primary); border:1px solid var(--mm-primary); }}
.pill-done    {{ background:#DCFCE7; color:var(--mm-success); }}
.result-banner {{
    background:linear-gradient(90deg,#1B4FE4,#00C2A8);
    color:#fff; padding:18px 24px; border-radius:12px;
    margin-bottom:20px; font-size:1.15rem; font-weight:700;
}}
.post-card {{
    border-left:4px solid var(--mm-primary); background:var(--mm-surface);
    border-radius:0 10px 10px 0; padding:16px 20px; margin-bottom:14px;
    box-shadow:0 1px 6px rgba(27,79,228,.10);
}}
.ad-card {{
    border-left:4px solid var(--mm-accent); background:var(--mm-surface);
    border-radius:0 10px 10px 0; padding:16px 20px; margin-bottom:14px;
    box-shadow:0 1px 6px rgba(0,194,168,.10);
}}
.video-card {{
    border-left:4px solid #8B5CF6; background:var(--mm-surface);
    border-radius:0 10px 10px 0; padding:16px 20px; margin-bottom:14px;
    box-shadow:0 1px 6px rgba(139,92,246,.10);
}}
.scene-block {{
    background:#F8F4FF; border-radius:8px;
    padding:12px 16px; margin-bottom:10px;
    border:1px solid #DDD6FE;
}}
.platform-tag {{
    font-size:.68rem; font-weight:700; text-transform:uppercase;
    letter-spacing:.1em; color:var(--mm-primary); margin-bottom:6px;
}}
.email-preview {{
    background:var(--mm-surface); border:1px solid var(--mm-border);
    border-radius:10px; padding:24px 28px;
    font-family:Georgia,serif; line-height:1.75;
}}
.email-subject {{ font-size:1.1rem; font-weight:700; color:var(--mm-dark); margin-bottom:4px; }}
.email-preview-text {{ font-size:.8rem; color:#94A3B8; margin-bottom:16px; font-style:italic; }}
.email-cta {{
    display:inline-block; margin-top:14px; padding:10px 24px;
    background:var(--mm-primary); color:#fff !important;
    border-radius:6px; font-weight:700; font-size:.88rem;
}}
.guardrail-item {{
    background:#FEF3C7; border-left:3px solid #F59E0B;
    padding:5px 12px; margin-bottom:3px; border-radius:0 4px 4px 0; font-size:.8rem;
}}
.calendar-table {{ width:100%; border-collapse:collapse; margin-top:8px; }}
.calendar-table th {{
    background:var(--mm-primary); color:#fff;
    padding:8px 12px; font-size:.8rem; text-align:left;
}}
.calendar-table td {{
    padding:7px 12px; font-size:.8rem;
    border-bottom:1px solid var(--mm-border);
    background:var(--mm-surface);
}}
.calendar-table tr:nth-child(even) td {{ background:#F0F4FF; }}
.mm-divider {{ border:none; border-top:1px solid var(--mm-border); margin:22px 0; }}
#MainMenu,footer {{ visibility:hidden; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------
DEFAULTS: dict = {
    "stage": 1,
    "substage": "form",
    "profile":          None,
    "research_report":  None,
    "research_doc_id":  None,
    "gtm_strategy":     None,
    "calendar_md":      None,
    "strategy_doc_id":  None,
    "content_package":  None,
    "content_doc_id":   None,
    "refined_content":  None,
    "brand_doc_id":     None,
    "form_company_name":   "",
    "form_products":       "",
    "form_brand_voice":    "",
    "form_past_marketing": "",
    "form_pdf_text":       "",
    "form_budget":         "SAR 20,000 / month",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# API clients
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_clients(openai_key: str, tavily_key: str):
    return OpenAI(api_key=openai_key), TavilyClient(api_key=tavily_key)

def resolve_clients():
    ok = os.environ.get("OPENAI_API_KEY", "") or st.session_state.get("openai_key", "")
    tk = os.environ.get("TAVILY_API_KEY", "") or st.session_state.get("tavily_key", "")
    if not ok or not tk:
        st.error("⚠️ API keys missing — enter them in the sidebar and click **Save Keys**.")
        st.stop()
    return get_clients(ok, tk)

# ---------------------------------------------------------------------------
# PDF download helper
# ---------------------------------------------------------------------------
def pdf_download_button(label: str, agent_label: str, content: str,
                         filename: str, key: str) -> None:
    """Render a Streamlit download button that serves a PDF."""
    profile_name = ""
    if st.session_state.profile:
        profile_name = st.session_state.profile.company_name
    try:
        pdf_bytes = build_pdf(agent_label, profile_name, content)
        st.download_button(
            label=label,
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf",
            key=key,
        )
    except Exception as exc:
        st.warning(f"PDF export note: {exc} — try copying the text instead.")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            "<h1 style='color:#00C2A8;font-size:1.35rem;font-weight:800;"
            "letter-spacing:.04em;margin-bottom:0'>🧠 MarketMind AI</h1>"
            "<p style='font-size:.75rem;color:#94A3B8;margin-top:2px'>"
            "Saudi Arabia GTM Intelligence Platform</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("## Pipeline")
        current = st.session_state.stage
        for num, icon, title, subtitle in [
            (1, "🔍", "Research Agent",       "Saudi Arabia Market Analysis"),
            (2, "🗺️", "Strategy Agent",       "ICP · Budget · Calendar"),
            (3, "✍️", "Content Agent",         "Posts · Email · Ads · Video"),
            (4, "🎨", "Brand Alignment",       "Final Quality Gate"),
        ]:
            done    = current > num
            active  = current == num
            pill_cls = "pill-done" if done else ("pill-active" if active else "pill-pending")
            pill_txt = "✓ Done" if done else ("▶ Active" if active else "Pending")
            st.markdown(
                f'<span class="stage-pill {pill_cls}">{pill_txt}</span><br>'
                f'<span style="font-weight:700;font-size:.88rem">{icon} {title}</span><br>'
                f'<span style="font-size:.74rem;color:#94A3B8">{subtitle}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)

        st.divider()

        st.markdown("## API Keys")
        st.caption("Keys are read from environment variables or entered below (password-masked).")

        load_dotenv()
        env_ok = os.environ.get("OPENAI_API_KEY", "")
        env_tk = os.environ.get("TAVILY_API_KEY", "")

        ok_input = st.text_input(
            "OpenAI API Key",
            value=st.session_state.get("openai_key", ""),
            type="password",
            placeholder="sk-… (or set OPENAI_API_KEY in .env)",
            help="Leave blank if OPENAI_API_KEY is already in your .env file.",
        )
        tv_input = st.text_input(
            "Tavily API Key",
            value=st.session_state.get("tavily_key", ""),
            type="password",
            placeholder="tvly-… (or set TAVILY_API_KEY in .env)",
            help="Leave blank if TAVILY_API_KEY is already in your .env file.",
        )
        if st.button("💾 Save Keys", use_container_width=True):
            if ok_input and tv_input:
                st.session_state.openai_key = ok_input
                st.session_state.tavily_key = tv_input
                st.success("Keys saved for this session.")
            else:
                st.error("Enter both keys.")

        has_keys = bool((env_ok or st.session_state.get("openai_key")) and
                        (env_tk or st.session_state.get("tavily_key")))
        colour  = "#16A34A" if has_keys else "#DC2626"
        status  = "● Keys loaded" if has_keys else "● No keys detected"
        st.markdown(f"<span style='color:{colour};font-size:.78rem'>{status}</span>",
                    unsafe_allow_html=True)

        st.divider()
        if st.button("🔄 Reset Pipeline", use_container_width=True):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def hr():
    st.markdown('<hr class="mm-divider">', unsafe_allow_html=True)

def section_header(icon: str, title: str, subtitle: str = "") -> None:
    st.markdown(
        f"<div style='margin-bottom:4px'>"
        f"<span style='font-size:1.5rem'>{icon}</span> "
        f"<span style='font-size:1.25rem;font-weight:800;color:#0D1B2A'>{title}</span>"
        f"</div>"
        + (f"<p style='color:#64748B;margin:0 0 12px 0;font-size:.88rem'>{subtitle}</p>" if subtitle else ""),
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Reusable output sections
# ---------------------------------------------------------------------------
def show_research_output() -> None:
    if not st.session_state.research_report:
        return
    with st.expander("📊 Research Agent Output", expanded=True):
        st.markdown(st.session_state.research_report)
        hr()
        pdf_download_button(
            "📥 Download Research Report (PDF)",
            "Research Agent Output",
            st.session_state.research_report,
            "research_report.pdf",
            "dl_research",
        )

def show_strategy_output() -> None:
    if not st.session_state.gtm_strategy:
        return
    with st.expander("🗺️ Strategy Agent Output", expanded=True):
        st.markdown(st.session_state.gtm_strategy)

        if st.session_state.calendar_md:
            st.markdown("### 📅 30-Day Marketing Calendar")
            st.markdown(st.session_state.calendar_md)
            hr()
            cal_content = "# 30-Day Marketing Calendar\n\n" + st.session_state.calendar_md
            pdf_download_button(
                "🖨️ Download / Print Calendar (PDF)",
                "30-Day Marketing Calendar",
                cal_content,
                "marketing_calendar.pdf",
                "dl_calendar",
            )
        hr()
        full_strategy = st.session_state.gtm_strategy
        if st.session_state.calendar_md:
            full_strategy += "\n\n## 30-Day Marketing Calendar\n" + st.session_state.calendar_md
        pdf_download_button(
            "📥 Download GTM Strategy (PDF)",
            "Strategy Agent Output",
            full_strategy,
            "gtm_strategy.pdf",
            "dl_strategy",
        )

def show_content_output() -> None:
    pkg: ContentPackage | None = st.session_state.content_package
    if not pkg:
        return

    refined: RefinedContent | None = st.session_state.refined_content
    posts = refined.refined_posts if refined else pkg.linkedin_x_posts
    email = refined.refined_email if refined else pkg.marketing_email
    ads   = refined.refined_ads   if refined else pkg.ad_copy_sets
    vs    = refined.refined_video_script if refined else pkg.video_script

    # FIX: st.tabs cannot be nested inside st.expander — render the header
    # as a styled markdown block and put tabs at the top level.
    st.markdown(
        "<div style='border:1px solid #D6E0F5;border-radius:10px;"
        "padding:14px 18px 4px 18px;margin-bottom:8px;background:#ffffffee'>"
        "<span style='font-size:1.1rem;font-weight:800;color:#0D1B2A'>✍️ Content Agent Output</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    tab_posts, tab_email, tab_ads, tab_video = st.tabs(
        ["📱 Social Posts", "✉️ Email", "📣 Ad Copy", "🎬 Video Script"]
    )

    # ── Posts ──────────────────────────────────────────────────────
    with tab_posts:
        for i, post in enumerate(posts, 1):
            st.markdown(
                f'<div class="post-card"><div class="platform-tag">'
                f'Post {i} — {post.platform} · {post.post_type}</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**🪝 Hook**\n\n{post.hook}")
            st.markdown(f"**📝 Body**\n\n{post.body}")
            st.markdown(f"**🎯 CTA**\n\n{post.call_to_action}")
            if post.hashtags:
                st.markdown(" ".join(f"`{h}`" for h in post.hashtags))
            plain = f"{post.hook}\n\n{post.body}\n\n{post.call_to_action}"
            if post.hashtags:
                plain += "\n\n" + " ".join(post.hashtags)
            with st.expander(f"📋 Copy Post {i}"):
                st.text_area("", value=plain, height=140,
                             label_visibility="collapsed", key=f"cp_post_{i}")
            st.divider()

    # ── Email ───────────────────────────────────────────────────────
    with tab_email:
        cm, cp = st.columns([1, 2])
        with cm:
            st.markdown(f"**Subject:** {email.subject_line}")
            st.markdown(f"**Preview:** *{email.preview_text}*")
            st.markdown(f"**CTA:** `{email.primary_cta}`")
        with cp:
            body_html = "".join(f"<p>{p}</p>" for p in email.body_paragraphs)
            st.markdown(
                f'<div class="email-preview">'
                f'<div class="email-subject">{email.subject_line}</div>'
                f'<div class="email-preview-text">{email.preview_text}</div>'
                f"<p>{email.greeting}</p>{body_html}"
                f'<a class="email-cta" href="#">{email.primary_cta}</a>'
                f"<br><br><p>{email.sign_off}</p></div>",
                unsafe_allow_html=True,
            )
        plain_email = (f"Subject: {email.subject_line}\nPreview: {email.preview_text}\n\n"
                       f"{email.greeting}\n\n" + "\n\n".join(email.body_paragraphs)
                       + f"\n\n{email.primary_cta}\n\n{email.sign_off}")
        with st.expander("📋 Copy Email"):
            st.text_area("", value=plain_email, height=200,
                         label_visibility="collapsed", key="cp_email")

    # ── Ads ─────────────────────────────────────────────────────────
    with tab_ads:
        for i, ad in enumerate(ads, 1):
            st.markdown(
                f'<div class="ad-card"><div class="platform-tag">'
                f'Ad {i} — {ad.placement}</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Headline:** {ad.headline}")
            st.markdown(f"**Body:** {ad.body_copy}")
            st.markdown(f"**CTA:** `{ad.cta_button}`")
            with st.expander(f"📋 Copy Ad {i}"):
                st.text_area("", value=f"Headline: {ad.headline}\nBody: {ad.body_copy}\nCTA: {ad.cta_button}",
                             height=90, label_visibility="collapsed", key=f"cp_ad_{i}")
            st.divider()

    # ── Video Script ────────────────────────────────────────────────
    with tab_video:
        st.markdown(f"### 🎬 {vs.title}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Platform",  vs.target_platform)
        c2.metric("Duration",  f"{vs.total_duration_seconds}s")
        c3.metric("Scenes",    len(vs.scenes))
        st.markdown(f"**Objective:** {vs.objective}")
        hr()
        for sc in vs.scenes:
            st.markdown(
                f'<div class="video-card">'
                f'<div class="platform-tag">Scene {sc.scene_number} — {sc.duration_seconds}s</div>'
                f'<strong>📷 Visual Cue:</strong><br>{sc.visual_cue}<br><br>'
                f'<strong>🎙️ Dialogue:</strong><br><em>{sc.spoken_dialogue}</em>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown(f"**🎯 Closing CTA:** {vs.closing_cta}")

        script_text = (
            f"# VIDEO SCRIPT: {vs.title}\n"
            f"Platform: {vs.target_platform} | Duration: {vs.total_duration_seconds}s\n"
            f"Objective: {vs.objective}\n\n"
        )
        for sc in vs.scenes:
            script_text += (f"## Scene {sc.scene_number} ({sc.duration_seconds}s)\n"
                             f"Visual: {sc.visual_cue}\nDialogue: {sc.spoken_dialogue}\n\n")
        script_text += f"Closing CTA: {vs.closing_cta}"
        with st.expander("📋 Copy Full Script"):
            st.text_area("", value=script_text, height=250,
                         label_visibility="collapsed", key="cp_script")

    # Content PDF download
    hr()
    content_text = _content_to_text(posts, email, ads, vs)
    pdf_download_button(
        "📥 Download Content Package (PDF)",
        "Content Agent Output",
        content_text,
        "content_package.pdf",
        "dl_content",
    )

def show_brand_alignment_output() -> None:
    refined: RefinedContent | None = st.session_state.refined_content
    if not refined:
        return
    with st.expander("🎨 Brand Alignment Agent Output", expanded=True):
        st.markdown(f"**Alignment Summary**\n\n{refined.alignment_summary}")
        hr()
        st.markdown(f"**Brand Voice Guidelines**\n\n{refined.brand_voice_guidelines}")
        hr()
        st.info(f"**Aligned Value Proposition:** {refined.aligned_value_proposition}")
        hr()
        st.markdown("**Content Guardrails Applied:**")
        for g in refined.content_guardrails:
            st.markdown(f'<div class="guardrail-item">🛡️ {g}</div>', unsafe_allow_html=True)
        hr()
        brand_text = (
            f"# BRAND ALIGNMENT REPORT\n\n"
            f"## Alignment Summary\n{refined.alignment_summary}\n\n"
            f"## Brand Voice Guidelines\n{refined.brand_voice_guidelines}\n\n"
            f"## Value Proposition\n{refined.aligned_value_proposition}\n\n"
            f"## Content Guardrails\n" +
            "\n".join(f"- {g}" for g in refined.content_guardrails)
        )
        pdf_download_button(
            "📥 Download Brand Alignment Report (PDF)",
            "Brand Alignment Agent Output",
            brand_text,
            "brand_alignment_report.pdf",
            "dl_brand",
        )

def _content_to_text(posts, email, ads, video_script) -> str:
    parts = ["# CONTENT PACKAGE\n\n## Social Posts\n"]
    for i, p in enumerate(posts, 1):
        parts.append(f"### Post {i} [{p.platform} / {p.post_type}]\n"
                     f"**Hook:** {p.hook}\n\n{p.body}\n\n**CTA:** {p.call_to_action}\n"
                     f"{'  '.join(p.hashtags)}\n")
    parts.append(f"\n## Marketing Email\n**Subject:** {email.subject_line}\n"
                 f"**Preview:** {email.preview_text}\n\n{email.greeting}\n\n"
                 + "\n\n".join(email.body_paragraphs)
                 + f"\n\n**CTA:** {email.primary_cta}\n\n{email.sign_off}\n")
    parts.append("\n## Ad Copy Sets\n")
    for i, a in enumerate(ads, 1):
        parts.append(f"### Ad {i} [{a.placement}]\nHeadline: {a.headline}\n"
                     f"Body: {a.body_copy}\nCTA: {a.cta_button}\n")
    vs = video_script
    parts.append(f"\n## Video Script\nTitle: {vs.title}\nPlatform: {vs.target_platform}\n"
                 f"Duration: {vs.total_duration_seconds}s\nObjective: {vs.objective}\n\n")
    for sc in vs.scenes:
        parts.append(f"Scene {sc.scene_number} ({sc.duration_seconds}s)\n"
                     f"Visual: {sc.visual_cue}\nDialogue: {sc.spoken_dialogue}\n\n")
    parts.append(f"Closing CTA: {vs.closing_cta}")
    return "".join(parts)

# ---------------------------------------------------------------------------
# Stage 1a — Profile form
# ---------------------------------------------------------------------------
def render_form() -> None:
    section_header("🔍", "Stage 1 — Research Agent",
                   "Enter your company profile. The agent will execute live Saudi Arabia market research.")
    hr()

    profile_source = st.radio("Profile input method:", ["✏️ Manual Entry", "📄 Upload PDF"], horizontal=True)
    pdf_text = ""
    if profile_source == "📄 Upload PDF":
        uploaded = st.file_uploader("Upload company profile PDF", type=["pdf"])
        if uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            with st.spinner("Extracting PDF text…"):
                pdf_text = load_and_validate_pdf(tmp_path) or ""
            Path(tmp_path).unlink(missing_ok=True)
            if pdf_text:
                st.success(f"PDF parsed — {len(pdf_text)} chars extracted.")
            else:
                st.error("Could not extract text — use manual entry.")

    hr()
    with st.form("profile_form"):
        c1, c2 = st.columns(2)
        with c1:
            company_name = st.text_input("Company Name *",
                placeholder="e.g. Acme Technologies Saudi Arabia")
            brand_voice  = st.text_area("Brand Voice / Tone *", height=110,
                placeholder="e.g. Professional yet approachable, data-driven, empowering for Saudi SMEs")
        with c2:
            products  = st.text_area("Products / Services & Pricing *", height=110,
                placeholder="e.g. SaaS ERP platform — Starter SAR 1,800/mo, Pro SAR 5,400/mo")
            past_mkt  = st.text_area("Past Successful Marketing (optional)", height=110,
                placeholder="e.g. LinkedIn Arabic-content series drove 3x pipeline in Q3 2023")

        st.markdown("**Marketing Budget / Cost**")
        bc1, bc2 = st.columns([2, 1])
        with bc1:
            budget_amount = st.number_input(
                "Budget Amount (SAR)",
                min_value=1000, max_value=10_000_000,
                value=20_000, step=1000,
                help="Total monthly or campaign marketing budget in Saudi Riyals",
            )
        with bc2:
            budget_period = st.selectbox(
                "Period",
                ["per month", "per quarter", "per campaign", "per year"],
            )

        submitted = st.form_submit_button(
            "Next: Review & Approve Search →", type="primary", use_container_width=True
        )

    if not submitted:
        return

    errors: list[str] = []
    using_pdf = bool(profile_source == "📄 Upload PDF" and pdf_text)
    if not using_pdf:
        if not company_name.strip(): errors.append("Company Name is required.")
        if not products.strip():     errors.append("Products / Services is required.")
        if not brand_voice.strip():  errors.append("Brand Voice is required.")
    for label, val in [("Company Name", company_name), ("Products", products), ("Brand Voice", brand_voice)]:
        if val.strip() and not validate_input_text(val):
            errors.append(f"'{label}' contains potentially unsafe content.")
    if past_mkt.strip() and not validate_input_text(past_mkt):
        errors.append("'Past Marketing' contains potentially unsafe content.")
    if errors:
        for e in errors: st.error(e)
        return

    budget_str = f"SAR {budget_amount:,} {budget_period}"
    st.session_state.form_company_name   = company_name
    st.session_state.form_products       = products
    st.session_state.form_brand_voice    = brand_voice
    st.session_state.form_past_marketing = past_mkt
    st.session_state.form_pdf_text       = pdf_text
    st.session_state.form_budget         = budget_str
    st.session_state.substage            = "hitl"
    st.rerun()

# ---------------------------------------------------------------------------
# Stage 1b — HITL gate
# ---------------------------------------------------------------------------
def render_hitl() -> None:
    section_header("🔐", "Human-in-the-Loop Approval")
    name   = st.session_state.form_company_name or "Company (from PDF)"
    budget = st.session_state.form_budget
    st.success(f"✅ Profile saved for: **{name}**  |  Budget: **{budget}**")
    st.info(
        "The next step calls the **Tavily Search API** for live Saudi Arabia "
        "market and competitor intelligence. This will consume API credits.\n\n"
        f"**Company:** {name} &emsp; **Budget:** {budget}\n\n"
        "Approve to launch the full 4-agent pipeline."
    )
    c1, c2, _ = st.columns([1, 1, 4])
    with c1:
        if st.button("✅ Approve & Run", type="primary", use_container_width=True):
            st.session_state.substage = "running"
            st.rerun()
    with c2:
        if st.button("✏️ Edit Profile", use_container_width=True):
            st.session_state.substage = "form"
            st.rerun()

# ---------------------------------------------------------------------------
# Stage 1c — Research pipeline execution
# ---------------------------------------------------------------------------
def run_research_pipeline() -> None:
    section_header("🔍", "Research Agent Running…")
    openai_client, tavily_client = resolve_clients()
    agent = ResearchAgent(openai_client=openai_client, tavily_client=tavily_client)

    pdf_text     = st.session_state.form_pdf_text
    company_name = st.session_state.form_company_name
    products     = st.session_state.form_products
    brand_voice  = st.session_state.form_brand_voice
    past_mkt     = st.session_state.form_past_marketing
    budget       = st.session_state.form_budget

    bar    = st.progress(0, text="Building company profile…")
    status = st.empty()

    try:
        if pdf_text:
            status.info("Parsing company profile from PDF…")
            profile = agent.parse_profile_from_text(pdf_text)
            if company_name.strip(): profile.company_name        = company_name.strip()
            if brand_voice.strip():  profile.brand_voice         = brand_voice.strip()
            if products.strip():     profile.products_and_prices = products.strip()
            profile.marketing_budget = budget
        else:
            profile = CompanyProfile(
                company_name=company_name.strip(),
                products_and_prices=products.strip(),
                brand_voice=brand_voice.strip(),
                past_successful_marketing=past_mkt.strip() or "No past marketing plans provided.",
                marketing_budget=budget,
            )

        queries  = agent.build_search_queries(profile)
        snippets: list[str] = []
        for i, q in enumerate(queries):
            status.info(f"Saudi Arabia search ({i+1}/{len(queries)}): {q[:60]}…")
            try:
                res = tavily_client.search(q, search_depth="advanced", max_results=5)
                for r in res.get("results", []):
                    s = r.get("content", "")
                    if s: snippets.append(s[:500])
            except Exception as exc:
                status.warning(f"Search note: {exc}")
            bar.progress(10 + int((i+1) / len(queries) * 45),
                         text=f"Searching ({i+1}/{len(queries)})…")

        context = "\n\n---\n\n".join(snippets) if snippets else "No live results; using parametric knowledge."

        bar.progress(60, text="Synthesising Saudi Arabia research report…")
        status.info("Synthesising report…")
        research_report = agent.synthesise_report(profile, context)

        bar.progress(90, text="Persisting to ChromaDB…")
        research_doc_id = agent.persist_research(profile, research_report)

        bar.progress(100, text="✓ Research complete!")
        status.empty()
        time.sleep(0.3)

        st.session_state.profile         = profile
        st.session_state.research_report = research_report
        st.session_state.research_doc_id = research_doc_id
        st.session_state.substage        = "form"
        st.session_state.stage           = 2
        st.rerun()

    except Exception as exc:
        bar.empty(); status.empty()
        st.error(f"❌ Research Agent error: {exc}")
        st.exception(exc)
        if st.button("← Go back"):
            st.session_state.substage = "hitl"
            st.rerun()

# ---------------------------------------------------------------------------
# Stage 2 — Strategy Agent
# ---------------------------------------------------------------------------
def render_stage_2() -> None:
    show_research_output()
    hr()
    section_header("🗺️", "Stage 2 — Strategy Agent",
                   "Generating budget-aware ICP, GTM strategy, and 30-day marketing calendar.")

    profile: CompanyProfile = st.session_state.profile
    st.info(f"**Budget:** {profile.marketing_budget}  |  Strategy will be explicitly designed for this budget.")

    if st.button("🗺️ Generate ICP, Strategy & Calendar", type="primary"):
        openai_client, _ = resolve_clients()
        with st.spinner("Building Saudi Arabia GTM strategy and 30-day calendar…"):
            try:
                agent = StrategyAgent(openai_client=openai_client)
                strategy_text, calendar_md, strategy_doc_id = agent.run(
                    profile=profile,
                    research_doc_id=st.session_state.research_doc_id,
                )
                st.session_state.gtm_strategy    = strategy_text
                st.session_state.calendar_md     = calendar_md
                st.session_state.strategy_doc_id = strategy_doc_id
                st.session_state.stage           = 3
                st.rerun()
            except Exception as exc:
                st.error(f"❌ Strategy Agent error: {exc}")
                st.exception(exc)

# ---------------------------------------------------------------------------
# Stage 3 — Content Agent
# ---------------------------------------------------------------------------
def render_stage_3() -> None:
    show_research_output()
    show_strategy_output()
    hr()
    section_header("✍️", "Stage 3 — Content Agent",
                   "Generating social posts, email, ad copy, and video script for Saudi Arabia.")

    if st.button("✍️ Generate Full Content Package", type="primary"):
        openai_client, _ = resolve_clients()
        with st.spinner("Generating Saudi Arabia content package (posts, email, ads, video)…"):
            try:
                agent = ContentAgent(openai_client=openai_client)
                content_package, content_doc_id = agent.run(
                    profile=st.session_state.profile,
                    strategy_doc_id=st.session_state.strategy_doc_id,
                )
                st.session_state.content_package = content_package
                st.session_state.content_doc_id  = content_doc_id
                st.session_state.stage           = 4
                st.rerun()
            except Exception as exc:
                st.error(f"❌ Content Agent error: {exc}")
                st.exception(exc)

# ---------------------------------------------------------------------------
# Stage 4 — Brand Alignment
# ---------------------------------------------------------------------------
def render_stage_4() -> None:
    show_research_output()
    show_strategy_output()
    show_content_output()
    hr()
    section_header("🎨", "Stage 4 — Brand Alignment Agent",
                   "Final quality gate: auditing and refining ALL content against your brand and KSA compliance standards.")

    if st.button("🎨 Run Final Brand Alignment", type="primary"):
        openai_client, _ = resolve_clients()
        with st.spinner("Running final brand alignment audit on all content…"):
            try:
                agent = BrandAlignmentAgent(openai_client=openai_client)
                refined, brand_doc_id = agent.run(
                    profile=st.session_state.profile,
                    pkg=st.session_state.content_package,
                    content_doc_id=st.session_state.content_doc_id,
                )
                st.session_state.refined_content = refined
                st.session_state.brand_doc_id    = brand_doc_id
                st.session_state.stage           = 5
                st.rerun()
            except Exception as exc:
                st.error(f"❌ Brand Alignment error: {exc}")
                st.exception(exc)

# ---------------------------------------------------------------------------
# Stage 5 — Final results dashboard
# ---------------------------------------------------------------------------
def render_final_results() -> None:
    profile: CompanyProfile = st.session_state.profile

    st.markdown(
        f'<div class="result-banner">'
        f'✅ Pipeline Complete — {profile.company_name} | Saudi Arabia GTM Intelligence'
        f'</div>',
        unsafe_allow_html=True,
    )

    show_brand_alignment_output()
    show_content_output()
    show_strategy_output()
    show_research_output()

    hr()
    st.markdown(
        f"**Session IDs** &emsp; "
        f"Research: `{st.session_state.research_doc_id}` &emsp; "
        f"Strategy: `{st.session_state.strategy_doc_id}` &emsp; "
        f"Content: `{st.session_state.content_doc_id}` &emsp; "
        f"Brand: `{st.session_state.brand_doc_id}`"
    )

# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------
def main() -> None:
    render_sidebar()

    st.markdown(
        "<h1 style='font-size:1.9rem;font-weight:800;margin-bottom:2px;color:#0D1B2A'>"
        "🧠 MarketMind AI</h1>"
        "<p style='color:#475569;margin:0 0 8px 0;font-size:.9rem'>"
        "Saudi Arabia GTM Intelligence Platform · Powered by 4 Agentic AI Specialists</p>",
        unsafe_allow_html=True,
    )

    stage    = st.session_state.stage
    substage = st.session_state.substage

    if stage == 1:
        if substage == "form":      render_form()
        elif substage == "hitl":    render_hitl()
        elif substage == "running": run_research_pipeline()
    elif stage == 2: render_stage_2()
    elif stage == 3: render_stage_3()
    elif stage == 4: render_stage_4()
    elif stage >= 5: render_final_results()


if __name__ == "__main__":
    main()
