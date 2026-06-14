"""
agents/content_agent.py
-----------------------
Agent 3 — Content Agent | MarketMind AI

Role: Generates social posts (LinkedIn, X, Snapchat), marketing email,
      ad copy sets, and a full video script — Saudi Arabia / GCC focused.
      Operates strictly within brand and safety guardrails.
      Brand Alignment Agent runs AFTER this agent as the final quality gate.
"""

from __future__ import annotations
import uuid
from openai import OpenAI
from pydantic import BaseModel, Field
from agents.research_agent import CompanyProfile
from database_setup import get_icp_and_strategy_collection, get_marketing_strategies_collection


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SocialPost(BaseModel):
    platform: str      = Field(..., description="'LinkedIn', 'X', or 'Snapchat'.")
    post_type: str     = Field(..., description="'Thought Leadership', 'Pain Point', or 'Insight'.")
    hook: str          = Field(..., description="Opening line to stop the scroll.")
    body: str          = Field(..., description="Value-dense educational body.")
    call_to_action: str = Field(..., description="One specific action CTA.")
    hashtags: list[str] = Field(default_factory=list)


class AdCopySet(BaseModel):
    placement: str  = Field(..., description="'LinkedIn Sponsored Content', 'Google Search', or 'Snapchat Ad'.")
    headline: str   = Field(..., description="Primary headline — max 150 chars.")
    body_copy: str  = Field(..., description="Ad body — concise, benefit-led.")
    cta_button: str = Field(..., description="CTA button text — max 25 chars.")


class MarketingEmail(BaseModel):
    subject_line: str        = Field(...)
    preview_text: str        = Field(...)
    greeting: str            = Field(...)
    body_paragraphs: list[str] = Field(...)
    primary_cta: str         = Field(...)
    sign_off: str            = Field(...)


class VideoScriptScene(BaseModel):
    scene_number: int    = Field(...)
    visual_cue: str      = Field(..., description="Camera setup, location, props, on-screen text.")
    spoken_dialogue: str = Field(..., description="Exact words the presenter says.")
    duration_seconds: int = Field(..., description="Estimated scene duration in seconds.")


class VideoScript(BaseModel):
    title: str                    = Field(..., description="Video title / hook for the thumbnail.")
    total_duration_seconds: int   = Field(..., description="Total estimated video length in seconds.")
    objective: str                = Field(..., description="What this video is designed to achieve.")
    target_platform: str          = Field(..., description="'Instagram Reels', 'YouTube', 'LinkedIn Video', etc.")
    scenes: list[VideoScriptScene] = Field(..., description="Full scene-by-scene script.")
    closing_cta: str              = Field(..., description="Final spoken CTA and on-screen action.")


class ContentPackage(BaseModel):
    linkedin_x_posts: list[SocialPost] = Field(..., description="3 social posts (2 LinkedIn + 1 X).")
    marketing_email: MarketingEmail    = Field(...)
    ad_copy_sets: list[AdCopySet]      = Field(..., description="3 ad sets (LinkedIn, Google, Snapchat).")
    video_script: VideoScript          = Field(..., description="Full promotional video script.")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ContentAgent:
    """
    Backstory:
        You are an award-winning B2B Content Director with 10 years producing
        marketing copy for enterprise brands scaling in Saudi Arabia and the GCC.
        Your writing is culturally intelligent, commercially sharp, and platform-
        calibrated. You operate under strict content governance — every word must
        be professional, business-focused, factually accurate, and compliant with
        the guardrails provided. You never produce offensive, political,
        discriminatory, or culturally insensitive content.

    Goal:
        Transform the brand-approved GTM strategy into high-performing, platform-
        ready content for the Saudi Arabia market: social posts, email, ad copy,
        and a full video production script — all perfectly on-brand and compliant.
    """

    CHAT_MODEL      = "gpt-4o-mini"
    EMBEDDING_MODEL = "text-embedding-3-small"

    _SAFETY_GUARDRAILS = (
        "MANDATORY CONTENT SAFETY RULES — non-negotiable for every output:\n"
        "1. NEVER produce content that is offensive, derogatory, or discriminatory.\n"
        "2. NEVER include political commentary, geopolitical analysis, or conflict references.\n"
        "3. NEVER include religious commentary, endorsements, or critiques.\n"
        "4. NEVER produce sexually suggestive, violent, or unprofessional content.\n"
        "5. NEVER make unverified statistical claims or present speculation as fact.\n"
        "6. ALL content must be professional, factual, and suited for GCC corporate audiences.\n"
        "7. ALL content must be culturally respectful for Saudi Arabia / GCC context."
    )

    def __init__(self, openai_client: OpenAI) -> None:
        self.openai_client       = openai_client
        self.strategy_collection = get_icp_and_strategy_collection()
        self.content_collection  = get_marketing_strategies_collection()

    def fetch_strategy(self, strategy_doc_id: str) -> str:
        result = self.strategy_collection.get(ids=[strategy_doc_id], include=["documents"])
        if not result["documents"] or not result["documents"][0]:
            raise ValueError(f"Strategy doc '{strategy_doc_id}' not found.")
        return result["documents"][0]

    def generate_content_package(
        self,
        profile: CompanyProfile,
        strategy_text: str,
        guardrails: list[str] | None = None,
    ) -> ContentPackage:
        guardrails_block = "\n".join(
            f"  {i+1}. {g}" for i, g in enumerate(guardrails or [])
        ) or "  (Standard professional and cultural compliance applies.)"

        system_prompt = (
            "You are an award-winning B2B Content Director specialising in Saudi Arabia and GCC markets.\n\n"
            f"{self._SAFETY_GUARDRAILS}"
        )
        user_prompt = (
            f"Company: {profile.company_name}\n"
            f"Products & Pricing: {profile.products_and_prices}\n"
            f"Brand Voice: {profile.brand_voice}\n"
            f"Marketing Budget: {profile.marketing_budget}\n\n"
            f"Content Guardrails (MUST follow):\n{guardrails_block}\n\n"
            "Generate a complete Saudi Arabia content package:\n\n"
            "SOCIAL POSTS (3):\n"
            "  Post 1 — LinkedIn, Thought Leadership: KSA industry trend insight. "
            "  Strong data hook. ≥150 words. Match brand voice.\n"
            "  Post 2 — LinkedIn, Pain Point: Position as solution to a specific GCC B2B pain. "
            "  Include a 3-step framework. ≥150 words.\n"
            "  Post 3 — X (Twitter), Insight: Hook ≤280 chars + 2–3 thread points.\n"
            "  All posts: Saudi/GCC-relevant hashtags (English + Arabic transliterations).\n\n"
            "MARKETING EMAIL:\n"
            "  Target: C-suite / VP / Director in KSA companies.\n"
            "  Subject: compelling, spam-trigger-free.\n"
            "  Body: educate → build trust → single CTA. 3–4 paragraphs. On-brand throughout.\n\n"
            "AD COPY SETS (3):\n"
            "  Set 1 — LinkedIn Sponsored Content\n"
            "  Set 2 — Google Search\n"
            "  Set 3 — Snapchat Ad (casual, visual-first tone for Saudi Snapchat audience)\n"
            "  Each: headline (≤150 chars) + body + CTA button (≤25 chars).\n\n"
            "VIDEO SCRIPT (full production script):\n"
            "  Platform: Instagram Reels / LinkedIn Video (60–90 seconds).\n"
            "  Include: title/thumbnail hook, objective, all scenes with visual cues "
            "  (camera angle, on-screen text, props, location), exact spoken dialogue, "
            "  scene duration in seconds, and closing CTA.\n"
            "  Tone must match brand voice. Script must be production-ready.\n\n"
            f"Saudi Arabia GTM Strategy:\n{strategy_text}"
        )
        completion = self.openai_client.beta.chat.completions.parse(
            model=self.CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            response_format=ContentPackage,
            max_tokens=4000,
        )
        return completion.choices[0].message.parsed

    def embed_text(self, text: str) -> list[float]:
        return self.openai_client.embeddings.create(
            model=self.EMBEDDING_MODEL, input=text
        ).data[0].embedding

    def persist_content(self, profile: CompanyProfile, pkg: ContentPackage, strategy_doc_id: str) -> str:
        doc_id = f"content_{uuid.uuid4().hex}"
        summary = (
            f"Company: {profile.company_name}\n"
            + "\n".join(f"Post {i}: {p.hook[:80]}" for i, p in enumerate(pkg.linkedin_x_posts, 1))
            + f"\nEmail: {pkg.marketing_email.subject_line}"
            + f"\nVideo: {pkg.video_script.title}"
        )
        self.content_collection.upsert(
            ids=[doc_id],
            embeddings=[self.embed_text(summary)],
            documents=[summary],
            metadatas=[{
                "company_name":          profile.company_name,
                "source_strategy_doc_id": strategy_doc_id,
                "agent":                 "content",
            }],
        )
        return doc_id

    def run(self, profile: CompanyProfile, strategy_doc_id: str,
            guardrails: list[str] | None = None) -> tuple[ContentPackage, str]:
        strategy_text   = self.fetch_strategy(strategy_doc_id)
        content_package = self.generate_content_package(profile, strategy_text, guardrails)
        content_doc_id  = self.persist_content(profile, content_package, strategy_doc_id)
        return content_package, content_doc_id