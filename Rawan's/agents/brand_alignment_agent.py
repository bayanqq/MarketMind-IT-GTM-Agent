"""
agents/brand_alignment_agent.py
--------------------------------
Agent 4 — Brand Alignment Agent | MarketMind AI

Role: Final quality gate. Runs AFTER the Content Agent. Audits and refines
      all generated content (posts, email, ads, video script) against the
      corporate knowledge base and brand guidelines.
"""

from __future__ import annotations
import uuid
from openai import OpenAI
from pydantic import BaseModel, Field
from agents.research_agent import CompanyProfile
from agents.content_agent import ContentPackage, SocialPost, AdCopySet, MarketingEmail, VideoScript
from database_setup import get_marketing_strategies_collection, get_brand_alignment_collection


class RefinedContent(BaseModel):
    """Brand-verified and refined version of the full content package."""
    alignment_summary: str          = Field(..., description="Overall audit summary: what was changed and why.")
    brand_voice_guidelines: str     = Field(..., description="Concrete do/don't writing rules for future use.")
    content_guardrails: list[str]   = Field(..., description="Explicit compliance rules applied.")
    refined_posts: list[SocialPost] = Field(..., description="Brand-refined social posts.")
    refined_email: MarketingEmail   = Field(..., description="Brand-refined email.")
    refined_ads: list[AdCopySet]    = Field(..., description="Brand-refined ad copy sets.")
    refined_video_script: VideoScript = Field(..., description="Brand-refined video script.")
    aligned_value_proposition: str  = Field(..., description="Value prop in exact brand voice.")


class BrandAlignmentAgent:
    """
    Backstory:
        You are a Senior Brand Strategist and Compliance Officer with 10 years
        of experience governing marketing output for global B2B brands in the
        Middle East. You are the absolute final quality gate — nothing publishes
        without your approval. You audit every word for brand consistency,
        cultural appropriateness for Saudi Arabia, and professional compliance.
        You refine content to be on-brand without losing commercial impact.

    Goal:
        Audit ALL generated content (posts, email, ads, video script) against
        the company's brand voice and knowledge base. Refine any off-brand or
        non-compliant elements. Produce a brand-verified content package with
        explicit guardrails documented for future campaigns.
    """

    CHAT_MODEL      = "gpt-4o-mini"
    EMBEDDING_MODEL = "text-embedding-3-small"

    _SAFETY_PROHIBITIONS = (
        "MANDATORY PROHIBITIONS — enforce for ALL content:\n"
        "- PROHIBITED: Offensive, discriminatory, or derogatory content of any kind.\n"
        "- PROHIBITED: Political commentary or geopolitical references.\n"
        "- PROHIBITED: Religious endorsements, critiques, or insensitive references.\n"
        "- PROHIBITED: Culturally inappropriate content for Saudi Arabia / GCC.\n"
        "- PROHIBITED: Unverified statistical claims or misleading data.\n"
        "- PROHIBITED: Sexually suggestive, violent, or unprofessional content.\n"
        "- REQUIRED: All copy professional, factual, and business-focused."
    )

    def __init__(self, openai_client: OpenAI) -> None:
        self.openai_client        = openai_client
        self.content_collection   = get_marketing_strategies_collection()
        self.alignment_collection = get_brand_alignment_collection()

    def _serialize_package(self, pkg: ContentPackage) -> str:
        """Convert ContentPackage to a readable string for the LLM."""
        parts = ["=== SOCIAL POSTS ==="]
        for i, p in enumerate(pkg.linkedin_x_posts, 1):
            parts += [f"\nPost {i} [{p.platform} / {p.post_type}]",
                      f"Hook: {p.hook}", f"Body: {p.body}",
                      f"CTA: {p.call_to_action}", f"Hashtags: {' '.join(p.hashtags)}"]
        e = pkg.marketing_email
        parts += ["\n=== EMAIL ===",
                  f"Subject: {e.subject_line}", f"Preview: {e.preview_text}",
                  f"Greeting: {e.greeting}", *e.body_paragraphs,
                  f"CTA: {e.primary_cta}", f"Sign-off: {e.sign_off}"]
        parts.append("\n=== AD COPY ===")
        for i, a in enumerate(pkg.ad_copy_sets, 1):
            parts += [f"\nAd {i} [{a.placement}]",
                      f"Headline: {a.headline}", f"Body: {a.body_copy}", f"CTA: {a.cta_button}"]
        vs = pkg.video_script
        parts += ["\n=== VIDEO SCRIPT ===",
                  f"Title: {vs.title}", f"Platform: {vs.target_platform}",
                  f"Objective: {vs.objective}", f"Duration: {vs.total_duration_seconds}s"]
        for sc in vs.scenes:
            parts += [f"\nScene {sc.scene_number} ({sc.duration_seconds}s)",
                      f"Visual: {sc.visual_cue}", f"Dialogue: {sc.spoken_dialogue}"]
        parts.append(f"Closing CTA: {vs.closing_cta}")
        return "\n".join(parts)

    def run_alignment(self, profile: CompanyProfile, pkg: ContentPackage) -> RefinedContent:
        content_str = self._serialize_package(pkg)
        system_prompt = (
            "You are a Senior Brand Strategist and Compliance Officer — the absolute "
            "final quality gate before any content is published. You audit every piece "
            "of generated content for:\n"
            "  1. Brand consistency — language, tone, and terminology match the brand voice exactly.\n"
            "  2. Cultural appropriateness for Saudi Arabia / GCC professional audiences.\n"
            "  3. Full compliance with safety and professional standards.\n\n"
            f"{self._SAFETY_PROHIBITIONS}"
        )
        user_prompt = (
            f"Company: {profile.company_name}\n"
            f"Brand Voice: {profile.brand_voice}\n"
            f"Products: {profile.products_and_prices}\n"
            f"Past Marketing: {profile.past_successful_marketing}\n\n"
            "AUDIT AND REFINE the following content package. For each piece:\n"
            "  - Identify any off-brand language, tone mismatches, or compliance issues.\n"
            "  - Rewrite to fix all issues while preserving commercial impact.\n"
            "  - Document all changes in the alignment_summary.\n\n"
            "Also produce:\n"
            "  - brand_voice_guidelines: concrete do/don't rules from this brand profile.\n"
            "  - content_guardrails: the full list of rules applied (brand + safety).\n"
            "  - aligned_value_proposition: value prop rewritten in the exact brand voice.\n\n"
            f"Content Package to Audit:\n{content_str}"
        )
        completion = self.openai_client.beta.chat.completions.parse(
            model=self.CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            response_format=RefinedContent,
            max_tokens=4000,
        )
        return completion.choices[0].message.parsed

    def embed_text(self, text: str) -> list[float]:
        return self.openai_client.embeddings.create(
            model=self.EMBEDDING_MODEL, input=text
        ).data[0].embedding

    def persist_alignment(self, profile: CompanyProfile, refined: RefinedContent,
                          content_doc_id: str) -> str:
        doc_id    = f"brand_{uuid.uuid4().hex}"
        summary   = f"Brand alignment for {profile.company_name}. {refined.alignment_summary[:300]}"
        guardrails_str = " | ".join(refined.content_guardrails)[:800]
        self.alignment_collection.upsert(
            ids=[doc_id],
            embeddings=[self.embed_text(summary)],
            documents=[summary],
            metadatas=[{
                "company_name":          profile.company_name,
                "brand_voice":           profile.brand_voice,
                "alignment_summary":     refined.alignment_summary[:400],
                "content_guardrails":    guardrails_str,
                "aligned_value_prop":    refined.aligned_value_proposition[:400],
                "source_content_doc_id": content_doc_id,
                "agent":                 "brand_alignment",
            }],
        )
        return doc_id

    def run(self, profile: CompanyProfile, pkg: ContentPackage,
            content_doc_id: str) -> tuple[RefinedContent, str]:
        refined  = self.run_alignment(profile, pkg)
        brand_id = self.persist_alignment(profile, refined, content_doc_id)
        return refined, brand_id