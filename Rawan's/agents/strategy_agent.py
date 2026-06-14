"""
agents/strategy_agent.py
------------------------
Agent 2 — Strategy Agent | MarketMind AI

Role: Defines positioning, ICP, messaging framework, channel strategy, and
      marketing calendar — explicitly budget-aware and Saudi Arabia-focused.
      Generates a 30-day content marketing calendar table.
"""

from __future__ import annotations
import uuid
from openai import OpenAI
from pydantic import BaseModel, Field
from agents.research_agent import CompanyProfile
from database_setup import get_market_research_collection, get_icp_and_strategy_collection


class ICPSegment(BaseModel):
    segment_name: str        = Field(..., description="Descriptive label (e.g. 'Mid-Market CFOs in KSA').")
    firmographics: str       = Field(..., description="Sector, size, revenue band, KSA geography.")
    job_titles: list[str]    = Field(..., description="Specific decision-maker and influencer titles.")
    pain_points: list[str]   = Field(..., description="3–5 professional pain points.")
    behavioural_traits: str  = Field(..., description="Research, evaluation, and buying behaviour.")
    value_drivers: str       = Field(..., description="Business outcomes they seek.")


class MarketingCalendarEntry(BaseModel):
    day: int              = Field(..., description="Day number (1–30).")
    activity: str         = Field(..., description="Marketing activity (e.g. 'LinkedIn Post', 'Cold Email').")
    description: str      = Field(..., description="One-line description of what to do.")
    channel: str          = Field(..., description="Channel: LinkedIn / Email / Instagram / Snapchat / Event / etc.")
    budget_allocation: str = Field(..., description="Estimated budget slice for this activity (e.g. SAR 2,000).")


class GTMStrategyDocument(BaseModel):
    executive_summary: str         = Field(...)
    icp_segments: list[ICPSegment] = Field(...)
    value_proposition: str         = Field(...)
    messaging_pillars: list[str]   = Field(...)
    channel_strategy: str          = Field(...)
    budget_breakdown: str          = Field(..., description="How to allocate the marketing budget across channels/phases.")
    past_marketing_scaled: str     = Field(...)
    execution_roadmap: str         = Field(...)
    kpis: list[str]                = Field(...)
    risk_mitigation: str           = Field(...)
    content_calendar: list[MarketingCalendarEntry] = Field(
        ..., description="30-day content marketing calendar with budget allocations."
    )


class StrategyAgent:
    """
    Backstory:
        You are a Senior GTM Strategist with 12 years of experience launching
        B2B products in Saudi Arabia and the GCC. You have led strategy for
        Fortune 500 companies entering the Kingdom under Vision 2030. You
        build strategies rooted in real firmographic data, Saudi buyer
        psychology, and budget realism. You never generate fictional personas.

    Goal:
        Produce a precise, budget-aware Saudi Arabia GTM strategy: define who
        the ideal customer is, how to position the product for KSA audiences,
        which channels deliver the best ROI within the specified budget, and
        generate a concrete 30-day marketing calendar.
    """

    CHAT_MODEL      = "gpt-4o-mini"
    EMBEDDING_MODEL = "text-embedding-3-small"

    def __init__(self, openai_client: OpenAI) -> None:
        self.openai_client       = openai_client
        self.research_collection = get_market_research_collection()
        self.strategy_collection = get_icp_and_strategy_collection()

    def fetch_research(self, research_doc_id: str) -> str:
        result = self.research_collection.get(ids=[research_doc_id], include=["documents"])
        if not result["documents"] or not result["documents"][0]:
            raise ValueError(f"Research doc '{research_doc_id}' not found.")
        return result["documents"][0]

    def generate_strategy(self, profile: CompanyProfile, research_report: str) -> GTMStrategyDocument:
        past_note = (
            f"Past Marketing to Scale:\n{profile.past_successful_marketing}"
            if profile.past_successful_marketing != "No past marketing plans provided."
            else "No prior marketing history — build from first principles."
        )
        system_prompt = (
            "You are a Senior Go-To-Market Strategist specialising in Saudi Arabia and GCC B2B markets. "
            "You build strategies grounded in real firmographic data and Saudi buyer psychology.\n\n"
            "CRITICAL ICP RULES:\n"
            "- Define segments using ONLY: sectors, company sizes, revenue bands, job titles, "
            "  pain points, and behavioural traits.\n"
            "- DO NOT generate fictional names, social handles, or invented personas.\n"
            "- Budget allocation must be explicit and realistic for the Saudi market."
        )
        user_prompt = (
            f"Company: {profile.company_name}\n"
            f"Products & Pricing: {profile.products_and_prices}\n"
            f"Brand Voice: {profile.brand_voice}\n"
            f"Marketing Budget: {profile.marketing_budget}\n"
            f"{past_note}\n\n"
            "Produce a complete Saudi Arabia GTM strategy:\n\n"
            "EXECUTIVE SUMMARY: Core strategic intent and primary KSA market opportunity.\n\n"
            "ICP SEGMENTS (2–3): firmographics, job titles, pain points, behavioural traits, value drivers.\n\n"
            "VALUE PROPOSITION: Localised for Saudi/GCC B2B buyers.\n\n"
            "MESSAGING PILLARS: 3–5 themes resonating with KSA B2B decision-makers.\n\n"
            "CHANNEL STRATEGY: Prioritised for KSA — LinkedIn, Snapchat, Instagram, X, "
            "WhatsApp Business, Google, industry events — with ROI rationale per channel.\n\n"
            f"BUDGET BREAKDOWN: Allocate {profile.marketing_budget} across channels and phases. "
            "Be specific with SAR amounts or percentages per channel.\n\n"
            "SCALING PAST SUCCESSES: Which past tactics transfer to KSA and how to localise them.\n\n"
            "90-DAY ROADMAP: Month 1, 2, 3 with specific deliverables.\n\n"
            "KPIs: Quantifiable targets per phase.\n\n"
            "RISK MITIGATION: Top 3 KSA-specific risks with contingency plans.\n\n"
            "CONTENT CALENDAR: Generate exactly 30 entries for a 30-day marketing calendar. "
            "Each entry: day number, activity type, one-line description, channel, and "
            f"budget allocation from the total budget of {profile.marketing_budget}. "
            "Include a mix of: LinkedIn posts, cold emails, Instagram stories, Snapchat ads, "
            "WhatsApp follow-ups, webinars, and engagement activities.\n\n"
            f"Saudi Arabia Market Research:\n{research_report}"
        )
        completion = self.openai_client.beta.chat.completions.parse(
            model=self.CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            response_format=GTMStrategyDocument,
            max_tokens=3500,
        )
        return completion.choices[0].message.parsed

    @staticmethod
    def strategy_to_text(doc: GTMStrategyDocument) -> str:
        lines: list[str] = ["# SAUDI ARABIA GTM STRATEGY\n"]
        lines += ["## Executive Summary", doc.executive_summary, ""]
        lines += ["## ICP Segments"]
        for i, seg in enumerate(doc.icp_segments, 1):
            lines += [
                f"\n### Segment {i}: {seg.segment_name}",
                f"**Firmographics:** {seg.firmographics}",
                f"**Job Titles:** {', '.join(seg.job_titles)}",
                "**Pain Points:**",
                *[f"  - {pp}" for pp in seg.pain_points],
                f"**Behavioural Traits:** {seg.behavioural_traits}",
                f"**Value Drivers:** {seg.value_drivers}",
            ]
        lines += ["", "## Localised Value Proposition", doc.value_proposition, ""]
        lines += ["## Messaging Pillars"]
        lines += [f"  {i}. {p}" for i, p in enumerate(doc.messaging_pillars, 1)]
        lines += ["", "## Channel Strategy", doc.channel_strategy, ""]
        lines += ["## Budget Breakdown", doc.budget_breakdown, ""]
        lines += ["## Scaling Past Marketing Successes", doc.past_marketing_scaled, ""]
        lines += ["## 90-Day Execution Roadmap", doc.execution_roadmap, ""]
        lines += ["## KPIs & Success Metrics"]
        lines += [f"  - {k}" for k in doc.kpis]
        lines += ["", "## Risk Mitigation", doc.risk_mitigation]
        return "\n".join(lines)

    @staticmethod
    def calendar_to_markdown(doc: GTMStrategyDocument) -> str:
        """Render the 30-day calendar as a Markdown table."""
        rows = ["| Day | Activity | Description | Channel | Budget |",
                "|-----|----------|-------------|---------|--------|"]
        for e in doc.content_calendar:
            desc = e.description.replace("|", "–")
            rows.append(f"| {e.day} | {e.activity} | {desc} | {e.channel} | {e.budget_allocation} |")
        return "\n".join(rows)

    def embed_text(self, text: str) -> list[float]:
        return self.openai_client.embeddings.create(
            model=self.EMBEDDING_MODEL, input=text
        ).data[0].embedding

    def persist_strategy(self, profile: CompanyProfile, strategy_text: str,
                         calendar_md: str, research_doc_id: str) -> str:
        doc_id    = f"strategy_{uuid.uuid4().hex}"
        combined  = strategy_text + "\n\n## 30-Day Content Calendar\n" + calendar_md
        embedding = self.embed_text(combined)
        self.strategy_collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[combined],
            metadatas=[{
                "company_name":           profile.company_name,
                "brand_voice":            profile.brand_voice,
                "marketing_budget":       profile.marketing_budget,
                "source_research_doc_id": research_doc_id,
                "region":                 "Saudi Arabia / Middle East",
                "agent":                  "strategy",
            }],
        )
        return doc_id

    def run(self, profile: CompanyProfile, research_doc_id: str) -> tuple[str, str, str]:
        """Returns (strategy_text, calendar_markdown, strategy_doc_id)."""
        research    = self.fetch_research(research_doc_id)
        doc         = self.generate_strategy(profile, research)
        text        = self.strategy_to_text(doc)
        calendar_md = self.calendar_to_markdown(doc)
        doc_id      = self.persist_strategy(profile, text, calendar_md, research_doc_id)
        return text, calendar_md, doc_id