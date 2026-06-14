"""
agents/research_agent.py
------------------------
Agent 1 — Research Agent | MarketMind AI

Role: Performs market, competitor, and trend analysis focused explicitly on
      Saudi Arabia and the broader Middle East / GCC region.
"""

from __future__ import annotations
import uuid
from openai import OpenAI
from pydantic import BaseModel, Field
from tavily import TavilyClient
from database_setup import get_market_research_collection
from utils.security import sanitize_for_query


class CompanyProfile(BaseModel):
    company_name: str = Field(..., description="Full legal or trading name.")
    products_and_prices: str = Field(..., description="Products/services and pricing.")
    brand_voice: str = Field(..., description="Tone and communication style.")
    past_successful_marketing: str = Field(
        default="No past marketing plans provided.",
        description="Optional summary of past successful campaigns.",
    )
    marketing_budget: str = Field(
        default="Not specified",
        description="Marketing budget / cost scope for the plan (e.g. SAR 50,000/month).",
    )


class ResearchAgent:
    """
    Backstory:
        You are a Principal Market Intelligence Analyst with 15 years of
        experience specialising in Saudi Arabia and the GCC. You have led
        market entry studies for multinationals entering the Kingdom under
        Vision 2030, and your reports are used by C-suites to make
        high-stakes expansion decisions. You combine live search data with
        deep regional expertise to deliver actionable intelligence.

    Goal:
        Deliver a comprehensive Saudi Arabia-first market research report
        covering competitive dynamics, audience profiles, channel intelligence,
        and growth trends — comparing the client against direct competitors
        operating in Saudi Arabia and the wider Middle East.
    """

    EMBEDDING_MODEL = "text-embedding-3-small"
    CHAT_MODEL      = "gpt-4o-mini"

    def __init__(self, openai_client: OpenAI, tavily_client: TavilyClient) -> None:
        self.openai_client = openai_client
        self.tavily_client = tavily_client
        self.collection    = get_market_research_collection()

    def parse_profile_from_text(self, extracted_text: str) -> CompanyProfile:
        completion = self.openai_client.beta.chat.completions.parse(
            model=self.CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a business analyst. Extract a structured company profile."},
                {"role": "user",   "content": (
                    f"Extract from the text below:\n\n{extracted_text}\n\n"
                    "Fields: company_name, products_and_prices, brand_voice, "
                    "past_successful_marketing, marketing_budget. Use 'Not specified' for missing fields."
                )},
            ],
            response_format=CompanyProfile,
        )
        return completion.choices[0].message.parsed

    def build_search_queries(self, profile: CompanyProfile) -> list[str]:
        company  = sanitize_for_query(profile.company_name, max_chars=55)
        products = sanitize_for_query(profile.products_and_prices, max_chars=70)
        raw = [
            f"{company} market opportunity Saudi Arabia Vision 2030",
            f"{products} B2B demand Saudi Arabia KSA market 2024",
            f"{company} competitors Saudi Arabia Middle East landscape",
            f"{products} competitors pricing Saudi Arabia UAE 2024",
            f"digital marketing trends Saudi Arabia B2B enterprise 2024",
            f"GCC consumer behaviour technology adoption Saudi Arabia 2024",
            f"Saudi Arabia {products} industry growth market size 2024",
        ]
        return [q[:250] for q in raw]

    def run_tavily_search(self, queries: list[str]) -> str:
        snippets: list[str] = []
        for q in queries:
            try:
                res = self.tavily_client.search(q, search_depth="advanced", max_results=5)
                for r in res.get("results", []):
                    s = r.get("content", "")
                    if s: snippets.append(s[:500])
            except Exception as exc:
                snippets.append(f"[Search note: {exc}]")
        return "\n\n---\n\n".join(snippets) if snippets else (
            "No live results. Generating from parametric knowledge.")

    def synthesise_report(self, profile: CompanyProfile, context: str) -> str:
        system_prompt = (
            "You are a Principal Market Intelligence Analyst specialising in Saudi Arabia "
            "and the GCC under Vision 2030. You produce rigorous, evidence-based reports "
            "used by C-suites for high-stakes market entry and expansion decisions. "
            "Your analysis always compares the client against named competitors operating "
            "in Saudi Arabia and the broader Middle East."
        )
        user_prompt = (
            f"Company: {profile.company_name}\n"
            f"Products & Pricing: {profile.products_and_prices}\n"
            f"Brand Voice: {profile.brand_voice}\n"
            f"Marketing Budget: {profile.marketing_budget}\n\n"
            "Produce a comprehensive Saudi Arabia-First Market Research Report:\n\n"
            "## 1. Saudi Arabia Market Landscape\n"
            "   - Market size, growth rate, Vision 2030 alignment, key macro drivers.\n"
            "   - Regulatory environment (SFDA, ZATCA, CITC, Nitaqat) as relevant.\n\n"
            "## 2. Competitive Analysis — Saudi Arabia & Middle East\n"
            "   - Name and profile 3–5 direct competitors operating in KSA/ME.\n"
            "   - Their positioning, pricing, strengths, weaknesses, and market share.\n"
            "   - Clear white-space opportunities the client can capture.\n\n"
            "## 3. Target Audience Segments in KSA\n"
            "   - 2–3 segments: sector, company size, decision-maker roles.\n"
            "   - Cultural and psychographic traits specific to Saudi buyers.\n\n"
            "## 4. Saudi Consumer & Business Behaviour\n"
            "   - Communication preferences, trust signals, Wasta dynamics, "
            "     purchasing process, and Ramadan/seasonal patterns.\n\n"
            "## 5. Digital Channel Intelligence — KSA\n"
            "   - Platform usage (Snapchat, Instagram, LinkedIn, X/Twitter, TikTok, "
            "     WhatsApp Business, Google) among KSA B2B audiences.\n"
            "   - Content format preferences and peak engagement windows.\n\n"
            "## 6. Key Opportunities & Risks\n"
            "   - Top 3 opportunities with Vision 2030 alignment notes.\n"
            "   - Top 3 risks (regulatory, cultural, competitive) with impact ratings.\n\n"
            "## 7. Strategic Implications\n"
            "   - Prioritised entry/expansion recommendations for the Saudi market.\n\n"
            f"Retrieved Intelligence:\n{context}"
        )
        response = self.openai_client.chat.completions.create(
            model=self.CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.35,
            max_tokens=2200,
        )
        return response.choices[0].message.content.strip()

    def embed_text(self, text: str) -> list[float]:
        return self.openai_client.embeddings.create(
            model=self.EMBEDDING_MODEL, input=text
        ).data[0].embedding

    def persist_research(self, profile: CompanyProfile, report: str) -> str:
        doc_id    = f"research_{uuid.uuid4().hex}"
        embedding = self.embed_text(report)
        self.collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[report],
            metadatas=[{
                "company_name":        profile.company_name,
                "brand_voice":         profile.brand_voice,
                "products_and_prices": profile.products_and_prices,
                "marketing_budget":    profile.marketing_budget,
                "agent":               "research",
            }],
        )
        return doc_id