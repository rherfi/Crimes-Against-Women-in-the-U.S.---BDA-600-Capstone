"""
Pydantic schemas for the API contract.

We keep these strict and explicit because:
- The frontend depends on stable fields
- The backend can include structured debug output for development
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MapEmbedPayload(BaseModel):
    """
    Optional ArcGIS Dashboard embed. The chat UI renders `embed_url` in an iframe when show=True.

    Note: Most dashboards cannot auto-zoom to arbitrary states via URL unless the author enabled
    Dashboard URL parameters; `states` / `metric_label` are hints for the user and future deep links.
    """

    show: bool = False
    title: str = ""
    embed_url: str = ""
    open_url: str = ""
    caption: str = ""
    states: List[str] = Field(default_factory=list)
    metric: Optional[str] = None
    metric_label: Optional[str] = None


class AnswerPayload(BaseModel):
    direct_answer: str
    evidence: List[str] = Field(default_factory=list)
    interpretation: str = ""
    caveats: List[str] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    map_embed: Optional[MapEmbedPayload] = None


class DebugPayload(BaseModel):
    intent: str
    tools_used: List[Dict[str, Any]] = Field(default_factory=list)
    docs_retrieved: List[Dict[str, Any]] = Field(default_factory=list)
    llm: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: AnswerPayload
    debug: DebugPayload


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's question")
    # For V1 we keep history optional. We can upgrade later to use it in retrieval.
    history: Optional[List[Dict[str, str]]] = None

