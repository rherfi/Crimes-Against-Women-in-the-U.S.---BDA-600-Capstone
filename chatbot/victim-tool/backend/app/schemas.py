"""
Pydantic schemas for the Victim Tool Bot API contract.

Important:
- This is a sensitive-use chatbot prototype.
- We keep the response fields explicit and stable so the UI can render them safely.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


Intent = Literal["crisis", "resource_lookup", "vawa_info", "support_guidance", "unknown"]


class ResourceOut(BaseModel):
    name: str
    resource_type: str
    phone: str = ""
    website: str = ""
    location: str = ""
    notes: str = ""
    source: str = ""


class ResponsePayload(BaseModel):
    support_message: str
    resources: List[ResourceOut] = Field(default_factory=list)
    practical_next_steps: List[str] = Field(default_factory=list)
    caveats: List[str] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)


class DebugPayload(BaseModel):
    intent: Intent
    location_detected: str = ""
    resource_filters: List[str] = Field(default_factory=list)
    docs_retrieved: List[Dict[str, Any]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    response: ResponsePayload
    debug: DebugPayload


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's message")
    # Optional history for future upgrades; V1 does not heavily rely on it.
    history: Optional[List[Dict[str, str]]] = None

