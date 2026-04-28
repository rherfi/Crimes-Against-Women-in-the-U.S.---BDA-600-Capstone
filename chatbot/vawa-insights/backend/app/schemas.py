"""
Pydantic schemas for the API contract.

We keep these strict and explicit because:
- The frontend depends on stable fields
- The backend can include structured debug output for development
"""

from typing import Any, Dict, List, Literal, Optional

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


# ---------------------------------------------------------------------------
# Victim resources search API
# ---------------------------------------------------------------------------


ResourceCategory = Literal[
    "domestic_violence",
    "sexual_assault",
    "mental_health",
    "legal_aid",
    "housing_shelter",
    "hotline",
    "other",
]


class ResourceSearchRequest(BaseModel):
    """
    Search for victim resources near a location.

    - Provide either (latitude, longitude) OR a free-text location (e.g., "San Diego, CA").
    - Results are returned sorted by distance when coordinates are available.
    """

    query: str = Field("", description="Optional free-text query (e.g., 'shelter', 'therapy').")
    location: str = Field("", description="Free-text location to geocode (e.g., 'Sacramento, CA').")
    latitude: Optional[float] = Field(None, description="Latitude in decimal degrees.")
    longitude: Optional[float] = Field(None, description="Longitude in decimal degrees.")
    radius_miles: float = Field(25.0, ge=1.0, le=250.0, description="Search radius in miles.")
    limit: int = Field(8, ge=1, le=25, description="Max number of results.")
    categories: List[ResourceCategory] = Field(default_factory=list, description="Optional category filter.")


class ResourceResult(BaseModel):
    resource_id: str
    name: str
    category: str = ""
    subcategory: str = ""
    services: str = ""
    address: str = ""
    city: str = ""
    state: str = ""
    postal_code: str = ""
    phone: str = ""
    website: str = ""
    distance_miles: Optional[float] = None
    notes: str = ""


class ResourceSearchResponse(BaseModel):
    ok: bool
    error: str = ""
    resolved_location: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    radius_miles: float = 0.0
    results: List[ResourceResult] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)

