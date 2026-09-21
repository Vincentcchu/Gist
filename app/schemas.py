from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ReviewOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    venue_id: int
    external_id: str | None
    raw_text: str
    source: str
    scraped_at: datetime
    status: str


class AspectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    aspect_text: str
    sentiment: str
    span: str | None
    model_version: str


class ReviewAspectsOut(BaseModel):
    review_id: int
    raw_text: str
    aspects: list[AspectOut]


class HealthOut(BaseModel):
    status: str
