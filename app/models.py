from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Venue(Base):
    __tablename__ = "venues"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    location: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    # Scrape checkpoint state - separate from Review.status (which tracks the
    # ABSA labeling pipeline). pending/in_progress/done/blocked_captcha.
    scrape_status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    last_review_external_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    last_scraped_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # Site's own advertised review count ("食評 (N)"), re-read each run since
    # it drifts as real users post - the actual completion signal, since a
    # single run finding 0 new reviews isn't reliable proof there are no more.
    total_reviews_expected: Mapped[int | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    reviews: Mapped[list["Review"]] = relationship(
        back_populates="venue", cascade="all, delete-orphan"
    )


class Review(Base):
    __tablename__ = "reviews"
    __table_args__ = (
        Index("ix_reviews_venue_id", "venue_id"),
        Index("ix_reviews_status", "status"),
        UniqueConstraint("venue_id", "external_id", name="uq_reviews_venue_external_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    venue_id: Mapped[int] = mapped_column(ForeignKey("venues.id"), nullable=False)
    # Source site's own review id (e.g. OpenRice's "6502606"). Nullable since
    # manual-seed reviews have no source id; unique per venue so a resumed
    # scrape run never inserts the same review twice.
    external_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    scraped_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    venue: Mapped["Venue"] = relationship(back_populates="reviews")
    aspects: Mapped[list["AspectExtraction"]] = relationship(
        back_populates="review", cascade="all, delete-orphan"
    )


class AspectExtraction(Base):
    __tablename__ = "aspect_extractions"
    __table_args__ = (Index("ix_aspect_extractions_review_id", "review_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    review_id: Mapped[int] = mapped_column(ForeignKey("reviews.id"), nullable=False)
    aspect_text: Mapped[str] = mapped_column(String(500), nullable=False)
    sentiment: Mapped[str] = mapped_column(String(20), nullable=False)
    span: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    review: Mapped["Review"] = relationship(back_populates="aspects")
