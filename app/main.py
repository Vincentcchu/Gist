from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

import db
import models
import schemas
from health import router as health_router

app = FastAPI(title="Review ABSA MVP")
app.include_router(health_router)


@app.on_event("startup")
def on_startup():
    db.init_db()


@app.get("/venues/{venue_id}/reviews", response_model=list[schemas.ReviewOut])
def get_venue_reviews(venue_id: int, session: Session = Depends(db.get_db)):
    venue = session.get(models.Venue, venue_id)
    if venue is None:
        raise HTTPException(status_code=404, detail="Venue not found")

    reviews = session.execute(
        select(models.Review).where(models.Review.venue_id == venue_id)
    ).scalars().all()
    return reviews


@app.get("/venues/{venue_id}/aspects", response_model=list[schemas.ReviewAspectsOut])
def get_venue_aspects(venue_id: int, session: Session = Depends(db.get_db)):
    venue = session.get(models.Venue, venue_id)
    if venue is None:
        raise HTTPException(status_code=404, detail="Venue not found")

    reviews = session.execute(
        select(models.Review).where(models.Review.venue_id == venue_id)
    ).scalars().all()
    return [
        schemas.ReviewAspectsOut(review_id=r.id, raw_text=r.raw_text, aspects=r.aspects)
        for r in reviews
    ]
