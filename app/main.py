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


@app.get("/hotels/{hotel_id}/reviews", response_model=list[schemas.ReviewOut])
def get_hotel_reviews(hotel_id: int, session: Session = Depends(db.get_db)):
    hotel = session.get(models.Hotel, hotel_id)
    if hotel is None:
        raise HTTPException(status_code=404, detail="Hotel not found")

    reviews = session.execute(
        select(models.Review).where(models.Review.hotel_id == hotel_id)
    ).scalars().all()
    return reviews


@app.get("/hotels/{hotel_id}/aspects", response_model=list[schemas.ReviewAspectsOut])
def get_hotel_aspects(hotel_id: int, session: Session = Depends(db.get_db)):
    hotel = session.get(models.Hotel, hotel_id)
    if hotel is None:
        raise HTTPException(status_code=404, detail="Hotel not found")

    reviews = session.execute(
        select(models.Review).where(models.Review.hotel_id == hotel_id)
    ).scalars().all()
    return [
        schemas.ReviewAspectsOut(review_id=r.id, raw_text=r.raw_text, aspects=r.aspects)
        for r in reviews
    ]
