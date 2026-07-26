import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from models import Base

# Defaults to a local SQLite file so the app runs with zero setup. Point
# DATABASE_URL at a Postgres instance (Railway/Render) later - no code
# changes needed since models.py avoids Postgres-only column types.
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./local.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    # No Alembic yet - schema changes during development mean deleting
    # local.db and reseeding.
    Base.metadata.create_all(bind=engine)
