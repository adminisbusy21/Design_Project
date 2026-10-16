import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker 
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:nandu@localhost:5432/damage_inspection",
)

engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)