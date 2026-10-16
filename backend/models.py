from sqlalchemy import Boolean, Column, DateTime, Integer, REAL, String, Text
from sqlalchemy.sql import func

from backend.database import Base


class InspectionResult(Base):
    __tablename__ = "inspection_results"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    image_format = Column(String(20), nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    size_bytes = Column(Integer, nullable=False)
    prediction = Column(String(20), nullable=False)
    confidence = Column(REAL, nullable=False)
    priority = Column(String(20), nullable=False)
    priority_reason = Column(Text, nullable=False)
    review_required = Column(Boolean, nullable=False, default=False)
    review_status = Column(String(20), nullable=False, default="PENDING")
    review_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.current_timestamp())