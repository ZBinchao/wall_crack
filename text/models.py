from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.sql import func
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default="inspector")  # admin / inspector
    real_name = Column(String(50), default="")


class WorkOrder(Base):
    __tablename__ = "work_orders"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    location = Column(String(500), default="")
    status = Column(String(20), default="pending")  # pending / in_progress / completed
    inspector_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    description = Column(Text, default="")
    created_at = Column(DateTime, server_default=func.now())
    completed_at = Column(DateTime, nullable=True)


class Anomaly(Base):
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True, index=True)
    work_order_id = Column(Integer, ForeignKey("work_orders.id"), nullable=False)
    image_path = Column(String(500), default="")
    annotated_path = Column(String(500), default="")
    confidence = Column(Float, default=0.0)
    severity = Column(String(10), default="low")  # low / medium / high
    area_pixels = Column(Integer, default=0)
    length_pixels = Column(Float, default=0.0)
    created_at = Column(DateTime, server_default=func.now())
