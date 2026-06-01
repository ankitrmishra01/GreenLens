from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base
from datetime import datetime
import enum

class ActivityType(str, enum.Enum):
    TRANSPORT = "transport"
    FOOD = "food"
    ELECTRICITY = "electricity"
    PURCHASES = "purchases"
    WASTE = "waste"

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String, nullable=True)
    campus = Column(String, default="Parul University")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    activities = relationship("Activity", back_populates="user", cascade="all, delete-orphan")
    stats = relationship("UserStats", back_populates="user", uselist=False, cascade="all, delete-orphan")
    nudges = relationship("Nudge", back_populates="user", cascade="all, delete-orphan")

class Activity(Base):
    __tablename__ = "activities"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    activity_type = Column(String, index=True)
    value = Column(Float)
    unit = Column(String)
    co2_kg = Column(Float)
    description = Column(String, nullable=True)
    image_hash = Column(String, nullable=True)
    receipt_id = Column(String, index=True, nullable=True)
    sdg_goal = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", back_populates="activities")

class UserStats(Base):
    __tablename__ = "user_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    total_co2_kg = Column(Float, default=0.0)
    weekly_co2_kg = Column(Float, default=0.0)
    streak_days = Column(Integer, default=0)
    xp_points = Column(Integer, default=0)
    rank = Column(Integer, default=0)
    trees_saved_equivalent = Column(Integer, default=0)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    user = relationship("User", back_populates="stats")

class Leaderboard(Base):
    __tablename__ = "leaderboard"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    rank = Column(Integer, index=True)
    xp_points = Column(Integer)
    weekly_co2_reduction = Column(Float)
    streak = Column(Integer)
    username = Column(String)
    campus = Column(String)
    badge = Column(String, nullable=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class Nudge(Base):
    __tablename__ = "nudges"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    content = Column(Text)
    category = Column(String)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", back_populates="nudges")
