from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL from environment variable
DATABASE_URL = os.getenv('DATABASE_URL')

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create declarative base
Base = declarative_base()

# Define Video model
class Video(Base):
    __tablename__ = 'videos'

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=False, index=True)
    transcript = Column(Text)
    summary = Column(Text, nullable=True)

# Create all tables
Base.metadata.create_all(bind=engine)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()