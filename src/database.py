from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from dotenv import load_dotenv
from datetime import datetime
import os

load_dotenv()

#CONNECTION
MYSQL_HOST     = os.getenv("MYSQL_HOST")
MYSQL_PORT     = os.getenv("MYSQL_PORT")
MYSQL_USER     = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

#Tables

class Product(Base):
    __tablename__ = "products"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    sku        = Column(String(50), unique=True, nullable=False)
    name       = Column(String(200), nullable=False)
    category   = Column(String(100))
    unit_price = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class Supplier(Base):
    __tablename__ = "suppliers"
    id               = Column(Integer, primary_key=True, autoincrement=True)
    name             = Column(String(200), nullable=False)
    country          = Column(String(100))
    city             = Column(String(100))
    lead_time_days   = Column(Integer)
    reliability_score= Column(Float)
    contact_email    = Column(String(200))
    created_at       = Column(DateTime, default=datetime.utcnow)

class Warehouse(Base):
    __tablename__ = "warehouses"
    id       = Column(Integer, primary_key=True, autoincrement=True)
    name     = Column(String(200), nullable=False)
    city     = Column(String(100))
    lat      = Column(Float)
    lon      = Column(Float)
    capacity = Column(Integer)

class Inventory(Base):
    __tablename__ = "inventory"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    product_id   = Column(Integer, ForeignKey("products.id"))
    warehouse_id = Column(Integer, ForeignKey("warehouses.id"))
    quantity     = Column(Integer)
    last_updated = Column(DateTime, default=datetime.utcnow)

class Order(Base):
    __tablename__ = "orders"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    product_id    = Column(Integer, ForeignKey("products.id"))
    supplier_id   = Column(Integer, ForeignKey("suppliers.id"))
    quantity      = Column(Integer)
    order_date    = Column(DateTime)
    delivery_date = Column(DateTime)
    status        = Column(String(50))
    late_delivery = Column(Integer, default=0)

class Anomaly(Base):
    __tablename__ = "anomalies"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    product_id   = Column(Integer, ForeignKey("products.id"))
    anomaly_type = Column(String(100))
    severity     = Column(String(20))
    detected_at  = Column(DateTime, default=datetime.utcnow)
    description  = Column(Text)
    resolved     = Column(Integer, default=0)

#Create all tables
def init_db():
    Base.metadata.create_all(bind=engine)
    print("All tables created successfully!")

#Get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    init_db()