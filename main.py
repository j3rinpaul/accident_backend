from fastapi import FastAPI, Query
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, TIMESTAMP, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from keras.models import load_model
import numpy as np
import uuid
import pytz
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
model = load_model('accident.h5')
SQLALCHEMY_DATABASE_URL = "sqlite:///./accident.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:63612",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Location(Base):
    __tablename__ = "location"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    time = Column(TIMESTAMP(timezone=True), nullable=True, server_default=text('CURRENT_TIMESTAMP'))
    latitude = Column(String)
    longitude = Column(String)

class Rash(Base):
    __tablename__ = "rash"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    time = Column(TIMESTAMP(timezone=True), nullable=True, server_default=text('CURRENT_TIMESTAMP'))
    x_acc = Column(String)
    y_acc = Column(String)
    z_acc = Column(String)
    x_tilt = Column(String)
    y_tilt = Column(String)
    z_tilt = Column(String)
    label = Column(String)

# Drop existing rash table and recreate it with the updated schema
# Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

class DeviceData(BaseModel):
    param1: float
    param2: float
    param3: float
    param4: float
    param5: float
    param6: float

class LocationData(BaseModel):
    latitude:float
    longitude:float

#ml model data updation prediction and storing it to a db
@app.put('/ml_model')
def ml_model(item: DeviceData):
    p1, p2, p3, p4, p5, p6 = item.param1, item.param2, item.param3, item.param4, item.param5, item.param6
    result = model.predict(np.array([[p1, p2, p3, p4, p5, p6]]))
    max_index = np.argmax(result)
    if max_index == 0:
        label = "Normal"
    elif max_index == 1:
        label = "Rash"
    else:
        label = "Accident"
    
    # Convert UTC time to Indian time
    tz = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(tz)
        
    db = SessionLocal()
    db_message = Rash(time=current_time, x_acc=p1, y_acc=p2, z_acc=p3, x_tilt=p4, y_tilt=p5, z_tilt=p6, label=label)
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return db_message


#to update the location data
@app.put('/location')
def location_data(location: LocationData):
    lat = str(location.latitude)
    longi = str(location.longitude)
    loc_db = SessionLocal()
    locdb_message = Location(latitude=lat, longitude=longi)
    loc_db.add(locdb_message)
    loc_db.commit()
    loc_db.refresh(locdb_message)
    return locdb_message


#to get the last location details
@app.get("/last_location")
def get_last_location():
    with SessionLocal() as db:
        last_location = db.query(Location).order_by(Location.time.desc()).first()
        if last_location:
            return {
                "timestamp": last_location.time,
                "latitude": last_location.latitude,
                "longitude": last_location.longitude
            }
        else:
            return {"message": "No location data available"}


# @app.get("/last_data")
# def get_last_data():
#     db = SessionLocal()
#     last_data = db.query(Rash).order_by(Rash.id.desc()).first()
#     return {
#         "id": last_data.id,
#         "timestamp": last_data.time,
#         "x_acc": last_data.x_acc,
#         "y_acc": last_data.y_acc,
#         "z_acc": last_data.z_acc,
#         "x_tilt": last_data.x_tilt,
#         "y_tilt": last_data.y_tilt,
#         "z_tilt": last_data.z_tilt,
#         "label": last_data.label
#     }

#get the  graph details of a particular date
@app.get("/requests_on_date")
def get_requests_on_date(date: datetime = Query(...)):
    start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = date.replace(hour=23, minute=59, second=59, microsecond=999999)
    db = SessionLocal()
    requests = db.query(Rash).filter(Rash.time >= start_date, Rash.time <= end_date).all()
    return [{"timestamp": req.time, "label": req.label} for req in requests]

#get the average details of that date of a month
@app.get("/average_label_on_date")
def get_average_label_on_date(date: datetime = Query(...)):
    start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = date.replace(hour=23, minute=59, second=59, microsecond=999999)
    db = SessionLocal()
    labels_on_date = db.query(Rash.label).filter(Rash.time >= start_date, Rash.time <= end_date).all()
    labels = [label[0] for label in labels_on_date]
    label_counts = {label: labels.count(label) for label in set(labels)}
    return {
        "date": date.date(),
        "average_label": {label: count / len(labels) for label, count in label_counts.items()}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
