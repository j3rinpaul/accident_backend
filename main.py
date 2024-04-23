import csv
import io
from fastapi import Depends, FastAPI,HTTPException, Response
from pydantic import BaseModel
from requests import Session
from sqlalchemy import  Column, String, TIMESTAMP, select, text
from sqlalchemy.ext.declarative import declarative_base
from datetime import date, datetime, timedelta
from keras.models import load_model
import numpy as np
import uuid
import pytz
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import create_async_engine,async_sessionmaker,AsyncSession


app = FastAPI()
model = load_model('accident.h5')
SQLALCHEMY_DATABASE_URL = "sqlite+aiosqlite:///./accident.db"
engine = create_async_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:63612",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def get_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    db = SessionLocal()
    try:
        yield db
    finally:
        await db.close()

class Location(Base):
    __tablename__ = "location"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    time = Column(TIMESTAMP(timezone=True), nullable=True, server_default=text('CURRENT_TIMESTAMP'))
    latitude = Column(String)
    longitude = Column(String)




class Phone_number(Base):
    __tablename__ = "phone"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    number = Column(String)  # Rename the column to "phoneNumber"
    name = Column(String)

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
    speed = Column(String)
    label = Column(String)

    
class Sample(Base):
    __tablename__ = "sample"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    time = Column(TIMESTAMP(timezone=True), nullable=True, server_default=text('CURRENT_TIMESTAMP'))
    x_acc = Column(String)
    y_acc = Column(String)
    z_acc = Column(String)
    x_tilt = Column(String)
    y_tilt = Column(String)
    z_tilt = Column(String)
    speed = Column(String)
    label = Column(String)

# Drop existing rash table and recreate it with the updated schema
# Base.metadata.drop_all(bind=engine)
# Base.metadata.create_all(bind=engine)

class DeviceData(BaseModel):
    param1: float
    param2: float
    param3: float
    param4: float
    param5: float
    param6: float
    speed: float | None = None

class LocationData(BaseModel):
    latitude:float
    longitude:float

class PhoneData(BaseModel):
    phoneNumber:int
    name: str



#toput the phone number
@app.post('/phone')
async def getPhone(item:PhoneData,db:AsyncSession = Depends(get_db)):
    phone = item.phoneNumber
    name = item.name
    
    db_message = Phone_number(number = phone,name=name)
    db.add(db_message)
    await db.commit()
    await db.refresh(db_message)
    return db_message

#to get all phone numbers

@app.get("/phone_number")
async def get_phone_number(db: AsyncSession = Depends(get_db)):
    phone_numbers = await db.execute(select(Phone_number))
    phone = phone_numbers.scalars().all()
    phone_data = [{"phone": phones.number, "name": phones.name, "id": phones.id} for phones in phone]

    return phone_data
    


@app.delete("/phone_number/{phone_number_id}")
async def delete_phone_number(phone_number_id: str,db: AsyncSession = Depends(get_db)):
    
        # phone_number = db.query(Phone_number).filter(Phone_number.number == phone_number_id).first()
        phone_number  = await db.execute(select(Phone_number).where(Phone_number.number == phone_number_id))
        if phone_number:
            db.delete(phone_number)
            db.commit()
            return {"message": "Phone number deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Phone number not found")

#ml model data updation prediction and storing it to a db
@app.post('/ml_model')
async def ml_model(item: DeviceData,db: AsyncSession = Depends(get_db)):
    p1, p2, p3, p4, p5, p6 = item.param1, item.param2, item.param3, item.param4, item.param5, item.param6
    if item.speed:
        speed = item.speed
    else:
        speed = None
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

    if speed:
        if speed > 60:
            label = "Rash"
        
    # db = SessionLocal()
    db_message = Rash(time=current_time, x_acc=p1, y_acc=p2, z_acc=p3, x_tilt=p4, y_tilt=p5, z_tilt=p6, speed = speed,label=label)
    db.add(db_message)
    await db.commit()
    await db.refresh(db_message)
    return db_message


#to update the location data
@app.post('/location')
async def location_data(location: LocationData, db: Session = Depends(get_db)):
    lat = str(location.latitude)
    longi = str(location.longitude)
    locdb_message = Location(latitude=lat, longitude=longi)
    db.add(locdb_message)
    await db.commit()
    await db.refresh(locdb_message)
    return locdb_message


#to get the last location details
@app.get("/last_location")
async def get_last_location(db: AsyncSession = Depends(get_db)):
    last_location = await db.execute(select(Location).order_by(Location.time.desc()).limit(1))
    last_location = last_location.scalars().first()
    if last_location:
        return {
            "timestamp": last_location.time,
            "latitude": last_location.latitude,
            "longitude": last_location.longitude
        }
    else:
        return {"message": "No location data available"}


@app.post('/sample')
async def sample(item: DeviceData,db: AsyncSession = Depends(get_db)):
    p1, p2, p3, p4, p5, p6,speed = item.param1, item.param2, item.param3, item.param4, item.param5, item.param6,item.speed
    db_message = Sample(x_acc=p1, y_acc=p2, z_acc=p3, x_tilt=p4, y_tilt=p5, z_tilt=p6, speed = speed)
    db.add(db_message)
    await db.commit()
    await db.refresh(db_message)
    return db_message

@app.get("/download_csv")
async def download_csv(db: AsyncSession = Depends(get_db)):
    # Fetch data from the database
    data = await db.execute(select(Sample))
    data = data.scalars().all()

    # Prepare CSV data
    output = io.StringIO()
    fieldnames = ["id", "time", "x_acc", "y_acc", "z_acc", "x_tilt", "y_tilt", "z_tilt", "speed", "label"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in data:
        # Filter out _sa_instance_state
        filtered_row = {key: getattr(row, key) for key in fieldnames if key != '_sa_instance_state'}
        writer.writerow(filtered_row)

    # Prepare response
    output.seek(0)
    response = Response(content=output.getvalue(), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=data.csv"
    return response

@app.get("/rash_data/{target_date}")
async def get_rash_data(target_date: date, db: AsyncSession = Depends(get_db)):
    # Execute a query to fetch rash data for the target date asynchronously
    query = select(Rash).where(Rash.time >= target_date, Rash.time < target_date + timedelta(days=1))
    rash_data = await db.execute(query)
    rash_data = rash_data.scalars().all()
    
    # Check if any data is fetched
    if not rash_data:
        raise HTTPException(status_code=404, detail="No rash data available for the specified date")
    
    # Return the fetched data
    return rash_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
