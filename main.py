import csv
import io
from typing import Counter
from fastapi import Depends, FastAPI,HTTPException, Response, WebSocket
from pydantic import BaseModel
from requests import Session
from sqlalchemy import  Column, String, TIMESTAMP, extract, func, select, text
from sqlalchemy.ext.declarative import declarative_base
from datetime import date, datetime, timedelta
from keras.models import load_model
import numpy as np
import uuid
import pytz
# from sqlalchemy.exc import SQLAlchemyError
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
    speed: float 

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
    phone_data = [{"phone": phones.number, "name": phones.name} for phones in phone]

    return phone_data
    

@app.delete("/phone_number/{phone_number_id}")
async def delete_phone_number(phone_number_id: str, db: AsyncSession = Depends(get_db)):
    try:
        # Query the database to find the phone number
        phone_number = await db.execute(select(Phone_number).where(Phone_number.number == phone_number_id))
        phone_number_obj = phone_number.scalar_one_or_none()

        if phone_number_obj:
            # If the phone number exists, delete it
            await db.delete(phone_number_obj)
            await db.commit()
            return {"message": "Phone number deleted successfully"}
        else:
            # If the phone number doesn't exist, raise a 404 error
            raise HTTPException(status_code=404, detail="Phone number not found")
    except Exception as e:
        # Handle any exceptions that occur during the deletion process
        return {"error": str(e)}
    


#ml model data updation prediction and storing it to a db
@app.post('/ml_model')
async def ml_model(item: DeviceData,db: AsyncSession = Depends(get_db)):
    p1, p2, p3, p4, p5, p6,speed = item.param1, item.param2, item.param3, item.param4, item.param5, item.param6,item.speed
    
    speed = float(speed) if speed is not None else 0

    result = model.predict(np.array([[p1, p2, p3, p4, p5, p6,speed]]))
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
    data = await db.execute(select(Rash))
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
    # Execute a query to fetch only the label and time of the rash data for the target date asynchronously
    query = select(Rash.label, func.strftime('%H', Rash.time)).where(
        Rash.time >= target_date,
        Rash.time < target_date + timedelta(days=1)
    )
    result = await db.execute(query)
    rash_data = result.fetchall()  # Use fetchall() to get raw tuples of results
    
    # Check if any data is fetched
    if not rash_data:
        raise HTTPException(status_code=404, detail="No rash data available for the specified date")
    
    hourly_data = {}
    for label, hour in rash_data:
        if hour not in hourly_data:
            hourly_data[hour] = []
        hourly_data[hour].append(label)

    most_repeated_labels = {}
    for hour, labels in hourly_data.items():
        label_counts = Counter(labels)
        most_common_label = label_counts.most_common(1)[0][0]
        most_repeated_labels[hour] = most_common_label

    return most_repeated_labels

@app.get('/pattern/{date}')
async def get_pattern(date: date, db: AsyncSession = Depends(get_db)):
    query = select(Rash.label, func.strftime('%H', Rash.time)).where(
        Rash.time >= date,
        Rash.time < date + timedelta(days=1)
    )
    result = await db.execute(query)
    rash_data = result.fetchall()

    percent = {}

    if not rash_data:
        raise HTTPException(status_code=404, detail="No rash data available for the specified date")
    hourly_data = {}
    for label, hour in rash_data:
        if hour not in hourly_data:
            hourly_data[hour] = []
        hourly_data[hour].append(label)

    most_repeated_labels = {}
    for hour, labels in hourly_data.items():
        label_counts = Counter(labels)
        most_common_label = label_counts.most_common(1)[0][0]
        most_repeated_labels[hour] = most_common_label

    for hour,label in most_repeated_labels.items():
        if label == "Rash":
            percent["rash"] = percent.get("rash", 0) + 1
        elif label == "Accident":
            percent["accident"] = percent.get("accident", 0) + 1
        else:
            percent["normal"] = percent.get("normal", 0) + 1
    
    for key in percent:
        percent[key] = (percent[key]/len(most_repeated_labels))*100

    
    return {"rash_percent":percent['rash']}




@app.get("/average-speed/{year}/{month}")
async def get_average_speed(year: int, month: int,db:AsyncSession = Depends(get_db)):
  
    start_date = datetime(year, month, 1)
    end_date = start_date.replace(month=start_date.month + 1) if start_date.month < 12 else start_date.replace(year=start_date.year + 1, month=1)

    # Calculate the average speed for the given month
    stmt = select(func.avg(Rash.speed)).where(
        extract('year', Rash.time) == year,
        extract('month', Rash.time) == month
    )
    result = await db.execute(stmt)
    avg_speed = result.scalar_one_or_none()

    if avg_speed is None:
        raise HTTPException(status_code=404, detail="No speed data available for the given month")

    return {"average_speed": avg_speed}


@app.get('/speed/{date}')
async def get_speed(date: date, db: AsyncSession = Depends(get_db)):
    query = select(Rash.speed,Rash.time).where(
        Rash.time >= date,
        Rash.time < date + timedelta(days=1)
    )
    result = await db.execute(query)
    speed_data = result.scalars().all()


    if not speed_data:
        raise HTTPException(status_code=404, detail="No speed data available for the specified date")
    avg_speed = 0
    for speed in speed_data:
        avg_speed += float(speed)
    return {"speed":avg_speed/len(speed_data)}



##############################
clients = []
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Open connection
    await websocket.accept()
    clients.append(websocket)
    try:
        # Listen for messages
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            # Broadcast message to all connected clients
            for client in clients:
                await client.send_text(data)
    except Exception as e:
        print(e)
    finally:
        # Close connection
        clients.remove(websocket)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
