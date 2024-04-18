from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from typing import List
import sqlite3
from keras.models import load_model
import numpy as np

app = FastAPI()
model = load_model('accident.h5')

# Define a Pydantic model for the data received from the device
class DeviceData(BaseModel):
    param1: float
    param2: float
    param3: float
    param4: float
    param5: float
    param6: float

# Connect to SQLite database
conn = sqlite3.connect('data.db')
c = conn.cursor()

@app.put('/ml_model')
def ml_model(item: DeviceData):
    p1 = item.param1
    p2 = item.param2
    p3 = item.param3
    p4 = item.param4
    p5 = item.param5
    p6 = item.param6

    result = model.predict(np.array([[p1, p2, p3, p4, p5, p6]]))

    max_index = np.argmax(result)
    if max_index == 0:
        label = "Normal"
    elif max_index == 1:
        label = "Rash"
    else:
        label = "Accident"
        
    return label

    



# Endpoint to retrieve last inserted location from the database
@app.get("/last_location")
def get_last_location():
    c.execute("SELECT * FROM location_data ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    if row:
        return {"latitude": row[1], "longitude": row[2]}
    else:
        return {"message": "No location data available"}

# Define the ML model and other necessary imports
# Make sure to import and load your trained model before defining the endpoint to receive data

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
