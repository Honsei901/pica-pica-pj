import numpy as np

from PIL import Image
from io import BytesIO
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile, Form
from tensorflow.keras.models import load_model

app = FastAPI()


@app.post("/image/estimate")
async def estimate_image(file: UploadFile = Form(...)):
    classes = ["Car", "Motorcycle"]
    image_size = 224

    contents = await file.read()
    image = Image.open(BytesIO(contents))
    image = image.convert("RGB")
    image = image.resize((image_size, image_size))
    data = np.asarray(image) / 255.0

    X = []
    X.append(data)

    X = np.array(X)

    model = load_model("../vgg16_transfer.h5")
    result = model.predict([X])[0]
    predicted = result.argmax()
    parcentage = int(result[predicted] * 100)

    return {"vehicle_type": classes[predicted], "parcentage": parcentage}


app.mount("/", StaticFiles(directory="public", html=True), name="public")
