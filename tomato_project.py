from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model("C:/Users/ARADHANA/Documents/tomato_model_corr.h5")
SUB_MODEL = tf.keras.models.load_model("C:/Users/ARADHANA/Documents/sub_model.h5")

CLASS_NAMES = ["Tomato__healthy", "Tomato__Unhealthy"]
SUB_CLASS_NAMES = ["Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
                   "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
                   "Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"]


@app.get("/ping")
async def ping():
    return "Hello , I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)

):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    print(predictions)

    pred = SUB_MODEL.predict(img_batch)
    print("defected class ")
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    if predicted_class == 'Tomato__Unhealthy':
        sub_class = SUB_CLASS_NAMES[np.argmax(pred[0])]
    else:
        sub_class = 'null'

    return {
        'Predicted Class ': predicted_class,
        'Predicted Disease': sub_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8001)
