import base64
import os
from io import BytesIO

from PIL import Image
import torch
from fastapi import FastAPI
from pydantic import BaseModel

from KneesOA.config import transform
from KneesOA.model.utils import load_network

app = FastAPI()


RELEASE_VERSION = os.getenv("RELEASE_VERSION", None)
assert RELEASE_VERSION is not None, "Release version is not specified in .env file"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_weights = os.path.join('../../model_release/', RELEASE_VERSION, "model.pkl")
net = load_network(model_weights)
net.eval()
net.to(device)


def get_prediction(image):
    image = transform(image).unsqueeze(0).to(device)
    image.to(device)
    with torch.no_grad():
        output = int(torch.argmax(net(image), dim=1)[0].cpu().numpy())
        return output


class Request(BaseModel):
    base64str: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/classify")
async def classify(image: Request):
    im_bytes = base64.b64decode(image.base64str)
    im_file = BytesIO(im_bytes)
    img = Image.open(im_file)
    result = get_prediction(img)
    return {"KL_GRADE": result}
