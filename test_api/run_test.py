import base64
from io import BytesIO
import requests

from PIL import Image

img = Image.open('test_api/knee_4_KL.png')
im_file = BytesIO()
img.save(im_file, format="png")
im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
im_b64 = base64.b64encode(im_bytes).decode()
request = requests.post("http://localhost:8000/classify", json={"base64str": im_b64})
print(request.json())  # has to be 4
