from cgi import FieldStorage
from flask import Flask, render_template, request
from getresult import predict_image
from PIL import Image
from model import model
import torch

import base64
from io import BytesIO

app = Flask(__name__)

@app.route('/')
def index():
    torch.cuda.empty_cache()
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    
    input_image = request.files['image']
    img = Image.open(input_image)
    answer, confidence = predict_image(img,model)

    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return render_template('result.html', answer=answer, confidence=confidence, image=img_str)

if __name__ == "__main__":
    app.run(debug=True)