import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request
from tensorflow.keras.applications.resnet_v2 import preprocess_input

app = Flask(__name__)

model = load_model("evgg.h5")


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == "POST":
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        f.save(filepath)
        img = image.load_img(filepath, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)

        pred = np.argmax(model.predict(img_data), axis=1)
        index = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
        text = str(index[pred[0]])
        return render_template("output.html", prediction=text)


if __name__ == '__main__':
    app.run()
