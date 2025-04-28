from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('crop_disease_model.h5')


disease_info = {
    'Pepper__bell___Bacterial_spot': {
        'name': 'Pepper Bell Bacterial Spot',
        'cause': 'Caused by the bacterium Xanthomonas campestris pv. vesicatoria.',
        'cure': 'Use copper-based bactericides and resistant pepper varieties. Avoid overhead watering.'
    },
    'Pepper__bell___healthy': {
        'name': 'Healthy Pepper Bell',
        'cause': 'No disease detected.',
        'cure': 'Continue good agricultural practices to maintain plant health.'
    },
    'Potato___Early_blight': {
        'name': 'Potato Early Blight',
        'cause': 'Caused by the fungus Alternaria solani.',
        'cure': 'Apply fungicides like chlorothalonil or mancozeb, rotate crops, and remove infected plant debris.'
    },
    'Potato___healthy': {
        'name': 'Healthy Potato',
        'cause': 'No disease detected.',
        'cure': 'Maintain regular monitoring and preventive care for healthy growth.'
    },
    'Potato___Late_blight': {
        'name': 'Potato Late Blight',
        'cause': 'Caused by the oomycete Phytophthora infestans.',
        'cure': 'Use fungicides such as metalaxyl, ensure good field drainage, and destroy infected plants.'
    },
    'Tomato__Target_Spot': {
        'name': 'Tomato Target Spot',
        'cause': 'Caused by the fungus Corynespora cassiicola.',
        'cure': 'Apply fungicides, improve air circulation, and avoid excessive leaf wetness.'
    },
    'Tomato__Tomato_mosaic_virus': {
        'name': 'Tomato Mosaic Virus',
        'cause': 'Caused by the Tomato Mosaic Virus (ToMV).',
        'cure': 'Remove infected plants immediately and disinfect tools; resistant varieties can help prevent infection.'
    },
    'Tomato__Tomato_YellowLeaf__Curl_Virus': {
        'name': 'Tomato Yellow Leaf Curl Virus',
        'cause': 'Spread by whiteflies carrying the TYLCV virus.',
        'cure': 'Control whiteflies with insecticides and use virus-resistant tomato varieties.'
    },
    'Tomato_Bacterial_spot': {
        'name': 'Tomato Bacterial Spot',
        'cause': 'Caused by Xanthomonas campestris pv. vesicatoria.',
        'cure': 'Apply copper-based sprays and avoid handling plants when they are wet.'
    },
    'Tomato_Early_blight': {
        'name': 'Tomato Early Blight',
        'cause': 'Caused by the fungus Alternaria solani.',
        'cure': 'Use fungicides like chlorothalonil or mancozeb and practice crop rotation.'
    },
    'Tomato_healthy': {
        'name': 'Healthy Tomato',
        'cause': 'No disease detected.',
        'cure': 'Continue with regular watering, fertilization, and pest control for healthy plants.'
    },
    'Tomato_Late_blight': {
        'name': 'Tomato Late Blight',
        'cause': 'Caused by Phytophthora infestans, a water mold.',
        'cure': 'Use fungicides containing copper, remove infected plants, and ensure good drainage.'
    },
    'Tomato_Leaf_Mold': {
        'name': 'Tomato Leaf Mold',
        'cause': 'Caused by the fungus Passalora fulva (previously Fulvia fulva).',
        'cure': 'Improve ventilation, avoid overhead irrigation, and apply appropriate fungicides.'
    },
    'Tomato_Septoria_leaf_spot': {
        'name': 'Tomato Septoria Leaf Spot',
        'cause': 'Caused by the fungus Septoria lycopersici.',
        'cure': 'Use fungicides, rotate crops, and remove infected leaves to prevent spread.'
    },
    'Tomato_Spider_mites_Two_spotted_spider_mite': {
        'name': 'Tomato Spider Mites (Two-Spotted Spider Mite)',
        'cause': 'Caused by infestation of Tetranychus urticae.',
        'cure': 'Use miticides or insecticidal soaps. Maintain humidity and remove heavily infested plants.'
    }
}
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None  # define prediction at the start

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file part in the request!")

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', prediction="No file selected for uploading!")

        if file:
            filepath = os.path.join('uploads', file.filename)
            os.makedirs('uploads', exist_ok=True)  # Ensure uploads directory exists
            file.save(filepath)

            img = image.load_img(filepath, target_size=(128, 128))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = img / 255.0

            prediction_array = model.predict(img)
            predicted_label = list(disease_info.keys())[np.argmax(prediction_array)]
            info = disease_info[predicted_label]

            prediction = f"The crop is affected by {info['name']}. {info['cause']} Recommended cure: {info['cure']}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)