'''from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load your pre-trained model here
model = tf.keras.models.load_model('plant_identification_model.h5')
#model = tf.keras.models.load_model('plant_identification_model.keras')  # Or .h5 if that's the file format you used
#model = tf.keras.models.load_model('models/plant_identification_model.keras')

# Define your label mapping dictionary
label_mapping = {
    0: 'Alpinia Galanga (Rasna)',
    1: 'Amaranthus Viridis (Arive-Dantu)',
    2: 'Artocarpus Heterophyllus (Jackfruit)',
    3: 'Azadirachta Indica (Neem)',
    4: 'Basella Alba (Basale)',
    5: 'Brassica Juncea (Indian Mustard)',
    6: 'Carissa Carandas (Karanda)',
    7: 'Citrus Limon (Lemon)',
    8: 'Ficus Auriculata (Roxburgh fig)',
    9: 'Ficus Religiosa (Peepal Tree)',
    10: 'Hibiscus Rosa-sinensis',
    11: 'Jasminum (Jasmine)',
    12: 'Mangifera Indica (Mango)',
    13: 'Mentha (Mint)',
    14: 'Moringa Oleifera (Drumstick)',
    15: 'Muntingia Calabura (Jamaica Cherry-Gasagase)',
    16: 'Murraya Koenigii (Curry)',
    17: 'Nerium Oleander (Oleander)',
    18: 'Nyctanthes Arbor-tristis (Parijata)',
    19: 'Ocimum Tenuiflorum (Tulsi)',
    20: 'Piper Betle (Betel)',
    21: 'Plectranthus Amboinicus (Mexican Mint)',
    22: 'Pongamia Pinnata (Indian Beech)',
    23: 'Psidium Guajava (Guava)',
    24: 'Punica Granatum (Pomegranate)',
    25: 'Santalum Album (Sandalwood)',
    26: 'Syzygium Cumini (Jamun)',
    27: 'Syzygium Jambos (Rose Apple)',
    28: 'Tabernaemontana Divaricata (Crape Jasmine)',
    29: 'Trigonella Foenum-graecum (Fenugreek)'
}

# Preprocess the uploaded image
def preprocess_image(image):
    # Resize the image to match the input size of your model (e.g., 224x224)
    image = image.resize((224, 224))
    
    # Convert the image to an array and preprocess for your specific model
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize pixel values (if required)
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

# Process predictions to get label and confidence
def process_predictions(predictions):
    predicted_label_index = np.argmax(predictions)
    predicted_label = label_mapping.get(predicted_label_index, 'Unknown')
    confidence = predictions[0][predicted_label_index]
    
    return f'Predicted Label: {predicted_label}, Confidence: {confidence:.2f}'

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image classification
@app.route('/classify', methods=['POST'])
def classify():
    # Get the uploaded image file from the form
    uploaded_image = request.files['image']

    if uploaded_image.filename != '':
        # Open and preprocess the image
        image = Image.open(uploaded_image)
        preprocessed_image = preprocess_image(image)

        # Make predictions using your model
        predictions = model.predict(preprocessed_image)

        # Process the predictions and return the result
        result = process_predictions(predictions)
        return render_template('result.html', result=result)
    else:
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)'''
'''from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load your pre-trained model
model = tf.keras.models.load_model('plant_identification_model.h5')

# Define a confidence threshold (You can adjust this based on your dataset)
CONFIDENCE_THRESHOLD = 0.6  # For example, if the model's confidence is below 60%, it's an "Unknown" image.

def prepare_image(image):
    """Prepare image for model prediction."""
    img = Image.open(image)
    img = img.resize((224, 224))  # Resize to match the model input size
    img_array = np.array(img) / 255.0  # Normalize the image
    return np.expand_dims(img_array, axis=0)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the image from the form
        file = request.files["file"]
        if file:
            # Prepare the image for prediction
            img = prepare_image(file)
            
            # Predict the label
            predictions = model.predict(img)
            class_idx = np.argmax(predictions, axis=1)
            confidence = np.max(predictions)

            # Check if the prediction confidence is below the threshold
            if confidence < CONFIDENCE_THRESHOLD:
                label = "Unknown"
            else:
                # You can customize the label with your classes
                label = f"Class {class_idx[0]}"  # Replace with your actual class names

            # Render the template with the prediction and image
            return render_template("index.html", label=label, confidence=confidence, image_path=file.filename)

    return render_template("index.html", label=None)

if __name__ == "__main__":
    app.run(debug=True)
'''
'''from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load your pre-trained model here
model = tf.keras.models.load_model('plant_identification_model.h5')

# Define your label mapping dictionary
label_mapping = {
    0: 'Alpinia Galanga (Rasna)',
    1: 'Amaranthus Viridis (Arive-Dantu)',
    2: 'Artocarpus Heterophyllus (Jackfruit)',
    3: 'Azadirachta Indica (Neem)',
    4: 'Basella Alba (Basale)',
    5: 'Brassica Juncea (Indian Mustard)',
    6: 'Carissa Carandas (Karanda)',
    7: 'Citrus Limon (Lemon)',
    8: 'Ficus Auriculata (Roxburgh fig)',
    9: 'Ficus Religiosa (Peepal Tree)',
    10: 'Hibiscus Rosa-sinensis',
    11: 'Jasminum (Jasmine)',
    12: 'Mangifera Indica (Mango)',
    13: 'Mentha (Mint)',
    14: 'Moringa Oleifera (Drumstick)',
    15: 'Muntingia Calabura (Jamaica Cherry-Gasagase)',
    16: 'Murraya Koenigii (Curry)',
    17: 'Nerium Oleander (Oleander)',
    18: 'Nyctanthes Arbor-tristis (Parijata)',
    19: 'Ocimum Tenuiflorum (Tulsi)',
    20: 'Piper Betle (Betel)',
    21: 'Plectranthus Amboinicus (Mexican Mint)',
    22: 'Pongamia Pinnata (Indian Beech)',
    23: 'Psidium Guajava (Guava)',
    24: 'Punica Granatum (Pomegranate)',
    25: 'Santalum Album (Sandalwood)',
    26: 'Syzygium Cumini (Jamun)',
    27: 'Syzygium Jambos (Rose Apple)',
    28: 'Tabernaemontana Divaricata (Crape Jasmine)',
    29: 'Trigonella Foenum-graecum (Fenugreek)'
}

# Preprocess the uploaded image
def preprocess_image(image):
    # Resize the image to match the input size of your model (e.g., 224x224)
    image = image.resize((224, 224))
    
    # Convert the image to an array and preprocess for your specific model
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize pixel values (if required)
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

# Process predictions to get label and confidence with a threshold for "unknown"
def process_predictions(predictions, confidence_threshold=0.5):
    predicted_label_index = np.argmax(predictions)
    predicted_label = label_mapping.get(predicted_label_index, 'Unknown')
    confidence = predictions[0][predicted_label_index]
    
    # Check if confidence is below the threshold to classify it as "Unknown"
    if confidence < confidence_threshold:
        predicted_label = "Unknown"
        confidence = predictions[0][np.argmax(predictions)]  # Confidence of "Unknown" image
    
    return f'Predicted Label: {predicted_label}, Confidence: {confidence:.2f}'

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image classification
@app.route('/classify', methods=['POST'])
def classify():
    # Get the uploaded image file from the form
    uploaded_image = request.files['image']

    if uploaded_image.filename != '':
        # Open and preprocess the image
        image = Image.open(uploaded_image)
        preprocessed_image = preprocess_image(image)

        # Make predictions using your model
        predictions = model.predict(preprocessed_image)

        # Process the predictions and return the result
        result = process_predictions(predictions)
        return render_template('result.html', result=result)
    else:
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
'''
from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Directory to save uploaded images

# Load your pre-trained model
model = tf.keras.models.load_model('plant_identification_model.h5')

# Define your label mapping dictionary
label_mapping = {  # Mapping unchanged
    0: 'Alpinia Galanga (Rasna)',
    1: 'Amaranthus Viridis (Arive-Dantu)',
    2: 'Artocarpus Heterophyllus (Jackfruit)',
    3: 'Azadirachta Indica (Neem)',
    4: 'Basella Alba (Basale)',
    5: 'Brassica Juncea (Indian Mustard)',
    6: 'Carissa Carandas (Karanda)',
    7: 'Citrus Limon (Lemon)',
    8: 'Ficus Auriculata (Roxburgh fig)',
    9: 'Ficus Religiosa (Peepal Tree)',
    10: 'Hibiscus Rosa-sinensis',
    11: 'Jasminum (Jasmine)',
    12: 'Mangifera Indica (Mango)',
    13: 'Mentha (Mint)',
    14: 'Moringa Oleifera (Drumstick)',
    15: 'Muntingia Calabura (Jamaica Cherry-Gasagase)',
    16: 'Murraya Koenigii (Curry)',
    17: 'Nerium Oleander (Oleander)',
    18: 'Nyctanthes Arbor-tristis (Parijata)',
    19: 'Ocimum Tenuiflorum (Tulsi)',
    20: 'Piper Betle (Betel)',
    21: 'Plectranthus Amboinicus (Mexican Mint)',
    22: 'Pongamia Pinnata (Indian Beech)',
    23: 'Psidium Guajava (Guava)',
    24: 'Punica Granatum (Pomegranate)',
    25: 'Santalum Album (Sandalwood)',
    26: 'Syzygium Cumini (Jamun)',
    27: 'Syzygium Jambos (Rose Apple)',
    28: 'Tabernaemontana Divaricata (Crape Jasmine)',
    29: 'Trigonella Foenum-graecum (Fenugreek)'
}

# Preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize the image for the model
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Process predictions to get label and confidence
def process_predictions(predictions, confidence_threshold=0.5):
    predicted_label_index = np.argmax(predictions)
    predicted_label = label_mapping.get(predicted_label_index, 'Unknown')
    confidence = predictions[0][predicted_label_index]
    
    # If confidence is below the threshold, classify as "Unknown"
    if confidence < confidence_threshold:
        predicted_label = "Unknown"
        confidence = None

    return predicted_label, confidence

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image classification
@app.route('/classify', methods=['POST'])
def classify():
    uploaded_image = request.files['image']
    if uploaded_image.filename != '':
        # Save the uploaded image
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_image.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        uploaded_image.save(file_path)

        # Preprocess the image and make predictions
        image = Image.open(file_path)
        preprocessed_image = preprocess_image(image)
        predictions = model.predict(preprocessed_image)

        # Process predictions and prepare data for the result page
        label, confidence = process_predictions(predictions)
        confidence_text = f"{confidence:.2f}" if confidence else "N/A"

        return render_template('result.html', 
                               result=label, 
                               confidence=confidence_text, 
                               image_path=file_path)
    else:
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
