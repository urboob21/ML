from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model from file
loaded_model = load_model('horse-or-human.h5')

# Check sumary the model
loaded_model.summary()

file_path = "horse-or-human/training/horses/horse02-1.png"

img = image.load_img(file_path, target_size=(300, 300))  # Load and resize
x = image.img_to_array(img)  # Convert to 2D array
x = np.expand_dims(x, axis=0)  # Add new dimension to the array -> 3D as input_shape

classes = loaded_model.predict(x)  # Predict

# There's only one classification: 0 for horse, 1 for human
if classes[0] < 0.5:
    print("It is a horse.")
else:
    print("It is a human.")
