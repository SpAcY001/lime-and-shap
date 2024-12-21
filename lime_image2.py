import lime
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
from keras.applications import inception_v3
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from lime import lime_image, lime_tabular
from skimage.segmentation import mark_boundaries
from skimage.transform import resize

img_path = r'C:\Users\SESA737860\Downloads\pexels-pixabay-417173.jpg'
img = skimage.io.imread(img_path)
img = skimage.transform.resize(img, (299,299))
img = (img - 0.5)*2

img = inception_v3.preprocess_input(img)
img = np.expand_dims(img, axis=0)

model = inception_v3.InceptionV3()

preds = model.predict(img)
decoded_preds = decode_predictions(preds)

print("Top 5 Predictions:")
for i, (imagenet_id, label, score) in enumerate(decoded_preds[0]):
    print(f"{i + 1}: {label} ({score:.2f})")

predicted_class_label = decoded_preds[0][0][1]

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(img[0], model.predict, top_labels=5, hide_color=0, num_samples=1000)

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.title(f"Predicted Class: {predicted_class_label}")
plt.show()
