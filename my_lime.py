import lime
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier

X = np.array([[300, 0], [250, 1], [150, 1], [200, 0]])
y = np.array([0, 0, 1, 1])  # 0 for apple, 1 for orange

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X, y)

feature_names = ['Weight', 'Color']

explainer = LimeTabularExplainer(X, feature_names=feature_names, class_names=['Apple', 'Orange'], discretize_continuous=True)
instance_to_explain = np.array([[150, 0.9]])  # [Weight, Color]

explanation = explainer.explain_instance(instance_to_explain[0], classifier.predict_proba)
print(explanation.as_list())


probs = classifier.predict_proba(instance_to_explain)
predicted_class_index = np.argmax(probs)
predicted_class = ['Apple', 'Orange'][predicted_class_index]
print("Predicted class:", predicted_class)
