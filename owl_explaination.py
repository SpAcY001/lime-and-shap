import hermit

onotology=hermit.load_onotology(r"C:\Users\SESA737860\Downloads\owl.zip\owl\airport owl\airport_terminal_ADE_train_00000001_sumo.owl")

class_to_classify = onotology.classes["Myclass"]
subclasses = hermit.classify(onotology, class_to_classify)

for subclass in subclasses:
    print(subclass.name)