from os import replace
from decision_tree import DecisionTree, Example, Attribute
import random
import pandas as pd
import numpy as np

titanic = pd.read_csv('titanic.csv')
categorial_attributes = ["Pclass","Embarked","Gender","Survived"]
titanic = titanic[categorial_attributes]

attributes = []
for attribute in ["Pclass","Embarked","Gender"]:
    attributes.append(Attribute(*([i for i in map(str, titanic[attribute].unique()) if i != 'nan']), name=attribute))

dataset = []
for _,row in titanic.iterrows():
    if str(row["Survived"]) != 'nan' and str(row[attributes[0].name]) != 'nan' and \
        str(row[attributes[1].name]) != 'nan' and str(row[attributes[2].name]) != 'nan':
        dataset.append(Example(str(row["Survived"]))
                    .set_attr_value(attributes[0], random.choices([str(row[attributes[0].name])] + ['NA'], weights=[95, 5], k=1)[0])
                    .set_attr_value(attributes[1], random.choices([str(row[attributes[1].name])] + ['NA'], weights=[95, 5], k=1)[0])
                    .set_attr_value(attributes[2], random.choices([str(row[attributes[2].name])] + ['NA'], weights=[95, 5], k=1)[0]) )

train_set = dataset[:800]
test_set = dataset[800:]
tree = DecisionTree(train_set, attributes)
true_classified = 0
for test in test_set:
    classification, classification_probability = tree.classify(test)
    if classification == test.classification:
        true_classified += 1

accuracy = true_classified / len(test_set)
print(accuracy)
