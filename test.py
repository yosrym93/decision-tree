from decision_tree import DecisionTree, Example, Attribute
import random

Gender = Attribute("M", "F", name="Gender")
Education = Attribute("High", "Moderate", "None", name="Education")
FinancialStatus = Attribute("R", "M", "P", name="FinancialStatus")
classifications = ['X', 'Y']

attributes = [Gender, Education, FinancialStatus]
data_set = []
for idx in range(10100):
    data_set.append(Example(random.choice(classifications))
                    .set_attr_value(Gender, random.choice(Gender.values + ['NA']))
                    .set_attr_value(Education, random.choice(Education.values + ['NA']))
                    .set_attr_value(FinancialStatus, random.choice(FinancialStatus.values + ['NA'])))


training_set = data_set[:10000]
test_set = data_set[10000:]

tree = DecisionTree(training_set, attributes)
true_classified = 0
for test in test_set:
    classification, classification_probability = tree.classify(test)
    print(classification, classification_probability)
    if classification == test.classification:
        true_classified += 1

print(true_classified)
