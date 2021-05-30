from decision_tree import DecisionTree, Example, Attribute

Gender = Attribute("M", "F", name="Gender")
Education = Attribute("High", "Moderate", "None", name="Education")
FinancialStatus = Attribute("R", "M", "P", name="FinancialStatus")

attributes = [Gender, Education, FinancialStatus]

examples = [
    Example('Y').set_attr_value(Gender, 'M').set_attr_value(Education, 'High').set_attr_value(FinancialStatus, 'R'),
    Example('N').set_attr_value(Gender, 'M').set_attr_value(Education, 'Moderate').set_attr_value(FinancialStatus, 'R'),
    Example('N').set_attr_value(Gender, 'F').set_attr_value(Education, 'None').set_attr_value(FinancialStatus, 'P'),
    Example('Y').set_attr_value(Gender, 'M').set_attr_value(Education, 'High').set_attr_value(FinancialStatus, 'P'),
    Example('Y').set_attr_value(Gender, 'F').set_attr_value(Education, 'High').set_attr_value(FinancialStatus, 'M'),
    Example('N').set_attr_value(Gender, 'F').set_attr_value(Education, 'None').set_attr_value(FinancialStatus, 'M'),
    Example('N').set_attr_value(Gender, 'F').set_attr_value(Education, 'Moderate').set_attr_value(FinancialStatus, 'P'),
    Example('Y').set_attr_value(Gender, 'F').set_attr_value(Education, 'Moderate').set_attr_value(FinancialStatus, 'R'),
    Example('Y').set_attr_value(Gender, 'M').set_attr_value(Education, 'High').set_attr_value(FinancialStatus, 'R'),
    Example('Y').set_attr_value(Gender, 'M').set_attr_value(Education, 'High').set_attr_value(FinancialStatus, 'P'),
]

tree = DecisionTree(examples, attributes)
classification = tree.classify(Example(None).set_attr_value(Gender, 'M')
                               .set_attr_value(Education, 'None')
                               .set_attr_value(FinancialStatus, 'P')
                               )
print(classification)
print(tree)
