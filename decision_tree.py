from collections import defaultdict, Counter
from math import log2
from abc import ABC, abstractmethod


class Attribute:
    def __init__(self, *values, name=""):
        self.values = list(values)
        self.name = name

    def __str__(self):
        return "Attribute({}: {})".format(self.name, self.values)


class Example:
    def __init__(self, classification):
        self.classification = classification
        self.attr_values = {}

    def set_attr_value(self, attribute: Attribute, value):
        if value not in attribute.values:
            raise Exception(
                "Unsupported attribute value: '{}', supported values are: {}".format(value, attribute.values)
            )
        self.attr_values[attribute] = value
        return self

    def get_attr_value(self, attribute: Attribute):
        return None if attribute not in self.attr_values else self.attr_values[attribute]

    def __str__(self):
        return "Example({}, {})".format(self.attr_values, self.classification)


class Node(ABC):
    def __init__(self, examples):
        self.examples = examples

    @abstractmethod
    def classify(self, example):
        pass


class InternalNode(Node):
    def __init__(self, examples, attributes):
        super().__init__(examples)
        self.attributes = attributes.copy()
        self.split_attr = None
        self.children = {}
        self.majority_class = None

    def classify(self, example):
        # TODO: Handle missing attributes
        next_child = self.children[example.get_attr_value(self.split_attr)]
        return next_child.classify(example)

    def __str__(self):
        return "InternalNode(Split Attribute: {}, Majority Class: {}, Children: {})"\
            .format(self.split_attr, self.majority_class, self.children)

    def __repr__(self):
        return self.__str__()


class LeafNode(Node):
    def __init__(self, examples, classification):
        super().__init__(examples)
        self.classification = classification

    def classify(self, example):
        return self.classification

    def __str__(self):
        return "LeafNode(Classification: {})".format(self.classification)

    def __repr__(self):
        return self.__str__()


class DecisionTree:
    def __init__(self, examples, attributes):
        self.root = InternalNode(examples, attributes)
        self.__train(self.root)

    def __str__(self):
        return "DecisionTree(Root: {})".format(self.root)

    @staticmethod
    def __calc_entropy(examples):
        class_counts = defaultdict(lambda: 0)
        for example in examples:
            class_counts[example.classification] += 1

        entropy = 0
        for classification, count in class_counts.items():
            p = count / len(examples)
            entropy -= p * log2(p)
        return entropy

    @staticmethod
    def __get_majority_class(examples):
        class_counts = defaultdict(lambda: 0)
        for example in examples:
            class_counts[example.classification] += 1

        class_counts = Counter(class_counts)
        return class_counts.most_common(1)[0][0]

    @staticmethod
    def __create_child_node(parent_node, examples, attributes) -> Node:
        # If we ran out of examples, create a leaf with classification based on parent's majority
        if len(examples) == 0:
            return LeafNode(examples, parent_node.majority_class)

        # If this is a pure node, create a leaf node with the classification of all examples
        class_counts = defaultdict(lambda: 0)
        for example in examples:
            class_counts[example.classification] += 1

        if len(class_counts) == 1:
            classification = list(class_counts.keys())[0]
            return LeafNode(examples, classification)

        # If we ran out of attributes, create a leaf with classification based on parent's majority
        if len(attributes) == 0:
            return LeafNode(examples, parent_node.majority_class)

        # Otherwise create an internal node
        return InternalNode(examples, attributes)

    def __train(self, node: InternalNode):
        # Find the best split attribute
        node.majority_class = self.__get_majority_class(node.examples)
        node_entropy = self.__calc_entropy(node.examples)

        max_information_gain = 0
        split_attr = None
        split_attr_examples_subsets = None
        for attr in node.attributes:
            # Divide examples into subsets based on attribute values
            # TODO: Handle missing attributes
            examples_subsets = defaultdict(lambda: list())
            for example in node.examples:
                attr_value = example.get_attr_value(attr)
                examples_subsets[attr_value].append(example)

            # Calculate information gain
            information_gain = node_entropy
            for value in attr.values:
                subset_entropy = self.__calc_entropy(examples_subsets[value])
                information_gain -= subset_entropy * len(examples_subsets[value]) / len(node.examples)

            # A better split attribute is found, store the attribute and the examples split based on its values
            if information_gain > max_information_gain:
                max_information_gain = information_gain
                split_attr = attr
                split_attr_examples_subsets = examples_subsets

        # Create children nodes based on the split attribute
        node.split_attr = split_attr
        remaining_attributes = node.attributes.copy()
        remaining_attributes.remove(split_attr)
        for value in split_attr.values:
            child_node = self.__create_child_node(node, split_attr_examples_subsets[value], remaining_attributes)
            node.children[value] = child_node
            if isinstance(child_node, InternalNode):
                self.__train(child_node)

    def classify(self, example):
        return self.root.classify(example)
