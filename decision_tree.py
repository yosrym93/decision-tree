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
        if value not in attribute.values and value != "NA":
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
    def __init__(self, examples, missing_value_weight):
        self.examples = examples
        self.missing_value_weight = missing_value_weight

    @abstractmethod
    def classify(self, example):
        pass

    @abstractmethod
    def classify_missing_values(self, example, path_weight, probability_weights):
        pass


class InternalNode(Node):
    def __init__(self, examples, attributes, missing_value_weight):
        super().__init__(examples, missing_value_weight)
        self.attributes = attributes.copy()
        self.split_attr = None
        self.children = {}
        self.majority_class = None

    def classify(self, example):
        example_split_attribute_value = example.get_attr_value(self.split_attr)

        probability_weights = defaultdict(lambda: 0.0)
        if example_split_attribute_value == "NA":
            self.classify_missing_values(example, 1, probability_weights)
            # Normalization
            max_prob_class = None
            max_prob = 0
            probability_sum = sum(probability_weights.values())
            for cls, probability in probability_weights.items():
                probability_weights[cls] = probability / probability_sum
                if probability_weights[cls] > max_prob:
                    max_prob = probability_weights[cls]
                    max_prob_class = cls
            return max_prob_class, dict(probability_weights)
        else:
            next_child = self.children[example.get_attr_value(self.split_attr)]
            return next_child.classify(example)

    def classify_missing_values(self, example, path_weight, probability_weights):
        example_split_attribute_value = example.get_attr_value(self.split_attr)
        parent_path_weight = path_weight
        if example_split_attribute_value == "NA":
            for _, child in self.children.items():
                path_weight = parent_path_weight * child.missing_value_weight
                child.classify_missing_values(example, path_weight, probability_weights)
        else:
            next_child = self.children[example.get_attr_value(self.split_attr)]
            next_child.classify_missing_values(example, path_weight, probability_weights)

    def __str__(self):
        return "InternalNode(Split Attribute: {}, Majority Class: {}, Children: {}\n)"\
            .format(self.split_attr, self.majority_class, self.children)

    def __repr__(self):
        return self.__str__()


class LeafNode(Node):
    def __init__(self, examples, classification, missing_value_weight):
        super().__init__(examples, missing_value_weight)
        self.classification = classification

    def classify(self, example):
        return self.classification, None

    def classify_missing_values(self, example, path_weight, probability_weights):
        probability_weights[self.classification] += path_weight

    def __str__(self):
        return "LeafNode(Classification: {}\n)".format(self.classification)

    def __repr__(self):
        return self.__str__()


class DecisionTree:
    def __init__(self, examples, attributes):
        self.root = InternalNode(examples, attributes, 1)
        self.__train(self.root)

    def __str__(self):
        return "DecisionTree(Root: {}\n)".format(self.root)

    @staticmethod
    def __calc_entropy(examples):
        class_counts = defaultdict(lambda: 0)
        for example in examples:
            class_counts[example.classification] += 1

        entropy = 0
        for _, count in class_counts.items():
            p = count / len(examples)
            entropy -= p * log2(p)
        return entropy

    @staticmethod
    def __get_majority_class(examples):
        class_counts = defaultdict(lambda: 0)
        for example in examples:
            if example.classification != "NA":
                class_counts[example.classification] += 1

        class_counts = Counter(class_counts)
        return class_counts.most_common(1)[0][0]

    @staticmethod
    def __create_child_node(parent_node, examples, attributes, missing_value_weight) -> Node:
        # If we ran out of examples, create a leaf with classification based on parent's majority
        if len(examples) == 0:
            return LeafNode(examples, parent_node.majority_class, missing_value_weight)

        # If this is a pure node, create a leaf node with the classification of all examples
        class_counts = defaultdict(lambda: 0)
        for example in examples:
            class_counts[example.classification] += 1

        if len(class_counts) == 1:
            classification = list(class_counts.keys())[0]
            return LeafNode(examples, classification, missing_value_weight)
        # If we ran out of attributes, create a leaf with classification based on parent's majority
        if len(attributes) == 0:
            return LeafNode(examples, parent_node.majority_class, missing_value_weight)

        # Otherwise create an internal node
        return InternalNode(examples, attributes, missing_value_weight)

    def __train(self, node: InternalNode):
        # Find the best split attribute
        node.majority_class = self.__get_majority_class(node.examples)
        node_entropy = self.__calc_entropy(node.examples)

        max_information_gain = 0
        split_attr = None
        split_attr_examples_subsets = None
        for attr in node.attributes:
            missing_values_examples_count = 0
            # Divide examples into subsets based on attribute values
            examples_subsets = defaultdict(lambda: list())
            for example in node.examples:
                attr_value = example.get_attr_value(attr)
                if attr_value != "NA":
                    examples_subsets[attr_value].append(example)
                else:
                    missing_values_examples_count += 1

            # Calculate information gain
            information_gain = node_entropy
            for value in attr.values:
                if value != "NA":
                    subset_entropy = self.__calc_entropy(examples_subsets[value])
                    value_pure_freq = len(examples_subsets[value]) / len(node.examples)
                    value_freq = (len(examples_subsets[value]) +
                                  (missing_values_examples_count * value_pure_freq)) / len(node.examples)
                    information_gain -= subset_entropy * value_freq

            # A better split attribute is found, store the attribute and the examples split based on its values
            information_gain = 0 if information_gain < 0 else information_gain
            if information_gain >= max_information_gain:
                max_information_gain = information_gain
                split_attr = attr
                split_attr_examples_subsets = examples_subsets

        # Create children nodes based on the split attribute
        node.split_attr = split_attr
        remaining_attributes = node.attributes.copy()
        remaining_attributes.remove(split_attr)

        for value in split_attr.values:
            if value != "NA":
                missing_value_weight = len(split_attr_examples_subsets[value]) / len(node.examples)
                child_node = self.__create_child_node(
                    node, split_attr_examples_subsets[value], remaining_attributes, missing_value_weight
                )
                node.children[value] = child_node
                if isinstance(child_node, InternalNode):
                    self.__train(child_node)

    def classify(self, example):
        return self.root.classify(example)
