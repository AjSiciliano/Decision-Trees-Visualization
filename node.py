import math
from collections import Counter
import uuid
from treelib import Node as printed_node
from treelib import Tree as printed_tree
from operator import itemgetter
from functions import *

#Authors: Andrew Siciliano, Christopher Wu
#Simple C4.5 Implementation For Decision Tree
#Uses Miroslov Kubat's 'Introduction to Machine Learning' as a reference

class node:

    def __init__(self,isLeaf,depth,is_x):

        self.depth = depth
        self.isLeaf = isLeaf
        self.is_x = is_x
        self.children = [None,None]
        self.value = None
        self.num_passed = 0
        self.sub_tree_error_rate = None
        self.reached_values = []
        self.num_incorrect = 0
        self.uuid = str(uuid.uuid4())

    def printify_helper(self,tree,parent_uid):
        tree.create_node(self.stringify(),self.uuid,parent=parent_uid)

        if self.children[0] != None:
            self.children[0].printify_helper(tree,self.uuid)
        if self.children[1] != None:
            self.children[1].printify_helper(tree,self.uuid)

    def printify(self):
        ptree = printed_tree()
        ptree.create_node(self.stringify(), self.uuid)

        if self.children[0] != None:
            self.children[0].printify_helper(ptree,self.uuid)
        if self.children[1] != None:
            self.children[1].printify_helper(ptree,self.uuid)
        return ptree

    def stringify(self):
        if(self.isLeaf):
            return str(self.value)
        else: #It is a node
            s = "X - " if self.is_x else "Y - "
            return s + str(self.threshold)

    def evaluate_error_rate_of_tree(self,testing_set,pruned_ids=None):
        accuracy_total = 0
        for element in testing_set:
            if self.evaluate(element[0],element[1],None,pruned_ids) == element[2]:
                accuracy_total+=1
        return accuracy_total / len(testing_set)

    def get_prune_ids(self,c,testing_set):
        sub_trees_per_depth = self.toArray()

        pruned_ids = []
        min_error = 0
        
        done = False

        actual=1-self.evaluate_error_rate_of_tree(testing_set)

        for depth in sub_trees_per_depth[::-1]:
            for i in range(len(depth)):
                error_test = 1-self.evaluate_error_rate_of_tree(testing_set,pruned_ids + depth[len(depth)-1:])
                if(error_test-actual) < c:
                    min_error = error_test
                else:
                    #print(1-min_error)
                    done = True
                    break
            if (not done):
                pruned_ids += depth[:]
            else:
                break
        return pruned_ids

    def build_pruned_value_helper(self):
        if(not self.isLeaf):

            if(len(self.reached_values) != 0):
                val = Counter(self.reached_values).most_common()[0][0]
                self.value = val
                self.num_incorrect = len(self.reached_values)-self.reached_values.count(val)
            else:
                self.num_incorrect = 0

        if self.children[0] != None:
            self.children[0].build_pruned_value_helper()
        if self.children[1] != None:
            self.children[1].build_pruned_value_helper()

    def build_pruned_values(self,training_set):

        self.build(find_thresholds(training_set),training_set)

        for element in training_set:
            self.evaluate(element[0],element[1],element[2])

        if(not self.isLeaf):

            if(len(self.reached_values) != 0):
                val = Counter(self.reached_values).most_common()[0][0]
                self.value = val
                self.num_incorrect = len(self.reached_values)-self.reached_values.count(val)
            else:
                self.num_incorrect = 0

        if self.children[0] != None:
            self.children[0].build_pruned_value_helper()
        if self.children[1] != None:
            self.children[1].build_pruned_value_helper()

    def evaluate_helper(self,x,y,passed_down_class,pruned_ids):
        if passed_down_class != None and pruned_ids == None:
            self.reached_values.append(passed_down_class)

        if(pruned_ids != None):
            if(pruned_ids.count(self.uuid) > 0):
                #Treat pruned ID as a leaf if it is
                return self.value

        if(self.isLeaf): #or error rate is less than or equal to preset C value
            return self.value
        else:
            if self.is_x:
                if x < self.threshold:
                    return self.children[0].evaluate_helper(x,y,passed_down_class,pruned_ids)
                else:
                    return self.children[1].evaluate_helper(x,y,passed_down_class,pruned_ids)
            else:
                if y < self.threshold:
                    return self.children[0].evaluate_helper(x,y,passed_down_class,pruned_ids)
                else:
                    return self.children[1].evaluate_helper(x,y,passed_down_class,pruned_ids)

    def evaluate(self,x,y,passed_down_class=None,pruned_ids=None):

        if(pruned_ids != None):
            if(pruned_ids.count(self.uuid) > 0):
                return self.value

        if passed_down_class != None and pruned_ids == None:
            self.reached_values.append(passed_down_class)

        if self.is_x:
            if x < self.threshold:
                return self.children[0].evaluate_helper(x,y,passed_down_class,pruned_ids)
            else:
                return self.children[1].evaluate_helper(x,y,passed_down_class,pruned_ids)
        else:
            if y < self.threshold:
                return self.children[0].evaluate_helper(x,y,passed_down_class,pruned_ids)
            else:
                return self.children[1].evaluate_helper(x,y,passed_down_class,pruned_ids)

    def build(self,target_thresholds,data_input):

        def is_leaf_calc(data_input_now):
            first = data_input_now[0][2]
            for value in data_input_now:
                if value[2] != first: 
                    return False
            return True

        x_thresh, x_infogain = target_thresholds[0]
        y_thresh, y_infogain = target_thresholds[1]

        self.is_x = x_infogain >= y_infogain
        self.threshold = x_thresh if self.is_x else y_thresh

        def split_array():
            left = []
            right = []
            for element in data_input:
                if element[0 if self.is_x else 1] < self.threshold:
                    left.append(element)
                else:
                    right.append(element)
            return [left,right]

        left_data,right_data = split_array()

        left_child = node((y_infogain == 0 and x_infogain == 0) or is_leaf_calc(left_data),self.depth + 1,None)
        right_child = node((y_infogain == 0 and x_infogain == 0) or is_leaf_calc(right_data),self.depth + 1,None)

        self.children[0]=(left_child)
        self.children[1]=(right_child)

        if not left_child.isLeaf:
            left_thresholds = find_thresholds(left_data)
            self.children[0].build(left_thresholds[:],left_data) 
        else:
            self.parent_of_leaf = True
            if(left_data != []):
                self.children[0].value = left_data[0][2]
            else:
                l = [element_instance[2] for element_instance in right_data]
                self.children[0].value = not Counter(l).most_common()[0][0]

        if not right_child.isLeaf:
            right_thresholds = find_thresholds(right_data) 
            self.children[1].build(right_thresholds[:],right_data)
        else:
            self.parent_of_leaf = True
            if(right_data != []):
                self.children[1].value = right_data[0][2]
            else:
                l = [element_instance[2] for element_instance in left_data]
                self.children[1].value = not Counter(l).most_common()[0][0]

    def toArray(self):
        #All the subtrees
        #Call from root
        first_array = self.array_helper([])
        return_array = []

        for node in first_array:
            if(len(return_array) <= node[0]):
                return_array.append([node[1]])
            else:
                return_array[node[0]].append(node[1])

        return return_array

    def array_helper(self, tree):

        if(not self.isLeaf):
            tree.append([self.depth,self.uuid])

        if(not self.children[0].isLeaf):
            self.children[0].array_helper(tree)

        if(not self.children[1].isLeaf):
            self.children[1].array_helper(tree)

        return tree

#___________References___________#
#https://stackoverflow.com/questions/34012886/print-binary-tree-level-by-level-in-python
#https://stackoverflow.com/questions/138250/how-to-read-the-rgb-value-of-a-given-pixel-in-python
#https://stackoverflow.com/questions/65340769/getting-attributeerror-im-must-have-seek-method-while-trying-to-save-as-gif
#https://www.codegrepper.com/code-examples/python/python+create+directory+if+not+exists

