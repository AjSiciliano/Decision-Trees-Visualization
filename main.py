from PIL import Image
import math
import numpy as np
from collections import Counter
from termcolor import colored
import uuid
from itertools import chain, combinations
import sys
from treelib import Node as printed_node
from treelib import Tree as printed_tree

#Authors: Andrew Siciliano, Christopher Wu

#_____________________________________________Methods__________________________________________________#

verbose = True

def printer(text, color=None,end=None): 
    if verbose:
        if color != None and end == None:
            print(colored(text, color))
        elif color != None and end != None:
            print(colored(text, color),end=end)
        else:
            print(text)

# def data_to_splittable(input_data,width,height):

#     # return_array = [[]*height]*width

#     for element in input_data:
#         x,y,value = element

#     #return return_array

# def reverse_split(input_data):
#     return_array = []
#     for x in len(input_data):
#         for y in len(input_data[0])
#             return_array.append([x,y,input_data[x][y]])
#     return return_array

def build_data_from_image(file_name):

    data = []

    original_image = Image.open(file_name, 'r').convert('RGB')
    width, height = original_image.size

    for x in range(width):
        column = []
        for y in range(height):
            pixel = original_image.getdata()[original_image.size[0]*y+x]
            column.append(pixel[0] == 0)

        data.append(column)

    return data

def calculate_entropy(total_pos, total):
    if(total != 0):
        prob_pos = total_pos / total
        prob_neg = 1 - prob_pos

        if(prob_pos != 0 and prob_neg != 0):
            return -prob_pos*math.log2(prob_pos)-prob_neg*math.log2(prob_neg)
        elif(prob_pos == 0 and prob_neg == 0):
            return 0
        elif(prob_neg == 0):
            return -prob_pos*math.log2(prob_pos)
        else:
            return -prob_neg*math.log2(prob_neg)
    else:
        return 0
    
def find_info_gains(thresholds_x, thresholds_y, data_input):

    width = len(data_input)
    height = len(data_input[0])

    total_training_set = height*width
    total_positive = 0
    
    total_below_positive_x = [0]*len(thresholds_x)
    total_below_positive_y = [0]*len(thresholds_y)

    for x in range(width):

        for y in range(height):

            if(data_input[x][y]):

                total_positive += 1

                for threshold_index in range(len(thresholds_x)):
                    if(x < thresholds_x[threshold_index]):
                        total_below_positive_x[threshold_index] += 1

                for threshold_index in range(len(thresholds_y)):
                    if(y < thresholds_y[threshold_index]):
                        total_below_positive_y[threshold_index]+=1

    hT = calculate_entropy(total_positive,total_training_set)

    def info_gain(thresholds,total_below_positive_attribute,multiplier):

        info_gain = []

        for threshold_index in range(len(thresholds)):

            thresh = thresholds[threshold_index]

            total_below = int(math.ceil(thresh - 1) + 1) * (multiplier)
            total_above = total_training_set - total_below

            total_below_positive = total_below_positive_attribute[threshold_index]
            total_above_positive = total_positive - total_below_positive

            average_entropy_below = (total_below/total_training_set)*calculate_entropy(total_below_positive,total_below) 
            average_entropy_above = (total_above/total_training_set)*calculate_entropy(total_above_positive,total_above)

            info_gain.append(hT - (average_entropy_below + average_entropy_above))

        return info_gain

    return [info_gain(thresholds_x,total_below_positive_x,height), info_gain(thresholds_y,total_below_positive_y,width)]

def find_thresholds(data_input):

    width = len(data_input)
    height = len(data_input[0])

    x_thresholds, y_thresholds = [],[]

    previous = [0,data_input[0][0]]

    all_x_values = data_input[0][0]

    for x in range(width):

        for y in range(height):

            if data_input[x][y] != previous[1]:
                x_thresholds.append((previous[0] + x) / 2)
                break

            previous = [x,data_input[x][y]]
    #Y
    previous = [0,data_input[0][0]]

    all_y_values = data_input[0][0]

    for y in range(height):

        for x in range(width):
            if data_input[x][y] != previous[1]:
                y_thresholds.append((previous[0] + y) / 2)
                break
            previous = [y,data_input[x][y]]

    # < threshold is Positive, cant be less than zero therefore neg
    x_thresholds.append(width if all_x_values else 0) if len(x_thresholds) == 0 else None
    y_thresholds.append(height if all_y_values else 0) if len(y_thresholds) == 0 else None

    #we now found the X/Y thresholds, and accounted for if they are of length zero Woot Woot!

    #Now lets calculate all the information gains for each!
    info_gains = find_info_gains(x_thresholds,y_thresholds,data_input)

    #first index: threshold; second index: information gain for that threshold
    x_best_threshold = [x_thresholds[info_gains[0].index(max(info_gains[0]))],max(info_gains[0])]
    y_best_threshold = [y_thresholds[info_gains[1].index(max(info_gains[1]))],max(info_gains[1])]

    return [x_best_threshold, y_best_threshold]

#______________________________________________Tree___________________________________________________#
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

    def stringify(self):
        if(self.isLeaf):
            return "L"
        else:
            return "N"

    def evaluate_error_rate_of_tree(self,training_set,pruned_ids=None):
        accuracy_total = 0
        for x in range(len(training_set)):
            for y in range(len(training_set[0])):
                if tree.evaluate(x,y,x,y,training_set,pruned_ids) == training_set[x][y]:
                    accuracy_total+=1
        return accuracy_total / (len(training_set)*len(training_set[0]))

    def get_prune_ids(self,c,training_set):
        sub_trees_per_depth = self.toArray()

        pruned_ids = []
        min_error = 0
        
        done = False

        actual=1-self.evaluate_error_rate_of_tree(training_set)

        for depth in sub_trees_per_depth[::-1]:
            for i in range(len(depth)):
                error_test = 1-self.evaluate_error_rate_of_tree(training_set,pruned_ids + depth[len(depth)-1:])
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

        for x in range(len(training_set)):
            for y in range(len(training_set[1])):
                tree.evaluate(x,y,x,y,training_set)

        if(not self.isLeaf):

            #print(Counter(self.reached_values).most_common())
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

    def evaluate_helper(self,x,y,origin_x,origin_y,data_original,pruned_ids):
        if origin_x != None and origin_y != None and data_original != None and pruned_ids == None:
            self.reached_values.append(data_original[origin_x][origin_y])

        if(pruned_ids != None):
            if(pruned_ids.count(self.uuid) > 0):
                #Treat pruned ID as a leaf if it is
                return self.value

        if(self.isLeaf): #or error rate is less than or equal to preset C value
            return self.value
        else:
            t_reset=self.threshold

            if self.is_x:
                if x < self.threshold:
                    return self.children[0].evaluate_helper(x,y,origin_x,origin_y,data_original,pruned_ids)
                else:
                    return self.children[1].evaluate_helper(x-t_reset,y,origin_x,origin_y,data_original,pruned_ids)
            else:
                if y < self.threshold:
                    return self.children[0].evaluate_helper(x,y,origin_x,origin_y,data_original,pruned_ids)
                else:
                    return self.children[1].evaluate_helper(x,y-t_reset,origin_x,origin_y,data_original,pruned_ids)

    def evaluate(self,x,y,origin_x=None,origin_y=None,data_original=None,pruned_ids=None):
        # t_reset=int(math.ceil(self.threshold - 1) + 1)
        t_reset=self.threshold

        if(pruned_ids != None):
            if(pruned_ids.count(self.uuid) > 0):
                return self.value

        if origin_x != None and origin_y != None and data_original != None and pruned_ids == None:
            self.reached_values.append(data_original[origin_x][origin_y])

        if self.is_x:
            if x < self.threshold:
                return self.children[0].evaluate_helper(x,y,origin_x, origin_y,data_original,pruned_ids)
            else:
                return self.children[1].evaluate_helper(x-t_reset,y,origin_x, origin_y,data_original,pruned_ids)
        else:
            if y < self.threshold:
                return self.children[0].evaluate_helper(x,y,origin_x, origin_y,data_original,pruned_ids)
            else:
                return self.children[1].evaluate_helper(x,y-t_reset,origin_x, origin_y,data_original,pruned_ids)

    def build(self,target_thresholds, data_input):

        def is_leaf_calc(data_input_now):
            previous = data_input_now[0][0]
            for x in range(len(data_input_now)):
                for y in range(len(data_input_now[0])):
                    if data_input_now[x][y] != previous:
                        return False
            return True

        x_thresh, x_infogain = target_thresholds[0]
        y_thresh, y_infogain = target_thresholds[1]

        self.is_x = x_infogain >= y_infogain
        self.threshold = x_thresh if self.is_x else y_thresh

        left_data = None
        right_data = None

        total_below = int(math.ceil(self.threshold - 1) + 1)

        np_array = np.array(data_input)

        if(self.is_x):
            #working with a split by x value
            left_data = np_array[:total_below,:] #less than x
            right_data = np_array[total_below:,:] #greater than x
        else:
            #working with a split by y value
            left_data = np_array[:,:total_below] #less than y
            right_data = np_array[:,total_below:] #greater than y

        left_data = left_data.tolist()
        right_data = right_data.tolist()

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
                self.children[0].value = (left_data[0][0])
            else:
                l = []
                for row in right_data:
                    for column in row:
                        l.append(column)
                self.children[0].value = not Counter(l).most_common()[0][0]

        if not right_child.isLeaf:
            right_thresholds = find_thresholds(right_data) 
            self.children[1].build(right_thresholds[:],right_data)
        else:
            self.parent_of_leaf = True
            if(right_data != []):
                self.children[1].value = (right_data[0][0])
            else:
                l = []
                for row in left_data:
                    for column in row:
                        l.append(column)
                self.children[1].value = not Counter(l).most_common()[0][0]

    def toArray(self):
        #ALL the subtrees
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
        # else:
        #     tree.append([self.children[0].depth,self.children[0].uuid])

        if(not self.children[1].isLeaf):
            self.children[1].array_helper(tree)
        # else:
        #     tree.append([self.children[1].depth,self.children[1].uuid])

        return tree

#_____________________________________________________________________________________________________#

#images = ["abstract", "star","trapezoid","triangle","halloween"]

images = ["abstract"]

printer("\nRunning Image Tests.....\n")

for i in images:

    printer("______________________________________  Training "+i+".jpg" + " ______________________________________\n")

    init_data = build_data_from_image("test_images/"+i+".jpg")

    initial_thresholds = find_thresholds(init_data)
    #NEED TO CREATE A SMALLER TRAINING SET

    tree = node(False,0,None)
    tree.build(initial_thresholds,init_data) #call on the root node

    predicted_image=Image.new(mode="RGB",size=(len(init_data),len(init_data[1])),color=(255,255,255))

    total = len(init_data) * len(init_data[1])
    total_positive,total_negative,accuracy_total,accuracy_positive,accuracy_negative = 0,0,0,0,0

    for x in range(len(init_data)):
        for y in range(len(init_data[1])):
            if tree.evaluate(x,y):
                predicted_image.putpixel((x,y),(0,0,0))

            if(init_data[x][y]):
                total_positive += 1
                if(tree.evaluate(x,y)): accuracy_positive += 1

            if(not init_data[x][y]):
                total_negative += 1
                if(not tree.evaluate(x,y)): accuracy_negative += 1

            if(tree.evaluate(x,y) == init_data[x][y]):
                accuracy_total += 1

    img_path = "predicted_images/"+i+"_predicted.jpg"

    accuracy_total /= total
    accuracy_positive /= total_positive
    accuracy_negative /= total_negative

    printer("Accuracies: "+i+".jpg\n","green")

    printer("Total Accuracy: " + str(accuracy_total * 100))
    printer("Positive Accuracy: " + str(accuracy_positive * 100))
    printer("Negative Accuracy: " + str(accuracy_negative * 100))

    predicted_image.save(img_path)

    printer("\nSaved image in path: " + img_path +"\n")

    tree.build_pruned_values(init_data)

    pruned_ids_list = []

    printer("Generating Pruned Gif and Images.....\n", "red")

    printer("Now Initiliazing Parameters for the Pruned Trees .....")
    for x in range(25):
        if (x != 0):
            printer("Progress: " + str(100*(x/25)//1 - .01) + "% \r","green","")
        else:
            printer("Progress: " + str(0.0) + "% \r","green","")
        pruned_ids_list.append([x,tree.get_prune_ids((x/100),init_data)])

    printer("Done Initiliazing Parameters!\n", "green")
    printer("Now creating sequence of images.....")

    pruned_images = []
    first_path = ""

    for c in pruned_ids_list:
        pruned_image = Image.new(mode="RGB",size=(len(init_data),len(init_data[1])),color=(255,255,255))

        pruned_image_path =  i + "_pruned/c_" + str(c[0]) + ".jpg"
        if first_path == "":
            first_path = pruned_image_path

        for x in range(len(init_data)):
            for y in range(len(init_data[1])):
                if tree.evaluate(x,y,x,y,init_data,c[1]):
                    pruned_image.putpixel((x,y),(0,0,0))

        pruned_image.save(pruned_image_path)
        pruned_images.append(np.asarray(Image.open(pruned_image_path)))

    printer("\nSaved pruned images in diectory : /"+i+"_pruned\n")

    the_gif_format = [Image.fromarray(img) for img in pruned_images]

    first = Image.open(first_path)

    first.save("gifs/" + i + '_pruned_animation.gif', save_all=True, append_images=the_gif_format,optimize=False, duration=200, loop=0)

    printer("Saved pruned image gif in path: " + i + '_pruned_animation.gif' +"\n")
    printer("Finished with " + i + ".jpg","magenta")

printer("___________________________________________________________________________________________________________\n")

printer("Done Woot Woot!!\n","magenta")

#___________References___________#
#https://stackoverflow.com/questions/34012886/print-binary-tree-level-by-level-in-python
#https://stackoverflow.com/questions/138250/how-to-read-the-rgb-value-of-a-given-pixel-in-python
#https://stackoverflow.com/questions/65340769/getting-attributeerror-im-must-have-seek-method-while-trying-to-save-as-gif
