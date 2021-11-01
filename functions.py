import math
from termcolor import colored
import sys
from operator import itemgetter
import os
from PIL import Image

#Authors: Andrew Siciliano, Christopher Wu
#Simple C4.5 Implementation For Decision Tree
#Uses Miroslov Kubat's 'Introduction to Machine Learning' as a reference

def get_subdirectory(sd):
    if not os.path.exists(sd):
        os.makedirs(sd)
    return sd

def revert_slittable(attr, classif):

    return_array = []
    for index in range(len(attr)):
        element = attr[index]
        return_array.append((element[0],element[1],classif[index]))
    return return_array

def data_to_splittable(input_data):
    attributes = []
    classifications = []
    for x in range(len(input_data)):
        for y in range(len(input_data[0])):
            attributes.append([x,y])
            classifications.append(input_data[x][y])
    return [attributes,classifications]

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

def find_info_gains(thresholds_x,thresholds_y,data_input):

    total_training_set = len(data_input)
    total_positive = 0
    
    total_below_positive_x = [0]*len(thresholds_x)
    total_below_positive_y = [0]*len(thresholds_y)

    total_below_x = [0]*len(thresholds_x)
    total_below_y = [0]*len(thresholds_y)

    for element in data_input:
        if element[2] == True:
            total_positive += 1

        for threshold_index in range(len(thresholds_x)):
            if(element[0] < thresholds_x[threshold_index]):
                total_below_x[threshold_index] +=1
                total_below_positive_x[threshold_index] += 1 if(element[2] == True) else 0

        for threshold_index in range(len(thresholds_y)):
            if(element[1] < thresholds_y[threshold_index]):
                total_below_y[threshold_index] +=1
                total_below_positive_y[threshold_index]+=1 if(element[2] == True) else 0

    hT = calculate_entropy(total_positive,total_training_set)

    def info_gain(thresholds,total_below,total_below_positive_attribute):

        info_gain = []

        for threshold_index in range(len(thresholds)):

            total_above = total_training_set - total_below[threshold_index]

            total_below_positive = total_below_positive_attribute[threshold_index]
            total_above_positive = total_positive - total_below_positive

            average_entropy_below = (total_below[threshold_index]/total_training_set)*calculate_entropy(total_below_positive,total_below[threshold_index]) 
            average_entropy_above = (total_above/total_training_set)*calculate_entropy(total_above_positive,total_above)

            info_gain.append(hT - (average_entropy_below + average_entropy_above))

        return info_gain

    return [info_gain(thresholds_x,total_below_x,total_below_positive_x), info_gain(thresholds_y,total_below_y,total_below_positive_y)]

def find_thresholds(data_input):

    width = len(data_input)
    height = len(data_input[0])

    x_thresholds, y_thresholds = [],[]

    data_input_by_x = data_input[:]
    data_input_by_x.sort(key=itemgetter(0))
    data_input_by_y = data_input[:]
    data_input_by_y.sort(key=itemgetter(1))

    all_x_values = data_input_by_x[0][2]
    all_y_values = data_input_by_y[0][2]

    previous = None
    
    for entry in data_input_by_x:
        previous = entry if(previous == None) else previous
        if(entry[2] != previous[2]):
            tester = (previous[0] + entry[0])/2
            x_thresholds.append(tester) if x_thresholds.count(tester) == 0 else None

    previous = None

    for entry in data_input_by_y:
        previous = entry if(previous == None) else previous
        if(entry[2] != previous[2]):
            tester = (previous[1] + entry[1])/2
            y_thresholds.append(tester) if y_thresholds.count(tester) == 0 else None

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

#___________References___________#
#https://stackoverflow.com/questions/34012886/print-binary-tree-level-by-level-in-python
#https://stackoverflow.com/questions/138250/how-to-read-the-rgb-value-of-a-given-pixel-in-python
#https://stackoverflow.com/questions/65340769/getting-attributeerror-im-must-have-seek-method-while-trying-to-save-as-gif
#https://www.codegrepper.com/code-examples/python/python+create+directory+if+not+exists

