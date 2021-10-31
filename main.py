from PIL import Image
import math
import numpy as np

#_______________________________________________________________________________________________#

def build_data_from_image(file_name):

    data = []

    original_image = Image.open(file_name, 'r').convert('RGB')
    width, height = original_image.size

    for x in range(width):
        column = []
        for y in range(height):
            pixel = original_image.getdata()[original_image.size[0]*(height-y-1)+x]
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

#_______________________________________________________________________________________________#


class node:

    def __init__(self,isLeaf,depth,is_x):

        self.depth = depth
        self.isLeaf = isLeaf
        self.is_x = is_x
        self.children = [None,None]

    def stringify(self):
        if(self.isLeaf):
            return "L"
        else:
            return "N"


    def evaluate_helper(self,x,y,result):

        if(self.isLeaf):
            return result
        else:
            t_reset=self.threshold
            #t_reset=int(math.ceil(self.threshold - 1) + 1)
            #print(self.threshold)
            
            if self.is_x:
                if x <= self.threshold:
                    return self.children[0].evaluate_helper(t_reset-x,y,True)
                else:
                    return self.children[1].evaluate_helper(x-t_reset,y,False)
            else:
                if y <= self.threshold:
                    return self.children[0].evaluate_helper(x,t_reset-y,True)
                else:
                    return self.children[1].evaluate_helper(x,y-t_reset,False)

    def evaluate(self, x,y):
        # t_reset=int(math.ceil(self.threshold - 1) + 1)
        t_reset=self.threshold

        if self.is_x:
            if x <= self.threshold:
                return self.children[0].evaluate_helper(t_reset-x,y,True)
            else:
                return self.children[1].evaluate_helper(x-t_reset,y,False)
        else:
            if y <= self.threshold:
                return self.children[0].evaluate_helper(x,t_reset-y,True)
            else:
                return self.children[1].evaluate_helper(x,y-t_reset,False)

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
        #print(self.is_x)
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

        if not right_child.isLeaf:
            right_thresholds = find_thresholds(right_data) 
            self.children[1].build(right_thresholds[:],right_data)
    
    def toArray(self):
        #Call from root
        return self.array_helper([])

    def array_helper(self, tree):

        tree.append([self.depth,self.stringify()])
        
        if(not self.children[0].isLeaf):
            self.children[0].array_helper(tree)
        else:
            tree.append([self.children[0].depth,self.children[0].stringify()])

        if(not self.children[1].isLeaf):
            self.children[1].array_helper(tree)
        else:
            tree.append([self.children[1].depth,self.children[1].stringify()])

        return tree

    def stringify_tree(self):
        tree_array = self.toArray()
        max_depth = max(tree_array)[0]
        grid_width = len(tree_array) // 4
        #format into strings

        string_array = [""]*(max_depth + 1)
        rows = []
        for node in tree_array: string_array[node[0]] += str(node[1]) + " "

        for depth in string_array:
            row = [""]*(2*grid_width + 1)

            array_row = depth.split(" ")
            num_of_nodes_per_row = len(array_row)

            spacing = (len(row)//(num_of_nodes_per_row*2))

            for x in range(len(array_row) - 1):

                node = str(array_row[x])

                row[x] = node

            rows.append(''.join(row))

        return rows

    def print_tree(self):
        rows_as_strings = self.stringify_tree()
        grid_width = len(max(rows_as_strings, key=len)) * 3 

        root = rows_as_strings[0]
        left_half_start = rows_as_strings[1][0]
        right_half_start = rows_as_strings[1][1]
        
    
        #left side
        if(left_half_start=="L"):
            print("   " + "L")
        # else:

        if(right_half_start=="L"):
            print("   " + "L")
        else:

            num_of_shifts = 0

            for row in range(len(rows_as_strings)):
                num_of_shifts+=1

                printed_row = "\n"
                start_index = 2*row
                for node in rows_as_strings[row]:
                    printed_row += node + " "*num_of_shifts

                print(" "*2*num_of_shifts + printed_row)

        
            
#_______________________________________________________________________________________________#


#_______________________________________________________________________________________________#


images = ["small.jpg", "shape3.jpg", "mushroom.jpg", "triangle.jpg"]

for i in images:

    init_data = build_data_from_image(i)
    initial_thresholds = find_thresholds(init_data)
    
    tree = node(False,0,None)
    tree.build(initial_thresholds,init_data) #call on the root node

    predicted_image=Image.new(mode = "RGB",size=(len(init_data),len(init_data[1])),color=(255, 255, 255))

    # accuracy = 0

    print(i)
    for x in range(len(init_data)):
        for y in range(len(init_data[1])):
            if tree.evaluate(x,y):
                #print(tree.evaluate(x,y))
                predicted_image.putpixel((x,y), (0,0,0))
                
            # if(tree.evaluate(x,y) == True and actual_class(x,y) == True):
            #     accuracy += 1

    predicted_image.save(i + "_predicted.jpg")

    #predicted_image.show()

    # print("Accuracy: " + str(accuracy / (original_image.width * original_image.height)))

    

#___________References___________#
#https://stackoverflow.com/questions/34012886/print-binary-tree-level-by-level-in-python
#https://stackoverflow.com/questions/138250/how-to-read-the-rgb-value-of-a-given-pixel-in-python

