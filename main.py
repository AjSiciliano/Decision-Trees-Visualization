from PIL import Image
import math
#_______________________________________________________________________________________________#

original_image = Image.new('RGB', (0,0))
image = original_image

width, height = 0,0

def set_image(file_name):
    global image, width, height

    original_image = Image.open(file_name, 'r').convert('RGB')
    image = original_image
    width, height = image.size

def actual_class(x, y): #returns True for positive, False for Negative
    #https://stackoverflow.com/questions/138250/how-to-read-the-rgb-value-of-a-given-pixel-in-python
    pixel = image.getdata()[image.size[0]*y+x]

    #if black, its inside, therefore positive, if white its 255 therefore negative
    return True if pixel[0] == 0 else False

#_______________________________________________________________________________________________#

x_threshold, y_threshold  = 0,0

def convert_cord(value_x, value_y):
    return [value_x < x_threshold, value_y < y_threshold]

def calculate_entropy(total_pos, total):
    prob_pos = total_pos / total
    prob_neg = 1 - prob_pos

    if(prob_pos != 0 and prob_neg != 0):
        return -prob_pos*math.log2(prob_pos)-prob_neg*math.log2(prob_neg)
    elif(prob_pos == 0):
        return -prob_neg*math.log2(prob_neg)
    else:
        return -prob_pos*math.log2(prob_pos)

def find_info_gains(thresholds_x, thresholds_y):

    total_positive = 0
    total = height*width

    total_below_positive_x = [0]*len(thresholds_x)
    total_below_positive_y = [0]*len(thresholds_y)

    for x in range(width):

        for y in range(height):

            if(actual_class(x,y)):

                total_positive += 1

                for threshold_index in range(len(thresholds_x)):
                    if(x < thresholds_x[threshold_index]):
                        total_below_positive_x[threshold_index] += 1

                for threshold_index in range(len(thresholds_y)):
                    if(y < thresholds_y[threshold_index]):
                        total_below_positive_y[threshold_index]+=1

    hT = calculate_entropy(total_positive,total)

    def info_gain(thresholds, total_below_positive_attribute,multiplier):

        info_gain = []

        for threshold_index in range(len(thresholds)):
            thresh = thresholds[threshold_index]

            total_below = int(math.ceil(thresh - 1)) * multiplier

            total_above = total - total_below

            total_below_positive = total_below_positive_attribute[threshold_index]

            total_above_positive = total_positive - total_below_positive

            average_entropy_below = (total_below/total)*calculate_entropy(total_below_positive,total_below) 
            average_entropy_above = (total_above/total)*calculate_entropy(total_above_positive,total_above)

            info_gain.append(hT - (average_entropy_below + average_entropy_above))

        return info_gain

    return [info_gain(thresholds_x,total_below_positive_x,height), info_gain(thresholds_y,total_below_positive_y,width)]

def find_thresholds():

    x_thresholds, y_thresholds = [],[]

    #X
    previous = [0,actual_class(0,0)]

    for x in range(width):

        for y in range(height):

            if actual_class(x,y) != previous[1]:
                x_thresholds.append((previous[0] + x) / 2)
                previous = [x,actual_class(x,y)]
                break

            previous = [x,actual_class(x,y)]

    #Y
    previous = [0,actual_class(0,0)]
    for y in range(height):

        for x in range(width):

            if actual_class(x,y) != previous[1]:
                y_thresholds.append((previous[0] + y) / 2)
                break

            previous = [y,actual_class(x,y)]

    info_gains = find_info_gains(x_thresholds,y_thresholds)
    

    if len(info_gains[1]) == 0 and len(info_gains[0]) == 0:
        return [[0,0],[0,0]]
    elif (len(info_gains[1]) == 0):
        return [[x_thresholds[info_gains[0].index(max(info_gains[0]))],max(info_gains[0])],[0,0]]
    elif (len(info_gains[0]) == 0):
        return [[0,0],[x_thresholds[info_gains[0].index(max(info_gains[0]))],max(info_gains[0])]]

    return [[x_thresholds[info_gains[0].index(max(info_gains[0]))],max(info_gains[0])],[y_thresholds[info_gains[1].index(max(info_gains[1]))],max(info_gains[1])]]

#_______________________________________________________________________________________________#

def subset_x(x_threshold):
    left = []
    right = []

    for x in range(width):
        for y in range(height):
            left.append(x,y,actual_class(x,y)) if x < x_threshold else right.append(x,y,actual_class(x,y))

    return [left, right]

def subset_y(y_threshold):
    left = []
    right = []

    for y in range(width):
        for x in range(height):
            #left.append([x,y,actual_class(x,y)]) if y < y_threshold else right.append([x,y,actual_class(x,y)])
            left.append(actual_class(x,y)) if y < y_threshold else right.append(actual_class(x,y))

    return [left, right]

class node:

    def __init__(self,isLeaf,index,attr):

        self.index = index
        self.children = []
        self.isLeaf = isLeaf
        self.threshold = 0
        self.attribute = attr

    def guess(self,value):
        return value < threshold

    def evaluate(self,value):

        if self.isLeaf:
            return self.guess(value)
        else:
            if value < threshold:
                return self.children[0].evaluate(value)
            else:
                return self.children[1].evaluate(value)

    #def __init__(self,isLeaf,value,file,index):

    def build(self,target_thresholds):
        global image
        global width
        global height

        x_thresh, x_infogain = target_thresholds[0]
        y_thresh, y_infogain = target_thresholds[1]

        if(self.attribute == None): #Root-Node
            self.attribute = x_infogain > y_infogain #False is Y True is X
            self.threshold = x_thresh if self.attribute else y_thresh

        left_image = original_image.crop((0,0,width,y_thresh)) #left Y TOP
        right_image = original_image.crop((0,math.ceil(y_thresh - 1),width,height)) #right Y  BOTTOM
        
        #if self.attribute: #if its an X
        if self.attribute: #attribite is treu if X
            left_image = original_image.crop((0,0,width - math.ceil(x_thresh - 1),height))
            right_image = original_image.crop((math.ceil(x_thresh - 1),0,width,height))

        image = left_image
        width, height = image.size
        left_thresh = find_thresholds()

        image = right_image
        width, height = image.size
        right_thresh = find_thresholds()

        left_node = node(self.index + 1 == 2,self.index + 1,not self.attribute)
        left_node.threshold = left_thresh[0 if self.attribute else 1]

        right_node = node(self.index + 1 == 2,self.index + 1,not self.attribute)
        right_node.threshold = right_thresh[0 if self.attribute else 1]

        self.children.append(left_node)
        self.children.append(right_node)

        if(self.children[0].isLeaf != True):
            self.children[0].build(left_thresh)
        if(self.children[1].isLeaf != True):
            self.children[1].build(right_thresh)
    
    def print_help(self,current):

        if current.isLeaf == True:
            return "|" + str(self.threshold) + "-" + str(self.attribute) + "|"
        else:
            return "\n" + str(current.print_help(current.children[0])) + " " + str(current.index) +  " " + str(current.print_help(current.children[0]))

    def print_tree(self):
        if self.isLeaf == True:
            print(self.attribute)
        else:
            print(str(self.attribute) + "-Root")
            print(self.print_help(self.children[0]))
            print(self.print_help(self.children[1]))

#_______________________________________________________________________________________________#


#def __init__(self,isLeaf,value,file,index):

set_image("circle.jpg")
tree = node(False,0,None)
tree.build(find_thresholds())

tree.print_tree()

#___________References___________#

#https://stackoverflow.com/questions/138250/how-to-read-the-rgb-value-of-a-given-pixel-in-python

# def repeatable(w,h,bb): #bool is X if true, bool is Y if false

#         thresholds = []
        
#         previous = [0,actual_class(0,0)]

#         for ww in range(w):
#             for hh in range(h):
#                 y = ww
#                 x = hh

#                 if(bb == True): #X
#                     x = ww
#                     y = hh

#                 if actual_class(x,y) != previous[1]:
#                     thresholds.append((previous[0] + ww) / 2)
#                     previous = [ww,actual_class(x,y)]
#                     break

#                 previous = [ww,actual_class(x,y)]

#         return thresholds

#     #X
#     x_thresholds = repeatable(width,height,True)

#     #Y
#     y_thresholds = repeatable(height,width,False)

