from PIL import Image
import math
#_______________________________________________________________________________________________#

original_image = Image.new('RGB', (0,0))
image = original_image

width, height = 0,0

def set_image(file_name):
    global image, width, height,original_image

    original_image = Image.open(file_name, 'r').convert('RGB')
    image = original_image
    width, height = image.size

def actual_class(x, y): #returns True for positive, False for Negative
    #https://stackoverflow.com/questions/138250/how-to-read-the-rgb-value-of-a-given-pixel-in-python

    if(image.size[0] != 0 and image.size[1] != 0):
        pixel = image.getdata()[image.size[0]*y+x]

        #if black, its inside, therefore positive, if white its 255 therefore negative
        return True if pixel[0] == 0 else False
    else:
        return False

#_______________________________________________________________________________________________#

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
        # print(thresholds)
        # print(total_below_positive_attribute)
        for threshold_index in range(len(thresholds)):

            thresh = thresholds[threshold_index]

            #print(multiplier)

            total_below = int(math.ceil(thresh - 1) + 1) * (multiplier)
            total_above = total - total_below

            total_below_positive = total_below_positive_attribute[threshold_index]
            total_above_positive = total_positive - total_below_positive


            average_entropy_below = (total_below/total)*calculate_entropy(total_below_positive,total_below) 
            average_entropy_above = (total_above/total)*calculate_entropy(total_above_positive,total_above)
            # print(info_gain)
            info_gain.append(hT - (average_entropy_below + average_entropy_above))

        return info_gain

    return [info_gain(thresholds_x,total_below_positive_x,height), info_gain(thresholds_y,total_below_positive_y,width)]

def find_thresholds():

    x_thresholds, y_thresholds = [],[]

    #X
    previous = [0,actual_class(0,0)]
    true_x = False
    for x in range(width):

        for y in range(height):

            if actual_class(x,y) != previous[1]:
                x_thresholds.append((previous[0] + x) / 2)
                previous = [x,actual_class(x,y)]
                break

            if actual_class(x,y):
                true_x = actual_class(x,y)

            previous = [x,actual_class(x,y)]

    #Y
    previous = [0,actual_class(0,0)]
    true_y = False
    for y in range(height):

        for x in range(width):

            if actual_class(x,y) != previous[1]:
                y_thresholds.append((previous[0] + y) / 2)
                break

            if actual_class(x,y):
                true_y = actual_class(x,y)

            previous = [y,actual_class(x,y)]

    info_gains = find_info_gains(x_thresholds,y_thresholds)

    x_end_case = [width if true_x else 0, 0]
    y_end_case = [height if true_y else 0, 0]

    if len(info_gains[1]) == 0 and len(info_gains[0]) == 0:
        return [x_end_case,y_end_case]
    elif (len(info_gains[1]) == 0):
        return [[x_thresholds[info_gains[0].index(max(info_gains[0]))],max(info_gains[0])],y_end_case]
    elif (len(info_gains[0]) == 0):
        return [x_end_case,[x_thresholds[info_gains[0].index(max(info_gains[0]))],max(info_gains[0])]]

    return [[x_thresholds[info_gains[0].index(max(info_gains[0]))],max(info_gains[0])],[y_thresholds[info_gains[1].index(max(info_gains[1]))],max(info_gains[1])]]

#_______________________________________________________________________________________________#

class node:

    def __init__(self,isLeaf,index,attr):

        self.index = index
        self.children = []
        self.isLeaf = isLeaf
        self.threshold = 0
        self.attribute = attr
        self.tree = [[],[],[]]

    def guess(self,x,y,b):
        value = x if b else y

        if(value < self.threshold):
            if self.children[0].isLeaf:
                return True
            return self.children[0].guess(x,y,not b)
        else:
            if self.children[1].isLeaf:
                return False
            return self.children[1].guess(x,y,not b)

    def evaluate(self,x,y):
        return self.guess(x,y,self.attribute)

    def build(self,target_thresholds, parent_image):
        global image
        global width
        global height

        if(not self.isLeaf):

            x_thresh, x_infogain = target_thresholds[0]
            y_thresh, y_infogain = target_thresholds[1]

            self.attribute = x_infogain >= y_infogain #False is Y True is X
            self.threshold = x_thresh if self.attribute else y_thresh

            left_image = parent_image.crop((0,0,parent_image.size[0],math.ceil(y_thresh - 1) + 1)) #left Y TOP
            right_image = parent_image.crop((0,math.ceil(y_thresh - 1) + 1,parent_image.size[0],parent_image.size[1])) #right Y  BOTTOM

            if self.attribute: #attribite is true if X
                left_image = parent_image5.crop((0,0,math.ceil(x_thresh - 1) + 1,parent_image.size[1]))
                right_image = parent_image.crop((math.ceil(x_thresh - 1) + 1,0,parent_image.size[0],parent_image.size[1]))
            

            #calc new set of thresholds for left node
            image = left_image
            width, height = image.size
            left_thresh = find_thresholds()

            #calc new set of thresholds for right node
            image = right_image
            width, height = image.size
            right_thresh = find_thresholds()

            def is_leaf_calc(image_instance):

                num_pos = 0

                if(image_instance.size[0] != 0 and image_instance.size[1] != 0):
                    for x in range(image_instance.width):
                        for y in range(image_instance.height):
                            pixel = image_instance.getdata()[image_instance.size[0]*y+x]
                            if pixel[0] ==0:
                                return False
                return True

            print(is_leaf_calc(right_image))
            print(is_leaf_calc(left_image))

            right_node = node(is_leaf_calc(right_image),self.index + 1,not self.attribute)
            left_node = node(is_leaf_calc(left_image),self.index + 1,not self.attribute)

            self.children.append(left_node)
            self.children.append(right_node)

            self.children[0].build(left_thresh)
            self.children[1].build(right_thresh)
        
    def stringify(self):

        if(self.index < 2):
            s = "X:" + str(self.threshold) if self.attribute else "Y:" + str(self.threshold)

            if(self.isLeaf):
                # s = "X" if self.attribute else "Y"
                s = "LEAF"

            return s
        return ""

    def print_help(self,current,position):
        
        self.tree[current.index].append(current)

        if current.isLeaf == False:

            self.print_help(current.children[0], "left:" + str(position)) #left
            self.print_help(current.children[1], "right:" + str(position)) #right


    def print_tree(self):
        self.tree = [[],[],[]]
        self.tree[self.index].append([])

        if self.isLeaf == False:
            self.print_help(self.children[0], "left")
            self.print_help(self.children[1], "right")

        length = len(self.stringify())

        for d in self.tree:
            row = ""

            if len(d) == 1:
                row += " "*(length*2) + str(self.stringify()) + " "*(length*2)
            else:
                for element in d:
                    row += (" "*int(length/element.index) + str(element.stringify()) + " "*int(length/element.index))
            print(row)
            print("\n")
            
#_______________________________________________________________________________________________#

images = ["circle.jpg", "shape3.jpg", "mushroom.jpg", "triangle.jpg"]

for i in images:
    print("|" + i.upper() + "|\n")

    set_image(i)
    tree = node(False,0,None)

    tree.build(find_thresholds()) #call on the root node

    tree.print_tree() #makes the tree, needs to be called before evaluate

    predicted_image=Image.new(mode = "RGB",size=(original_image.width,original_image.height),color=(255, 255, 255))
    
    image = original_image

    accuracy = 0

    #image.show()
    for x in range(predicted_image.width):
        for y in range(predicted_image.height):

            if tree.evaluate(x,y):
                predicted_image.putpixel( (x,y), (0,0,0))
                
            if(tree.evaluate(x,y) == True and actual_class(x,y) == True):
                accuracy += 1

    print("Accuracy: " + str(accuracy / (original_image.width * original_image.height)))

    predicted_image.save(i + "_predicted.jpg")

#___________References___________#

#https://stackoverflow.com/questions/138250/how-to-read-the-rgb-value-of-a-given-pixel-in-python

