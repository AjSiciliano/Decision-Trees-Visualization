from PIL import Image
import math
#_______________________________________________________________________________________________#

image = Image.new('RGB', (0,0))
width, height = 0,0

def set_image(file_name):
    global image, width, height
    image = Image.open(file_name, 'r').convert('RGB')
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
    
    print(max(info_gains[0]))
    print(max(info_gains[1]))

    return [x_thresholds[info_gains[0].index(max(info_gains[0]))],y_thresholds[info_gains[1].index(max(info_gains[1]))]]
    
#_______________________________________________________________________________________________#

set_image("shape3.jpg")

print(find_thresholds())

set_image("circle.jpg")

print(find_thresholds())

set_image("circle2.jpg")

print(find_thresholds())






#___________References___________#

#https://stackoverflow.com/questions/138250/how-to-read-the-rgb-value-of-a-given-pixel-in-python


