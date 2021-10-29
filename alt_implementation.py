from PIL import Image
import math


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree
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

def build_data():
    global image
    global data

    data = []*(image.width*image.height)
    data_class = []*(image.width*image.height)
    for x in range(image.width):
        for y in range(image.height):
            data.append([x,y])
            data_class.append(actual_class(x,y))

    return [data,data_class]

#_______________________________________________________________________________________________#
images = ["circle", "shape3", "mushroom", "triangle", "noisey_circle", "noisey_shape3", "noisey_mushroom","noisey_triangle"]

for i in images:
    print("|"+i+"|\n")

    set_image(i + ".jpg")

    X, y = build_data()

    data_feature_names = ['X','Y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    path= DecisionTreeClassifier(random_state=1).cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=1,ccp_alpha=ccp_alpha)
    #print(help(clf))
        clf.fit(X_train, y_train)
        clfs.append(clf)

    for c in clfs:
        #each one these C's is a DecisionTreeClassifier
        #get accurcacies from decision tree classifier


        # predicted_image=Image.new(mode = "RGB",size=(original_image.width,original_image.height),color=(255, 255, 255))
    
        # image = original_image

        # for x in range(predicted_image.width):
        #     for Y in range(predicted_image.height):

        #         if c.predict([[x,Y]])[0]:
        #             predicted_image.putpixel( (x,Y), (0,0,0))

        # predicted_image.save(i + "/alpha_" + str(c.ccp_alpha) + "_predicted.jpg")

        print("_"*50 + "alpha:" + str(c.ccp_alpha) + "_"*50)
        y_pred = c.predict(X_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        
        print("_"*(100+len(str(c.ccp_alpha))))
        


#___________References___________#
#https://www.tutorialspoint.com/scikit_learn/scikit_learn_decision_trees.htm

#https://www.datacamp.com/community/tutorials/decision-tree-classification-python
#https://stackabuse.com/decision-trees-in-python-with-scikit-learn/

#https://towardsdatascience.com/visualizing-decision-trees-with-python-scikit-learn-graphviz-matplotlib-1c50b4aa68dc

#https://medium.com/swlh/post-pruning-decision-trees-using-python-b5d4bcda8e23










