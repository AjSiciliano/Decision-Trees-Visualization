from PIL import Image
import numpy as np
from termcolor import colored

from treelib import Node as printed_node
from treelib import Tree as printed_tree

from sklearn.model_selection import train_test_split
from node import *
from functions import *

#Authors: Andrew Siciliano, Christopher Wu
#Simple C4.5 Implementation For Decision Tree
#Uses Miroslov Kubat's 'Introduction to Machine Learning' as a reference

verbose = True
print_tree = False

def printer(text, color=None,end=None): 
    if verbose:
        if color != None and end == None:
            print(colored(text, color))
        elif color != None and end != None:
            print(colored(text, color),end=end)
        else:
            print(text)

#_____________________________________________________________________________________________________#

images = ["2rect", "abstract","triangle","ovals","halloween"]

printer("\nRunning Image Tests.....\n")

#Driver, A bit Redundant but OKAY!
for i in images:

    printer("______________________________________  Training "+i+".jpg" + " ______________________________________\n")

    init_data = build_data_from_image("test_images/"+i+".jpg")

    attr,classif = data_to_splittable(init_data)

    x_train,x_test,y_train,y_test=train_test_split(attr,classif,test_size=0.3,stratify=classif)
    
    training_data = revert_slittable(x_train,y_train)
    testing_data = revert_slittable(x_test,y_test)

    training_thresholds = find_thresholds(training_data)

    tree = node(False,0,None)
    tree.build(training_thresholds,training_data) #call on the root node

    predicted_image=Image.new(mode="RGB",size=(len(init_data),len(init_data[1])),color=(255,255,255))

    total = len(testing_data)
    total_positive,total_negative,accuracy_total,accuracy_positive,accuracy_negative=0,0,0,0,0

    for element in testing_data:
        x = element[0]
        y = element[1]
        clss = element[2]

        if tree.evaluate(x,y):
            predicted_image.putpixel((x,y),(0,0,0))

        if(clss):
            total_positive += 1
            if(tree.evaluate(x,y)): accuracy_positive += 1
        else:
            total_negative += 1
            if(not tree.evaluate(x,y)): accuracy_negative += 1

        if(tree.evaluate(x,y) == clss):
            accuracy_total += 1

    img_path = get_subdirectory("predicted_images")+"/"+i+"_predicted.jpg"

    accuracy_total /= total
    accuracy_positive /= total_positive
    accuracy_negative /= total_negative

    printer("Accuracies: "+i+".jpg\n","green")

    printer("Total Accuracy: " + str(accuracy_total * 100))
    printer("Positive Accuracy: " + str(accuracy_positive * 100))
    printer("Negative Accuracy: " + str(accuracy_negative * 100))

    predicted_image.save(img_path)

    printer("\nSaved un_pruned image in path: " + img_path +"\n")

    tree_lib_tree = tree.printify()
    tree_lib_tree.save2file(get_subdirectory("tree_texts/")+i+'_full_tree.txt')

    tree_lib_tree.show(line_type="ascii-em") if print_tree == True else None

    printer("\nSaved tree as text output in path: " + "tree_texts/"+i+"_full_tree.txt" +"\n")

    tree.build_pruned_values(training_data)

    pruned_ids_list = []

    if(i != "halloween"):
        printer("Generating Pruned Gif and Images.....\n", "red")

        printer("Now Initiliazing Parameters for the Pruned Trees .....")
        for x in range(35):
            if (x != 0):
                printer("Progress: " + str(100*(x/35)//1 - .01) + "% \r","green","")
            else:
                printer("Progress: " + str(0.0) + "% \r","green","")
            pruned_ids_list.append([x,tree.get_prune_ids((x/100),testing_data)])

        printer("Done Initiliazing Parameters!\n", "green")
        printer("Now creating sequence of images.....")

        pruned_images = []
        first_path = ""

        for c in pruned_ids_list:
            pruned_image = Image.new(mode="RGB",size=(len(init_data),len(init_data[1])),color=(255,255,255))

            pruned_image_path = get_subdirectory(get_subdirectory("prunings") +"/"+ i+"_pruned")+"/c_" + str(c[0]) + ".jpg"

            if first_path == "":
                first_path = pruned_image_path

            for element in testing_data:
                if tree.evaluate(element[0],element[1],None,c[1]):
                    pruned_image.putpixel((element[0],element[1]),(0,0,0))

            pruned_image.save(pruned_image_path)
            pruned_images.append(np.asarray(Image.open(pruned_image_path)))

        printer("\nSaved pruned images in diectory : prunings/"+i+"_pruned\n")

        the_gif_format = [Image.fromarray(img) for img in pruned_images]

        first = Image.open(first_path)

        first.save(get_subdirectory("gifs")+"/" + i + '_pruned_animation.gif', save_all=True, append_images=the_gif_format,optimize=False, duration=200, loop=0)

        printer("Saved pruned image gif in path: " + i + '_pruned_animation.gif' +"\n")

    printer("Running Noisey Version....\n", "red")
    
    init_data = build_data_from_image("noisey_data/"+i+".jpg")

    attr,classif = data_to_splittable(init_data)

    x_train,x_test,y_train,y_test=train_test_split(attr,classif,test_size=0.3,stratify=classif)
    
    training_data = revert_slittable(x_train,y_train)
    testing_data = revert_slittable(x_test,y_test)

    training_thresholds = find_thresholds(training_data)

    tree = node(False,0,None)
    tree.build(training_thresholds,training_data) #call on the root node

    predicted_image=Image.new(mode="RGB",size=(len(init_data),len(init_data[1])),color=(255,255,255))

    total = len(testing_data)
    total_positive,total_negative,accuracy_total,accuracy_positive,accuracy_negative=0,0,0,0,0

    for element in testing_data:
        x = element[0]
        y = element[1]
        clss = element[2]

        if tree.evaluate(x,y):
            predicted_image.putpixel((x,y),(0,0,0))

        if(init_data[x][y]):
            total_positive += 1
            if(tree.evaluate(x,y)): accuracy_positive += 1
        else:
            total_negative += 1
            if(not tree.evaluate(x,y)): accuracy_negative += 1

        if(tree.evaluate(x,y) == init_data[x][y]):
            accuracy_total += 1

    img_path = get_subdirectory("predicted_noisey_images")+"/"+i+"_predicted.jpg"

    accuracy_total /= total
    accuracy_positive /= total_positive
    accuracy_negative /= total_negative

    printer("Accuracies Noisey: "+i+".jpg\n","green")

    printer("Total Accuracy: " + str(accuracy_total * 100))
    printer("Positive Accuracy: " + str(accuracy_positive * 100))
    printer("Negative Accuracy: " + str(accuracy_negative * 100))

    predicted_image.save(img_path)

    printer("\nSaved un_pruned image in path: " + img_path +"\n")

    tree_lib_tree = tree.printify()
    tree_lib_tree.save2file(get_subdirectory("tree_texts/")+i+'_full_noisey_tree.txt')

    tree_lib_tree.show(line_type="ascii-em") if print_tree == True else None

    printer("\nSaved tree as text output in path: " + "tree_texts/"+i+"_full_noisey_tree.txt" +"\n")

    printer("Finished with " + i + ".jpg","magenta")

printer("___________________________________________________________________________________________________________\n")

printer("Done Woot Woot!!\n","magenta")

#___________References___________#
#https://stackoverflow.com/questions/34012886/print-binary-tree-level-by-level-in-python
#https://stackoverflow.com/questions/138250/how-to-read-the-rgb-value-of-a-given-pixel-in-python
#https://stackoverflow.com/questions/65340769/getting-attributeerror-im-must-have-seek-method-while-trying-to-save-as-gif
#https://www.codegrepper.com/code-examples/python/python+create+directory+if+not+exists


