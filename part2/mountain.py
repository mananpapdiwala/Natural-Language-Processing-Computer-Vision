#!/usr/bin/python
#
# Mountain ridge finder
# Based on skeleton code by D. Crandall, Oct 2016
#


#bayes_net_1b
#
# Just return a ridge with each element in the array representing row with max
# edge strength value
#
#

from PIL import Image
from numpy import *
from scipy.ndimage import filters
from scipy.misc import imsave
import sys
import time

# calculate "Edge strength map" of an image
#
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale,0,filtered_y)
    return filtered_y**2

# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_edge(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range( max(y-thickness/2, 0), min(y+thickness/2, image.size[1]-1 ) ):
            image.putpixel((x, t), color)
    return image

# find row with max strength in given column
#
def find_row_with_max_val_and_total_each_col(x_col):
    x_index = 0
    x_max_value = 0
    x_total_value = 0
    for x_row in range(x_col.shape[0]):
        x_total_value += x_col[x_row]
        if x_col[x_row] > x_max_value:
            x_max_value = x_col[x_row]
            x_index = x_row
    return [x_index, x_total_value]


def bayes_net_1b(x_edge_strength):
    # iterate through rows and return an array with each element in the array representing row with max
    # edge strength value
    x_ridge = []
    x_total_each_col = []
    # Iterating each column
    for x_col in range(x_edge_strength.shape[1]):
        result = find_row_with_max_val_and_total_each_col(x_edge_strength[:, x_col])
        x_ridge.append(result[0])
        x_total_each_col.append(result[1])
    return [x_ridge, x_total_each_col]


def transition_probability(x_no_of_columns):
    # Find transition probability based on gaussian distribution
    x_transition_probability_array = []                     # an array where index represents the row difference between
                                                            # two consecutive columns
    x_standard_deviation = (float(x_no_of_columns)/100.0)
    x_2_variance = 2*x_standard_deviation*x_standard_deviation
    x_sigma_root2_pi = x_standard_deviation*math.sqrt(2.0*math.pi)
    total_p = 0.0
    for x_diff in range(x_no_of_columns):
        x_transition_probability_array.append((math.e**-(x_diff*x_diff/x_2_variance))/x_sigma_root2_pi)
        total_p += x_transition_probability_array[x_diff]
    # print "Transition Probability Array " + str(x_transition_probability_array)
    # print "Total P" + str(total_p)
    return x_transition_probability_array


#  dividing gradient in each cell by the sum of gradients in that column
#
def convert_edge_strength_to_probability_array(x_edge_strength, x_total_each_col):
    x_probability_array = array([[0.0]*x_edge_strength.shape[1]]*x_edge_strength.shape[0])
    for x_row in range(x_edge_strength.shape[0]):
        for x_col in range(x_edge_strength.shape[1]):
            x_probability_array[x_row][x_col] = float(x_edge_strength[x_row][x_col])/float(x_total_each_col[x_col])
    return x_probability_array


def createSamples(x_transition_probability_array, x_probability_of_Si_to_be_at_the_cell, lastSample):
    sample = lastSample
    rows_not_visited = range(len(lastSample))
    for j in range(len(lastSample)):
        probability_of_element_on_a_particular_row = []
        x_index = random.randint(len(rows_not_visited))
        random_column = rows_not_visited[x_index]
        rows_not_visited.pop(x_index)

        max_probability = 0.0
        possible_row_in_this_col_based_on_probability = 0

        for row in range(x_probability_of_Si_to_be_at_the_cell.shape[0]):

            # P(S_i|S_i-1)
            # Probability independent of col - 1 when col - 1 < 0 i.e. out of image
            if random_column - 1 < 0:
                probability_of_element_at_row_given_probability_of_preceding_column_row = 1
            else:
                probability_of_element_at_row_given_probability_of_preceding_column_row = x_transition_probability_array[abs(sample[random_column - 1] - row)]

            # P(S_i+1|S_i)
            # Probability independent of element in col + 1 when col +1 >= last col i.e. out of image
            if random_column + 1 >= len(sample):
                probability_of_element_at_row_given_probability_of_succeeding_column_row = 1
            else:
                probability_of_element_at_row_given_probability_of_succeeding_column_row = x_transition_probability_array[abs(sample[random_column+1] - row)]

            # P(S_i)
            probability_of_element_at_row_in_this_column = x_probability_of_Si_to_be_at_the_cell[row][random_column]

            probability_at_current_row = probability_of_element_at_row_given_probability_of_preceding_column_row * probability_of_element_at_row_given_probability_of_succeeding_column_row * probability_of_element_at_row_in_this_column
            probability_of_element_on_a_particular_row.append(probability_at_current_row)

            if max_probability < probability_at_current_row:
                possible_row_in_this_col_based_on_probability = row
                max_probability = probability_at_current_row

        sample[random_column] = possible_row_in_this_col_based_on_probability
    return sample


def createSamples2(x_transition_probability_array, x_probability_of_Si_to_be_at_the_cell, lastSample):
    sample = lastSample
    rows_not_visited = range(len(lastSample))
    for j in range(len(lastSample)):
        probability_of_element_on_a_particular_row = []
        x_index = 0
        random_column = rows_not_visited[x_index]
        rows_not_visited.pop(x_index)

        max_probability = 0.0
        possible_row_in_this_col_based_on_probability = 0

        for row in range(x_probability_of_Si_to_be_at_the_cell.shape[0]):

            # P(S_i|S_i-1)
            # Probability independent of col - 1 when col - 1 < 0 i.e. out of image
            if random_column - 1 < 0:
                probability_of_element_at_row_given_probability_of_preceding_column_row = 1
            else:
                probability_of_element_at_row_given_probability_of_preceding_column_row = x_transition_probability_array[abs(sample[random_column - 1] - row)]

            # P(S_i)
            probability_of_element_at_row_in_this_column = x_probability_of_Si_to_be_at_the_cell[row][random_column]

            probability_at_current_row = probability_of_element_at_row_given_probability_of_preceding_column_row * probability_of_element_at_row_in_this_column
            probability_of_element_on_a_particular_row.append(probability_at_current_row)

            if max_probability < probability_at_current_row:
                possible_row_in_this_col_based_on_probability = row
                max_probability = probability_at_current_row

        sample[random_column] = possible_row_in_this_col_based_on_probability
    return sample


def findProbability(x_transition_probability_array, x_probability_of_Si_to_be_at_the_cell, lastSample):
    x_probability = 0
    for col in range(x_probability_of_Si_to_be_at_the_cell.shape[1]):
        if col - 1 < 0:
            probability_of_element_at_row_given_probability_of_preceding_column_row = 1
        else:
            probability_of_element_at_row_given_probability_of_preceding_column_row = x_transition_probability_array[abs(lastSample[col - 1] - lastSample[col])]
        probability_of_element_at_row_in_this_column = x_probability_of_Si_to_be_at_the_cell[lastSample[col]][col]
        x_probability += probability_of_element_at_row_given_probability_of_preceding_column_row * probability_of_element_at_row_in_this_column
    return x_probability

start_time = time.time()
print time.asctime(time.localtime(time.time()))
# main program
#
(input_filename, output_filename, gt_row, gt_col, no_of_samples) = sys.argv[1:]

# load in image 
input_image = Image.open(input_filename)

# compute edge strength mask
edge_strength = edge_strength(input_image)
imsave('edges.jpg', edge_strength)

# You'll need to add code here to figure out the results! For now,
# just create a horizontal centered line.
# ridge = [ edge_strength.shape[0]/2 ] * edge_strength.shape[1]
result = bayes_net_1b(edge_strength)
transition_probability_array = transition_probability(edge_strength.shape[0])
probability_of_Si_to_be_at_the_cell = convert_edge_strength_to_probability_array(edge_strength, result[1])


# just create a horizontal centered line.
# ridge = [ edge_strength.shape[0]/2 ] * edge_strength.shape[1]
x_sample = result[0]
sampleList = [x_sample]
for i in range(int(no_of_samples)):
    print str(i)
    x_sample = createSamples(transition_probability_array, probability_of_Si_to_be_at_the_cell, x_sample)
    sampleList.append(x_sample)

result = [sum(x) for x in zip(*sampleList)]
result = [x / int(no_of_samples) for x in result]

print x_sample
print result

f = open('out.txt', 'w')
f.write(str(sampleList))
f.close()


""" max_probability = 0
for i in range(edge_strength.shape[0]):
    x_sample = [i] * edge_strength.shape[1]

    x_sample = createSamples2(transition_probability_array, probability_of_Si_to_be_at_the_cell, x_sample)
    probability_of_x_sample = findProbability(transition_probability_array, probability_of_Si_to_be_at_the_cell,
                                              x_sample)
    print str(i) + " " + str(probability_of_x_sample)
    if probability_of_x_sample > max_probability:
        max_probability = probability_of_x_sample
        finalSample = x_sample
sample = finalSample"""

# output answer
# imsave(output_filename, draw_edge(input_image, result[0], (255, 0, 0), 5))

output_filename = str(time.asctime(time.localtime(time.time())))+"_1_"+str(no_of_samples)+"_"+output_filename
imsave(output_filename, draw_edge(input_image, result, (0, 0, 255), 5))
output_filename = str(time.asctime(time.localtime(time.time())))+"_2_"+str(no_of_samples)+"_"+output_filename
imsave(output_filename, draw_edge(input_image, x_sample, (0, 255, 0), 5))

end_time = time.time()
print end_time - start_time