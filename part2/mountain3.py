#!/usr/bin/python
#
# Mountain ridge finder
# Based on skeleton code by D. Crandall, Oct 2016
#


# bayes_net_1b
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
    # Tried gaussian but the below one worked better
    """x_standard_deviation = (float(x_no_of_columns)/100.0)
    x_2_variance = 2*x_standard_deviation*x_standard_deviation
    x_sigma_root2_pi = x_standard_deviation*math.sqrt(2.0*math.pi)
    total_p = 0.0
    for x_diff in range(x_no_of_columns):
        x_transition_probability_array.append((math.e**-(x_diff*x_diff/x_2_variance))/x_sigma_root2_pi)
        total_p += x_transition_probability_array[x_diff]"""
    factor = 2
    for x_diff in range(x_no_of_columns):

        x_result = float(1.0/(float(x_diff+1)))
        x_transition_probability_array.append(x_result)
    # print "Transition Probability Array " + str(x_transition_probability_array)
    # print "Total P" + str(total_p)
    return x_transition_probability_array


#  dividing gradient in each cell by the sum of gradients in that column
#
def convert_edge_strength_to_probability_array(x_edge_strength, x_total_each_col, given_col, given_row):
    x_total_of_total_each_col = 0
    for i in x_total_each_col:
        x_total_of_total_each_col += i
    x_probability_array = array([[0.0]*x_edge_strength.shape[1]]*x_edge_strength.shape[0])
    x_probability_array_with_given_data = array([[0.0] * x_edge_strength.shape[1]] * x_edge_strength.shape[0])
    for x_row in range(0, int(math.floor(x_edge_strength.shape[0]/2))):
        for x_col in range(x_edge_strength.shape[1]):
            x_probability_array[x_row][x_col] = 15*float(x_edge_strength[x_row][x_col])/float(x_total_of_total_each_col)
            if(abs(given_row - x_row) == 50):
                x_probability_array_with_given_data[x_row][x_col] = 15*float(x_edge_strength[x_row][x_col])/float(x_total_of_total_each_col)
            else:
                x_probability_array_with_given_data[x_row][x_col] = float(x_edge_strength[x_row][x_col]) / float(
                    x_total_of_total_each_col)
    for x_row in range(int(math.floor(x_edge_strength.shape[0]/2+1)), x_edge_strength.shape[0]):
        for x_col in range(x_edge_strength.shape[1]):
            x_probability_array[x_row][x_col] = float(x_edge_strength[x_row][x_col]) / float(x_total_of_total_each_col)
            if (abs(given_row - x_row) == 50):
                x_probability_array_with_given_data[x_row][x_col] = 15 * float(x_edge_strength[x_row][x_col]) / float(
                    x_total_of_total_each_col)
            else:
                x_probability_array_with_given_data[x_row][x_col] = float(x_edge_strength[x_row][x_col]) / float(
                    x_total_of_total_each_col)
    return [x_probability_array, x_probability_array_with_given_data]


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
            #probability_at_current_row = probability_of_element_at_row_given_probability_of_preceding_column_row * probability_of_element_at_row_in_this_column
            probability_of_element_on_a_particular_row.append(probability_at_current_row)

            if max_probability < probability_at_current_row:
                possible_row_in_this_col_based_on_probability = row
                max_probability = probability_at_current_row

        sample[random_column] = possible_row_in_this_col_based_on_probability
    return sample


def createSamples2(x_transition_probability_array, x_probability_of_Si_to_be_at_the_cell, x_column, ridge_b_1):

    for i in range(x_column + 1, len(ridge_b_1)):
        probability_of_element_on_a_particular_row = []
        max_probability = 0
        for row in range(x_probability_of_Si_to_be_at_the_cell.shape[0]):

            # P(S_i|S_i-1)
            # Probability independent of col - 1 when col - 1 < 0 i.e. out of image
            if i <= 0:
                probability_of_element_at_row_given_probability_of_preceding_column_row = 1
            else:
                #print str(abs(ridge_b_1[i - 1] - row)) + " " + str(len(x_transition_probability_array))
                probability_of_element_at_row_given_probability_of_preceding_column_row = x_transition_probability_array[abs(ridge_b_1[i - 1] - row)]

            # P(S_i)
            probability_of_element_at_row_in_this_column = x_probability_of_Si_to_be_at_the_cell[row][i]

            probability_at_current_row = probability_of_element_at_row_given_probability_of_preceding_column_row * probability_of_element_at_row_in_this_column
            probability_of_element_on_a_particular_row.append(probability_at_current_row)

            if max_probability < probability_at_current_row:
                possible_row_in_this_col_based_on_probability = row
                max_probability = probability_at_current_row

        ridge_b_1[i] = possible_row_in_this_col_based_on_probability

    for i in range(x_column - 1, -1, -1):
        probability_of_element_on_a_particular_row = []
        max_probability = 0
        for row in range(x_probability_of_Si_to_be_at_the_cell.shape[0]):

            # P(S_i-1|S_i)
            # Probability independent of col - 1 when col - 1 < 0 i.e. out of image
            if i-1 < 0:
                probability_of_element_at_row_given_probability_of_succeding_column_row = 1
            else:
                probability_of_element_at_row_given_probability_of_succeding_column_row = \
                x_transition_probability_array[abs(ridge_b_1[i+1] - row)]

            # P(S_i)
            probability_of_element_at_row_in_this_column = x_probability_of_Si_to_be_at_the_cell[row][i]

            probability_at_current_row = probability_of_element_at_row_given_probability_of_succeding_column_row * probability_of_element_at_row_in_this_column
            probability_of_element_on_a_particular_row.append(probability_at_current_row)

            if max_probability < probability_at_current_row:
                possible_row_in_this_col_based_on_probability = row
                max_probability = probability_at_current_row

        ridge_b_1[i] = possible_row_in_this_col_based_on_probability
    return ridge_b_1


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


def findMaxFrequency(x_sampleList, x_no_of_rows):
    x_ridge = []
    for col in range(len(x_sampleList[0])):
        max_frequency = 0
        max_frequency_row = 0
        x_frequency_count = [0] * x_no_of_rows
        for sample_no in range(len(x_sampleList)):
            x_frequency_count[x_sampleList[sample_no][col]] += 1
            if x_frequency_count[x_sampleList[sample_no][col]] > max_frequency:
                max_frequency = x_frequency_count[x_sampleList[sample_no][col]]
                max_frequency_row = x_sampleList[sample_no][col]
        x_ridge.append(max_frequency_row)
    return x_ridge


start_time = time.time()
print time.asctime(time.localtime(time.time()))
# main program
#
(input_filename, output_filename, gt_row, gt_col) = sys.argv[1:]

# load in image
input_image = Image.open(input_filename)

# compute edge strength mask
edge_strength = edge_strength(input_image)
imsave(str(input_filename)+'edges.jpg', edge_strength)

# You'll need to add code here to figure out the results! For now,
# just create a horizontal centered line.
# ridge = [ edge_strength.shape[0]/2 ] * edge_strength.shape[1]
result = bayes_net_1b(edge_strength)
transition_probability_array = transition_probability(edge_strength.shape[0])
probability_of_Si_to_be_at_the_cell = convert_edge_strength_to_probability_array(edge_strength, result[1], int(gt_col), int(gt_row))


# just create a horizontal centered line.
# ridge = [ edge_strength.shape[0]/2 ] * edge_strength.shape[1]
x_ridge_part1 = result[0]
sampleList = []

for i in range(0, edge_strength.shape[1], 4):
    print str(i)
    x_sample = createSamples2(transition_probability_array, probability_of_Si_to_be_at_the_cell[0], i, x_ridge_part1)
    sampleList.append(x_sample)

x_ridge_part2 = findMaxFrequency(sampleList, edge_strength.shape[0])

print output_filename
probability_of_Si_to_be_at_the_cell[1][int(gt_row)][int(gt_col)] *= 200

sampleList = []
for i in range(0, edge_strength.shape[1], 4):
    if i < 0 or i > int(gt_col)+30:
        continue
    print str(i)
    x_sample = createSamples2(transition_probability_array, probability_of_Si_to_be_at_the_cell[1], i, x_ridge_part1)
    sampleList.append(x_sample)

x_ridge_part3 = findMaxFrequency(sampleList, edge_strength.shape[0])

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
end_time = time.time()
# output answer
# imsave(output_filename, draw_edge(input_image, result[0], (255, 0, 0), 5))


#output_filename2 = input_filename + str(time.strftime("%a, %d %H-%M-%S", time.localtime(time.time())))+"_2_"+str(end_time - start_time)+"_"+output_filename
imsave("1"+output_filename, draw_edge(input_image, x_ridge_part1, (255, 0, 0), 5))
input_image1 = Image.open("1"+output_filename)
imsave("2"+output_filename, draw_edge(input_image1, x_ridge_part2, (0, 0, 255), 5))
input_image2 = Image.open("2"+output_filename)
imsave("3"+output_filename, draw_edge(input_image2, x_ridge_part3, (0, 255, 0), 5))


#output_filename = input_filename + str(time.strftime("%a, %d %H-%M-%S", time.localtime(time.time())))+"_1_"+str(end_time - start_time)+"_"+output_filename2
#imsave(output_filename, draw_edge(input_image, x_ridge_part1, (255, 0, 0), 5))
#output_filename = str(time.asctime(time.localtime(time.time())))+"_2_"+str(no_of_samples)+"_"+output_filename
#imsave(output_filename, draw_edge(input_image, x_sample, (0, 255, 0), 5))


print end_time - start_time