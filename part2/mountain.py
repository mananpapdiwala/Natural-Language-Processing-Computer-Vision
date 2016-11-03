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


def find_max_row(x_col):
    # find row with max strength in given column
    x_index = 0
    x_max_value = 0
    x_total_value = 0
    for x_row in range(x_col.shape[0]):
        if x_col[x_row] > x_max_value:
            x_max_value = x_col[x_row]
            x_index = x_row
    return x_index


def bayes_net_1b(x_edge_strength):
    # iterate through rows and return an array with each element in the array representing row with max
    # edge strength value
    x_ridge = []
    # Iterating each column
    for col in range(x_edge_strength.shape[1]):
        row_with_max_value_for_this_column = find_max_row(x_edge_strength[:, col])
        x_ridge.append(row_with_max_value_for_this_column)
    print len(x_ridge)
    print x_edge_strength.shape
    return x_ridge










# main program
#
(input_filename, output_filename, gt_row, gt_col) = sys.argv[1:]

# load in image 
input_image = Image.open(input_filename)

# compute edge strength mask
edge_strength = edge_strength(input_image)
imsave('edges.jpg', edge_strength)




# You'll need to add code here to figure out the results! For now,
# just create a horizontal centered line.
#ridge = [ edge_strength.shape[0]/2 ] * edge_strength.shape[1]
ridge = bayes_net_1b(edge_strength)
print ridge

# output answer
imsave(output_filename, draw_edge(input_image, ridge, (255, 0, 0), 5))
