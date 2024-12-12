import cv2
import numpy as np
from scipy.spatial import distance
import time
import argparse

# to call performance test
parser = argparse.ArgumentParser()
parser.add_argument('--performanceTest', type=int, help='runs performance test.',default=0)
args = parser.parse_args()

# preprocess image and extract contour Points
def preprocess_points(image_path): 
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV) # 200 threshold value gave best accuracy results
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found.")
        return None
    max_contour = max(contours, key=cv2.contourArea)    
    points = np.squeeze(max_contour)  # extract points
    return points

# graham Scan Implementation
def graham_scan(points): 
    # find the point with the lowest Y-coordinate
    lowest_point = min(points, key=lambda p: (p[1], p[0]))

    # sort points by polar angle to the lowest point
    def polar_angle(p):
        dx, dy = p[0] - lowest_point[0], p[1] - lowest_point[1]
        return np.arctan2(dy, dx)

    sorted_points = sorted(points, key=polar_angle)

    # build the convex hull
    hull = []
    for p in sorted_points:
        while len(hull) >= 2 and cross_product(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)
    
    return hull

def cross_product(o, a, b): 
    """Calculate the cross product of vectors OA and OB.
    A positive value indicates a left turn, negative indicates a right turn, and 0 indicates collinear."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

# compute training Convex Hull
def compute_reference_hull(training_image): 
    hulls = []
    #for path in training_image:
    points = preprocess_points(training_image)
    if points is not None:
        hull = graham_scan(points)
        hulls.append(hull)
    if len(hulls)==1 :
        return hulls[0]
    else:
        print("No valid hulls found in reference images.")
        return None

# compare Convex Hulls
def compare_hulls(input_hull, reference_hull, threshold=50): 
    if len(input_hull) != len(reference_hull): #hull have unequal amount of sets so imterpolate
        target_size = max(len(input_hull), len(reference_hull))
        input_hull = interpolate_hull(input_hull, target_size)
        reference_hull = interpolate_hull(reference_hull, target_size)

    # compute point distances
    distances = [distance.euclidean(p1, p2) for p1, p2 in zip(input_hull, reference_hull)]
    avg_distance = sum(distances) / len(distances)
    if avg_distance>450:
        return "invalid"  #should filter out should absurd inputs if for some reason theres a few contours detected
    return avg_distance < threshold

#secondary check incase first check fails, but is less reliable
def bottom_distance(input_hull,yes_train,no_train): 
    input_hull = np.array(input_hull)
    yes_train = np.array(yes_train)
    no_train = np.array(no_train)

    # identify the bottom right point (point with max x + y)
    bottom_right_input = input_hull[np.argmax(input_hull[:, 0] + input_hull[:, 1])]
    bottom_right_yes = yes_train[np.argmax(yes_train[:, 0] + yes_train[:, 1])]
    bottom_right_no = no_train[np.argmax(no_train[:, 0] + no_train[:, 1])]

    # identify the rightmost point (point with max x)
    rightmost_input = input_hull[np.argmax(input_hull[:, 0])]
    rightmost_yes = yes_train[np.argmax(yes_train[:, 0])]
    rightmost_no = no_train[np.argmax(no_train[:, 0])]

    # calculate distances between bottom right and rightmost points
    input_distance = np.linalg.norm(bottom_right_input - rightmost_input)
    yes_distance = np.linalg.norm(bottom_right_yes - rightmost_yes)
    no_distance = np.linalg.norm(bottom_right_no - rightmost_no)

    # compare distances
    mse_yes = (input_distance - yes_distance) ** 2
    mse_no = (input_distance - no_distance) ** 2

    # classify based on smaller MSE
    if mse_yes < mse_no:
        return "The image is possibly a 'yes'"
    else:
        return "The image is possibly a 'no'"

def interpolate_hull(hull, target_size): 
    """Interpolate points of a hull to ensure equal number of points."""
    interpolated_hull = []
    step = len(hull) / target_size
    for i in range(target_size):
        interpolated_hull.append(hull[int(i * step)])
    return interpolated_hull

def timing_test():
    total_time =time.time()
    training_yes = './testing_data/yes_2.jpg'
    training_no = './testing_data/no_2.jpg'
    training_yes_hull=compute_reference_hull(training_yes)
    training_no_hull = compute_reference_hull(training_no)
    i=1
    yes_accuracy = 0
    no_accuracy = 0
    invalid_accuracy = 0
    accuracy = 0
    for i  in range(25):
        start_time = time.time()
        image_path = f"./performance_test_input/input_{i}.jpg"
        image_points = preprocess_points(image_path)
        if image_points is not None:
            image_hull = graham_scan(image_points)
            if training_yes_hull and compare_hulls(image_hull, training_yes_hull):
                if (i<10):
                    accuracy=accuracy+1
                    yes_accuracy=yes_accuracy+1
                    print("correct classifcation of yes gesture")
            elif training_no_hull and compare_hulls(image_hull, training_no_hull):
                if (i>9 and i<20):
                    accuracy=accuracy+1
                    no_accuracy=no_accuracy+1
                    print("correct classifcation of no gesture")
            else:
                if (i>19):
                    accuracy=accuracy+1
                    invalid_accuracy=invalid_accuracy+1
                    print("correct classifcation of invalid gesture")
            # if other check fails runs less accurate secondary check
                else: 
                    result = bottom_distance(image_hull,training_yes_hull,training_no_hull)
                    if (i<10) and (result == "The image is possibly a 'yes'"):
                        accuracy=accuracy+1
                        yes_accuracy=yes_accuracy+1
                        print("correct classifcation of yes gesture")
                    elif (i>9 and i<20) and (result == "The image is possibly a 'no'"):
                        accuracy=accuracy+1
                        no_accuracy=no_accuracy+1
                        print("correct classifcation of no gesture")
        else:
            if (i>19):
                accuracy=accuracy+1
                invalid_accuracy=invalid_accuracy+1
                print("correct classifcation of invalid gesture")
            print("Failed to extract input points.")
            print("failed to classify gesture")
        end_time = time.time()
        print(f"Execution Time: {end_time - start_time:.4f} seconds")
    print("accuracy is ",accuracy/25*100,"%")
    print("the yes gesture accuracy was ",yes_accuracy/10*100,"%")
    print("the no gesture accuracy was ",no_accuracy/10*100,"%")
    print("the invalid gesture accuracy was ",invalid_accuracy/5*100,"%")
    over_time = time.time()
    print(f"the total time to run was {over_time-total_time:.4f} seconds")

# to control if you want to test timing or check contours
timing = args.performanceTest
if timing == 1:
    timing_test()
else:
    # training cata
    training_yes = './testing_data/yes_2.jpg'
    training_no = './testing_data/no_2.jpg'
    input_image = './testing_data/no_1.jpg'
    # computing the training datas convex hulls
    training_yes_hull=compute_reference_hull(training_yes)
    training_no_hull = compute_reference_hull(training_no)
    # now to process input image
    input_points = preprocess_points(input_image)
    if input_points is not None:
        input_hull = graham_scan(input_points)
        image = cv2.imread(input_image)
        # to give a visualize of the convex hull
        cv2.polylines(image, [np.array(input_hull)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.imshow("Threshold Image", image)
        cv2.waitKey(0) == ord('q') # press 'q' to quit
        cv2.destroyAllWindows()
    # compare input hull with reference hull
        if training_yes_hull and compare_hulls(input_hull, training_yes_hull):
            print("The gesture matches the 'Yes' symbol!")
        elif training_no_hull and compare_hulls(input_hull, training_no_hull):
            print("The gesture matches the 'No' symbol!") 
        else:
            # if other check fails runs less accurate secondary check
            print(bottom_distance(input_hull,training_yes_hull,training_no_hull))
            print("or the gesture is invalid")
    else:
        print("Failed to extract input points.")




