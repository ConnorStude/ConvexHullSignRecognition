import cv2
import numpy as np
import time
import argparse

# to call performance test
parser = argparse.ArgumentParser()
parser.add_argument('--performanceTest', type=int, help='runs performance test.',default=0)
args = parser.parse_args()

def naive_pixel_comparison(image,time):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)  # thresholding

    # calculate the percentage of white pixels (indicating fingers)
    white_pixel_ratio = np.sum(thresh == 255) / thresh.size

    if time == 0:
        cv2.imshow("Thresholded Image", thresh)
        cv2.waitKey(0)==ord('q')
        cv2.destroyAllWindows()

    if white_pixel_ratio == 1 or white_pixel_ratio <0.01: #to filter out invalid images of just blank images
        print("Invalid gesture")
        return "Invalid"
    
    elif white_pixel_ratio > 0.4:  # mostly white area (open fingers)
        print("yes gesture detected.")
        return "Yes"
    
    else:  # more dark space (likely two fingers)
        print("No gesture detected.")
        return "No"

# for performance test
def timing_test():
    total_time =time.time()
    i=1
    yes_accuracy = 10
    no_accuracy = 10
    invalid_accuracy = 5
    accuracy = 25
    for i  in range(25):
        start_time = time.time()
        image_path = f"./performance_test_input/input_{i}.jpg"
        image = cv2.imread(image_path)
        result = naive_pixel_comparison(image,1)
        if i < 10 and (result!="Yes"):
            accuracy=accuracy-1
            yes_accuracy=yes_accuracy-1
            print("incorrect classification for yes gesture")
        if (i < 20 and i>9) and (result != "No"):
            accuracy=accuracy-1
            no_accuracy=no_accuracy-1
            print("incorrect classifcation for No gesture")
        if i>19 and result!="Invalid":
            accuracy=accuracy-1
            invalid_accuracy=invalid_accuracy-1
            print("incorrect classifcation for invalid gesture") 
        end_time = time.time()
        print(f"Execution Time: {end_time - start_time:.4f} seconds")
    print("accuracy is ",accuracy/25*100,"%")
    print("the yes gesture accuracy was ",yes_accuracy/10*100,"%")
    print("the no gesture accuracy was ",no_accuracy/10*100,"%")
    print("the invalid gesture accuracy was ",invalid_accuracy/5*100,"%")
    over_time = time.time()
    print(f"the total time to run was {over_time-total_time:.4f} seconds")

# main either calls unit tests to performance test
if args.performanceTest == 0:

    image = cv2.imread("./testing_data/yes_3.jpg")  # replace with your unit test image
    if image is None:
        print("Error: Image not found or unable to read the file. Check the file path or format.")
        exit() 
    else:
        print("Image successfully loaded.")
    result = naive_pixel_comparison(image,0)
    print(f"Detected gesture: {result}")
    cv2.imshow("Naive Pixel Comparison", image)
    cv2.waitKey(0) == ord('q') # press 'q' to quit
    cv2.destroyAllWindows()

#run timing tracking for performance
if args.performanceTest == 1:
    timing_test()

