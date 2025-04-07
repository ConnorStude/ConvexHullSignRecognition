All information about this project can be found in the cs450-final-report.pdf file located in this repository

To run the unit tests for either the Convex Hull or Pixel Comparison run the program by entering
"python ConvexHull.py" or "python PixelComparison.py" this will run a unit test and output the results 
into the console. To change the unit test open the "./testing_data" folder and determine the image you 
would like to test. In the ConvexHull algorithm locate line 186 and change the input image to the desired
file for the unit test, or line 72 for the PixelComparison algorithm. IMPORTANT when running the unit tests
a pop-up window showing the image, thresholding or Convex Hull will appear, to close them quickly press 'q'
on your keyboard.

To run the performance test add the tag "performanceTest 1" in the console. This automatically runs the 
performance test and outputs the results into the console.
"python PixelComparision.py --performanceTest 1"
