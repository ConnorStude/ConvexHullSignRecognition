# ASL Classification with Graham Scan and Pixel Comparison

## Summary

This project implements two algorithms to classify the American Sign Language (ASL) gestures **yes** (fist) and **no** (pinch) on a white background:

* **Pixel Comparison**: A naive approach that counts white pixels in a thresholded image. Fist (yes) yields more pixels; pinch (no) fewer due to the thumb–index gap.
* **Convex Hull (Graham Scan)**: A robust method that computes the convex hull of hand contours using Graham Scan and compares it to training hulls. If uncertain, a secondary distance-based test on the rightmost hull point is applied.

Performance was measured in terms of accuracy and execution time on a test set of yes, no, and invalid images.

## 1. Algorithm Selected

### 1.1 Pixel Comparison

A simple, single-pass pixel-counting algorithm:

```pseudo
Input: Grayscale image of hand on white background
Result: Classification (yes, no, invalid)

begin
 1. Convert image to grayscale
 2. Threshold to binary
 3. Count white pixels
 4. If count > threshold_yes then return yes
    else if count < threshold_no then return no
    else return invalid
end
```

**Time Complexity:** O(n), where n = total pixels.

### 1.2 Convex Hull (Graham Scan)

Builds convex hulls and compares shapes:

```pseudo
Input: Binary image of hand on white background
Result: Classification (yes, no, invalid, lean_yes, lean_no)

begin
 1. Preprocess training images:
    a. Blur, threshold, find contours
    b. Run Graham Scan to compute hulls
 2. For input image:
    a. Blur, threshold, find contours
    b. Compute convex hull via Graham Scan
 3. Compare input hull to training hulls
 4. If comparison inconclusive:
    a. Measure distance from rightmost hull point
    b. Classify based on distance
end
```

#### Graham Scan Pseudocode

```pseudo
Input: Contour points
Result: Stack of hull points

begin
 1. Sort points by y-coordinate (lowest first)
 2. Let start = points[0]
 3. Sort remaining points by polar angle about start
 4. Initialize stack with [start, points[1], points[2]]
 5. For each point p in points[3:]:
    while stack has ≥2 points and cross(stack[-2], stack[-1], p) ≤ 0:
      pop stack
    push p onto stack
 6. Return stack
end
```

**Time Complexity:** O(n log n) due to sorting; hull construction is O(n).

## 2. Implementation

Both algorithms are implemented in **Python** with [OpenCV](https://opencv.org/) and NumPy:

* **Preprocessing**: Grayscale conversion, Gaussian blur, binary thresholding (cv2.threshold).
* **Contour Detection**: cv2.findContours to extract hand boundary points.
* **Graham Scan**: Custom implementation using NumPy arrays for polar-angle sorting and cross-product calculations.
* **Pixel Comparison**: NumPy array operations to count white pixels.

**Unit Tests**: See Table 1 in the report for cases including blank/invalid images, varying lighting, off-center gestures.

## 3. Performance Tests

Run a performance test on 10 yes, 10 no, and 5 invalid images:

```bash
python ConvexHull.py --performanceTest 1
# or
python PixelComparison.py --performanceTest 1
```

Results were plotted externally (Excel) and summarized below.

## 4. Evaluation Results

| Gesture Type | Pixel Comparison Accuracy | Convex Hull Accuracy |
| ------------ | ------------------------: | -------------------: |
| Yes          |                       80% |                  60% |
| No           |                      100% |                  80% |
| Invalid      |                       60% |                  80% |

The Pixel Comparison ran roughly **2× faster** than Convex Hull under ideal conditions due to its O(n) vs. O(n log n) complexity.

## 5. Reflection

While Convex Hulls provide robustness to background noise and shape variance, a simple Pixel Comparison can outperform in controlled settings. In real-world scenarios with artifacts, hull-based methods (or modern ML approaches) generally yield higher reliability. Future work could include:

* Expanding training data for better hull matching.
* Implementing interpolation refinements in hull comparison.
* Exploring deep learning (e.g., TensorFlow) for end-to-end ASL classification.

## 6. Resources

* Birchfield, S. T. *Image Processing and Analysis*, CENGAGE, 2017.
* Shih, F. Y. *Image Processing and Pattern Recognition*, Wiley, 2010.
* Graham, R. L. *Finding the Convex Hull of a Simple Polygon*, Stanford TR CS-81-887, 1981.
* Eddins, S. *Binary Image Convex Hull – Algorithm Notes*, MathWorks blog, 2011. ([https://blogs.mathworks.com/steve/2011/10/04/binary-image-convex-hull-algorithm-notes/](https://blogs.mathworks.com/steve/2011/10/04/binary-image-convex-hull-algorithm-notes/))
* *Convex Hull Algorithm – Graham Scan and Jarvis March Tutorial*, Video, 2020. ([https://youtu.be/B2AJoQSZf4M?si=aolY\_c0ByVtfLyXF](https://youtu.be/B2AJoQSZf4M?si=aolY_c0ByVtfLyXF))
* OpenCV Documentation: cv2.findContours, cv2.threshold.
* NumPy Documentation: Array operations, distance calculations.

---

*This README is generated from the CS450 final report. For more detailed information about this project please read the pdf file in this repo*
