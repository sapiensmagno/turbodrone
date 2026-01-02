import cv2
import numpy as np

def find_board(image):
    """
    Detects the Ludo board in the image.
    Returns the computed width (in pixels) of the board and the contour box.
    """
    # Convert to grayscale and blur to reduce high frequency noise
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edged = cv2.Canny(gray, 35, 125)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (largest first) and keep the top 5
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    for c in contours:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # If the approximated contour has 4 points, we assume it's the board
        # You might want to add aspect ratio checks here for stricter detection
        if len(approx) == 4:
            # Compute the bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(approx)
            
            # Return the width (pixel dimension) and the contour approx
            # We average w and h to handle slight perspective skew
            return (w + h) / 2, approx

    return 0, None

def distance_to_camera(known_width, focal_length, pixel_width):
    """
    Computes distance based on triangle similarity.
    """
    return (known_width * focal_length) / pixel_width

def calibrate_focal_length(known_distance, known_width, pixel_width):
    """
    Calculate focal length from a reference image with known distance.
    """
    return (pixel_width * known_distance) / known_width

# --- CONFIGURATION ---
KNOWN_WIDTH = 30.0  # Real world width of the board in cm
# YOU MUST SET THIS:
# To get this value: take a picture of the board at a known distance (e.g., 50cm).
# Run the calibration step logic below to find F.
FOCAL_LENGTH = 800  # Example value (pixels)

# --- MAIN EXECUTION ---
def main():
    # Load image from experimental/e88_autopilot/landing pad templates/ludo-board.jpg
    image_path = "e88_autopilot/landing pad templates/ludo-board.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Image not found.")
        return

    # 1. Detect the board
    pixel_width, box_contour = find_board(image)
    
    if pixel_width == 0:
        print("Could not detect the Ludo board. Try adjusting lighting or Canny thresholds.")
    else:
        # 2. Calculate Distance (Altitude)
        altitude = distance_to_camera(KNOWN_WIDTH, FOCAL_LENGTH, pixel_width)
        
        # 3. Draw visuals
        cv2.drawContours(image, [box_contour], -1, (0, 255, 0), 2)
        
        # Put text on image
        label = f"Altitude: {altitude:.2f} cm"
        cv2.putText(image, label, (image.shape[1] - 300, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        print(f"Detected Board Width (px): {pixel_width:.2f}")
        print(f"Calculated Altitude: {altitude:.2f} cm")

        # Show result
        cv2.imshow("Ludo Board Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

# import cv2
# import numpy as np

# def find_board_debug(image):
#     # 1. Pre-processing
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Gaussian Blur: Increase the (5,5) to (9,9) if the image is very noisy
#     gray = cv2.GaussianBlur(gray, (5, 5), 0)

#     # 2. Edge Detection (Canny)
#     # Widen these thresholds to detect more lines
#     # If the window is black, lower the 50. If it's too white/noisy, raise it.
#     edged = cv2.Canny(gray, 100, 400) 
    
#     # SHOW THE DEBUG WINDOW
#     cv2.imshow("Debug: Canny Edges", edged)
#     cv2.waitKey(0) # Press any key to continue to contour finding

#     # 3. Find Contours
#     contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Sort by area, keep largest 10
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

#     board_contour = None
    
#     for i, c in enumerate(contours):
#         peri = cv2.arcLength(c, True)
#         # 0.02 is the precision. Try 0.04 if it's not detecting, or 0.01 if it detects circles as squares.
#         approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
#         print(f"Contour #{i} points: {len(approx)} | Area: {cv2.contourArea(c)}")

#         # Check if it has 4 points and is reasonably large (area > 1000)
#         if len(approx) == 4 and cv2.contourArea(c) > 1000:
#             board_contour = approx
#             x, y, w, h = cv2.boundingRect(board_contour)
            
#             # Draw the successful detection in GREEN
#             cv2.drawContours(image, [board_contour], -1, (0, 255, 0), 3)
#             return (w + h) / 2
#         else:
#             # Draw failed candidates in RED to see what is being picked up
#             cv2.drawContours(image, [approx], -1, (0, 0, 255), 1)

#     return 0

# # --- RUN THE DEBUGGER ---
# image = cv2.imread('e88_autopilot/landing pad templates/ludo-board.jpg')
# if image is not None:
#     width = find_board_debug(image)
#     if width == 0:
#         print("Still no board found. Check the 'Debug' window.")
#         print("If the board edges aren't white lines, lower the Canny numbers.")
#         print("If the board is disjointed lines, adjust lighting.")
#     else:
#         print(f"Success! Board width in pixels: {width}")
#         cv2.imshow("Result", image)
#         cv2.waitKey(0)
# cv2.destroyAllWindows()



###################
# ROBUST VERSION that exaggerates the board size
###################

# import cv2
# import numpy as np

# def find_board_robust(image):
#     # 1. Image Pre-processing
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
#     # 2. Canny (Using your tuned parameters)
#     edged = cv2.Canny(gray, 100, 400)

#     # 3. Dilation (Optional but helpful) - thickens lines slightly to help connection
#     kernel = np.ones((5, 5), np.uint8)
#     edged = cv2.dilate(edged, kernel, iterations=1)

#     # 4. Find ALL contours (even the broken bits)
#     contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # 5. Filter and Combine
#     # We want to ignore the noise at the top, so we keep only "significant" lines.
#     significant_contours = []
#     for c in contours:
#         # If the line is long enough (e.g., > 50 pixels), it's part of the board
#         if cv2.arcLength(c, True) > 50:
#             significant_contours.append(c)

#     if not significant_contours:
#         return 0, None

#     # THE TRICK: Stack all points from all significant lines into one big array
#     all_points = np.vstack(significant_contours)

#     # 6. Find the "Minimum Area Rectangle" that encloses this cloud of points
#     # This works even if the corners are completely missing!
#     rect = cv2.minAreaRect(all_points)
    
#     # rect returns ((center_x, center_y), (width, height), angle)
#     (x, y), (w, h), angle = rect

#     # 7. Convert to a box contour for drawing
#     box = cv2.boxPoints(rect)
#     box = np.int32(box)

#     # Return the average width (handling the fact that w/h might be swapped due to rotation)
#     detected_width = min(w, h) if w < h else max(w, h) 
#     # Note: Ludo boards are square, so min or max shouldn't matter much, 
#     # but taking the max is usually safer if there's perspective skew.
#     return max(w, h), box

# # --- Usage ---
# image = cv2.imread('e88_autopilot/landing pad templates/ludo-board.jpg')
# width_px, box = find_board_robust(image)

# if box is not None:
#     cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
#     print(f"Detected Width: {width_px:.2f} px")
#     cv2.imshow("Robust Detection", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# import cv2
# import numpy as np

# def find_board_debug(image):
#     # 1. Pre-processing
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Gaussian Blur: Helps reduce noise before edge detection
#     gray = cv2.GaussianBlur(gray, (5, 5), 0)

#     # 2. Edge Detection (Canny)
#     # Using your parameters 100, 400
#     edged = cv2.Canny(gray, 100, 400) 
    
#     # SHOW RAW CANNY (Before Closing)
#     cv2.imshow("Debug 1: Raw Canny", edged)

#     # --- METHOD 1 INSERTION: Morphological Closing ---
#     # This "smears" white pixels to bridge gaps.
#     # If lines still don't close, increase (5, 5) to (7, 7) or (9, 9).
#     # If it merges too much (blobs), decrease to (3, 3).
#     kernel = np.ones((40, 40), np.uint8)
#     closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    
#     # SHOW CLOSED EDGES (After Closing)
#     cv2.imshow("Debug 2: Closed Edges", closed)
#     print("Press any key to proceed to contour detection...")
#     cv2.waitKey(0) 
#     # -------------------------------------------------

#     # 3. Find Contours
#     # Note: We are now using 'closed' instead of 'edged'
#     contours, _ = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Sort by area, keep largest 10
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

#     board_contour = None
    
#     for i, c in enumerate(contours):
#         peri = cv2.arcLength(c, True)
#         # 0.02 is standard. 
#         # If the shape is slightly distorted, try loosening this to 0.03 or 0.04.
#         approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
#         print(f"Contour #{i} points: {len(approx)} | Area: {cv2.contourArea(c)}")

#         # Check if it has 4 points and is reasonably large (area > 1000)
#         if len(approx) == 4 and cv2.contourArea(c) > 1000:
#             board_contour = approx
#             x, y, w, h = cv2.boundingRect(board_contour)
            
#             # Draw the successful detection in GREEN
#             cv2.drawContours(image, [board_contour], -1, (0, 255, 0), 3)
#             return (w + h) / 2
#         else:
#             # Draw failed candidates in RED to see what is being picked up
#             cv2.drawContours(image, [approx], -1, (0, 0, 255), 1)

#     return 0

# # --- RUN THE DEBUGGER ---
# # I kept your specific path here
# image_path = 'e88_autopilot/landing pad templates/ludo-board.jpg'
# image = cv2.imread(image_path)

# if image is None:
#     print(f"Error: Could not load image from {image_path}")
# else:
#     width = find_board_debug(image)
#     if width == 0:
#         print("Still no board found. Check 'Debug 2' to see if gaps are closed.")
#     else:
#         print(f"Success! Board width in pixels: {width}")
#         cv2.imshow("Result", image)
#         cv2.waitKey(0)

# cv2.destroyAllWindows()

# ###############################
# # FILTER VERSION - kernel extension
# ###############################
# import cv2
# import numpy as np

# def find_board_filtered(image):
#     # 1. Pre-processing
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (5, 5), 0)

#     # 2. Canny Edge Detection (Using your specific settings)
#     # The image you uploaded is very distinct, so 50/150 might work better than 100/400
#     # but let's stick to what gave you the red lines.
#     edged = cv2.Canny(gray, 100, 400) 

#     # 3. Dilate slightly to make the dashed lines "thicker" and more robust
#     kernel = np.ones((3, 3), np.uint8)
#     edged = cv2.dilate(edged, kernel, iterations=1)

#     # 4. Find Contours (The "Red Lines")
#     contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # --- THE MAGIC STEP: FILTERING ---
#     significant_contours = []
    
#     # We create a copy of the image just to draw the "debug" view
#     debug_img = image.copy()

#     for c in contours:
#         # Calculate the length of the contour (red line)
#         length = cv2.arcLength(c, True)
        
#         # FILTER RULE: Only keep lines longer than 100 pixels.
#         # This deletes the QR code, the text "Xalingo", and the tablecloth scribbles.
#         # Keep only the long board edges and the big circles.
#         if length > 100:
#             significant_contours.append(c)
#             # Draw what we are keeping in BLUE
#             cv2.drawContours(debug_img, [c], -1, (255, 0, 0), 2)
#         else:
#             # Draw what we are ignoring in RED (Noise/Flowers)
#             cv2.drawContours(debug_img, [c], -1, (0, 0, 255), 1)

#     # Show the user what we kept vs what we threw away
#     cv2.imshow("Debug: Blue=Kept, Red=Ignored", debug_img)
#     cv2.waitKey(0)

#     if not significant_contours:
#         print("Error: No long lines found. Try lowering the '100' threshold.")
#         return 0, None

#     # 5. Cloud of Points Logic
#     # Combine all the "Blue" lines into one big pile of points
#     all_points = np.vstack(significant_contours)

#     # 6. Find the bounding box of that pile
#     rect = cv2.minAreaRect(all_points)
#     (x, y), (w, h), angle = rect
    
#     # Convert to box for drawing
#     box = cv2.boxPoints(rect)
#     box = np.int32(box) # Fix for the numpy error you saw earlier

#     return max(w, h), box

# # --- MAIN ---
# # Load your color image
# image = cv2.imread('e88_autopilot/landing pad templates/ludo-board.jpg')


# if image is None:
#     print("Image not found")
# else:
#     width_px, box_contour = find_board_filtered(image)
    
#     if box_contour is not None:
#         # Draw the final result in GREEN
#         cv2.drawContours(image, [box_contour], 0, (0, 255, 0), 3)
        
#         print(f"Final Detected Width: {width_px:.2f} pixels")
#         cv2.imshow("Final Result", image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()


import cv2
import numpy as np

def find_board_square_forced(image):
    # 1. Image Pre-processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Canny (Using your tuned parameters)
    edged = cv2.Canny(gray, 100, 400)

    # 3. Dilation (Optional but helpful) - thickens lines slightly to help connection
    kernel = np.ones((5, 5), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)

    # 4. Find ALL contours (even the broken bits)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. Filter and Combine
    # We want to ignore the noise at the top, so we keep only "significant" lines.
    significant_contours = []
    for c in contours:
        # If the line is long enough (e.g., > 50 pixels), it's part of the board
        if cv2.arcLength(c, True) > 50:
            significant_contours.append(c)

    if not significant_contours:
        return 0, None

    # THE TRICK: Stack all points from all significant lines into one big array
    all_points = np.vstack(significant_contours)

    # 6. Find the "Minimum Area Rectangle" that encloses this cloud of points
    rect = cv2.minAreaRect(all_points)
    
    # rect returns ((center_x, center_y), (width, height), angle)
    (x, y), (w, h), angle = rect

    # --- FORCING SQUARE GEOMETRY ---
    # Since the tablecloth is "extending" the box, the LONGER side is the wrong one.
    # The SHORTER side is likely the true width of the board.
    side_length = min(w, h)
    
    # Create a new rect tuple with the same center and angle, but square dimensions
    square_rect = ((x, y), (side_length, side_length), angle)
    # -------------------------------

    # 7. Convert to a box contour for drawing
    box = cv2.boxPoints(square_rect)
    box = np.int32(box)

    return side_length, box

# --- Usage ---
# Replace with your actual image path
image = cv2.imread('e88_autopilot/landing pad templates/ludo-board.jpg')

if image is None:
    print("Error: Image not found.")
else:
    width_px, box = find_board_square_forced(image)

    if box is not None:
        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
        print(f"Detected Square Side: {width_px:.2f} px")
        cv2.imshow("Forced Square Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()