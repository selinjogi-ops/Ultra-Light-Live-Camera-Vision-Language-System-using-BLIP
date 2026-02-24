"""Ultra-Light Live Camera VLM - Compact Version"""
# Brief description of the program

from transformers import BlipProcessor, BlipForConditionalGeneration
# Import BLIP model components from Hugging Face transformers library
# BlipProcessor: Handles image preprocessing (resize, normalize, convert to tensors)
# BlipForConditionalGeneration: The actual AI model that generates text from images

from PIL import Image
# Import Python Imaging Library for image manipulation

import torch, cv2, time, threading, warnings
# torch: PyTorch deep learning framework
# cv2: OpenCV library for camera capture and image display
# time: For timing operations and delays
# threading: For running AI analysis in background without freezing UI
# warnings: To suppress non-critical warning messages

warnings.filterwarnings('ignore')
# Suppress all warning messages to keep console output clean

print("="*60, "\nLIVE CAMERA VLM - Ultra-Light Version\n", "="*60)
# Print header banner with 60 equal signs for visual separation

print("\nLoading BLIP-base model...\n")
# Inform user that model loading has started

# Load model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# Download and load the BLIP preprocessor from Hugging Face hub
# This handles converting images into the format the model expects
# First run: downloads ~500MB, subsequent runs: loads from cache

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    # Download and load the BLIP model weights from Hugging Face
    torch_dtype=torch.float32, 
    # Use 32-bit floating point precision (higher accuracy, more memory)
    use_safetensors=True
    # Use safer tensor format for loading weights (prevents code injection)
).to("cpu")
# Move model to CPU memory (use .to("cuda") for GPU acceleration)

model.eval()
# Set model to evaluation mode (disables dropout, batch normalization training behavior)
# Required for inference, improves performance and consistency

torch.set_num_threads(4)
# Limit PyTorch to use 4 CPU threads for inference
# Prevents overloading the system, tune based on your CPU cores

print("✓ Model loaded!\n")
# Confirm successful model loading

# Globals
current_description, is_analyzing, last_analysis_time, analysis_interval = "Starting camera...", False, 0, 4
# Initialize global variables in one line:
# current_description: Stores the latest AI-generated description of the scene
# is_analyzing: Boolean flag to prevent multiple simultaneous analyses
# last_analysis_time: Timestamp of last analysis (in seconds since epoch)
# analysis_interval: Wait 4 seconds between automatic analyses

def analyze_frame(frame):
    # Function to analyze a single camera frame and generate description
    # Runs in a separate thread to avoid blocking the main camera loop
    
    global current_description, is_analyzing
    # Declare that we'll modify these global variables (not create local copies)
    
    is_analyzing = True
    # Set flag to indicate analysis is in progress
    # Prevents starting another analysis while this one runs
    
    start = time.time()
    # Record start time to measure how long analysis takes
    
    try:
        # Try-except block to handle any errors gracefully
        
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Convert frame from OpenCV format to PIL Image:
        # cv2.cvtColor: Convert color from BGR (OpenCV default) to RGB (PIL/model expects)
        # Image.fromarray: Create PIL Image object from numpy array
        
        img.thumbnail((384, 384), Image.Resampling.LANCZOS)
        # Resize image to maximum 384×384 pixels while maintaining aspect ratio
        # LANCZOS: High-quality resampling algorithm (slower but better quality)
        # Smaller size = faster inference, less memory
        
        inputs = processor(img, return_tensors="pt").to("cpu")
        # Preprocess the image for the model:
        # processor(): Normalizes pixel values, adds batch dimension
        # return_tensors="pt": Return PyTorch tensors (not numpy arrays)
        # .to("cpu"): Ensure tensors are on CPU (match model location)
        # Result: {'pixel_values': tensor of shape [1, 3, 384, 384]}
        
        with torch.no_grad():
            # Context manager that disables gradient calculation
            # Saves memory and speeds up inference (gradients only needed for training)
            
            out = model.generate(**inputs, max_length=50, num_beams=3)
            # Generate text caption from image:
            # **inputs: Unpacks dictionary into keyword arguments
            # max_length=50: Generate up to 50 tokens (words/subwords)
            # num_beams=3: Use beam search with 3 beams (balances quality vs speed)
            # Returns: Tensor of token IDs representing the generated text
        
        result = processor.decode(out[0], skip_special_tokens=True)
        # Convert token IDs back to readable text:
        # out[0]: Get first (and only) sequence from batch
        # skip_special_tokens=True: Remove [BOS], [EOS], [PAD] tokens
        # Result: Plain English description like "a woman sitting at a desk"
        
        current_description = f"{result} (took {time.time()-start:.1f}s)"
        # Update global description with result and timing info
        # .1f formats time to 1 decimal place
        
    except Exception as e:
        # Catch any error that occurs during analysis
        
        current_description = f"Error: {e}"
        # Store error message instead of crashing the program
        
    is_analyzing = False
    # Clear the analyzing flag so another analysis can start

def draw_text(img, text, pos, scale=0.7):
    # Function to draw multi-line text with background on an image
    # Handles word wrapping automatically based on image width
    
    font, thick, pad, lh = cv2.FONT_HERSHEY_SIMPLEX, 2, 10, 30
    # Set text rendering parameters:
    # font: OpenCV font type (simple, readable)
    # thick: Thickness of 2 pixels for bold text
    # pad: 10 pixel padding around text background
    # lh: Line height of 30 pixels between lines
    
    max_w = img.shape[1] - 40
    # Calculate maximum text width (image width minus 40 pixel margins)
    
    lines, line = [], []
    # Initialize empty lists:
    # lines: Will hold complete wrapped lines
    # line: Temporary storage for current line being built
    
    for word in text.split():
        # Iterate through each word in the text
        
        test = ' '.join(line + [word])
        # Create test string with current line plus new word
        
        if cv2.getTextSize(test, font, scale, thick)[0][0] < max_w:
            # Check if adding this word keeps us within max width:
            # getTextSize returns ((width, height), baseline)
            # [0][0] extracts just the width
            
            line.append(word)
            # Word fits, add it to current line
        else:
            # Word doesn't fit, need to wrap
            
            if line: lines.append(' '.join(line))
            # Save current line if it has content
            
            line = [word]
            # Start new line with the word that didn't fit
            
    if line: lines.append(' '.join(line))
    # Add the last line (won't be added in loop)
    
    x, y = pos
    # Unpack position tuple into x and y coordinates
    
    cv2.rectangle(img, (x-pad, y-pad), (img.shape[1]-x+pad, y+len(lines)*lh+pad), (0,0,0), -1)
    # Draw black background rectangle for text:
    # (x-pad, y-pad): Top-left corner with padding
    # (img.shape[1]-x+pad, y+len(lines)*lh+pad): Bottom-right corner
    # (0,0,0): Black color in BGR
    # -1: Fill the rectangle (not just outline)
    
    for i, l in enumerate(lines):
        # Iterate through lines with index
        
        cv2.putText(img, l, (x, y+(i*lh)+20), font, scale, (0,255,0), thick)
        # Draw each line of text:
        # l: The text to draw
        # (x, y+(i*lh)+20): Position (offset each line by line height)
        # font: Font type
        # scale: Font size multiplier (0.7)
        # (0,255,0): Green color in BGR
        # thick: Thickness of 2 pixels

# Camera setup
print("Opening camera...")
# Inform user that camera initialization is starting

cap = cv2.VideoCapture(0)
# Create video capture object connected to camera device 0
# 0 is usually the default/built-in webcam
# Change to 1, 2, etc. for external cameras

if not cap.isOpened():
    # Check if camera was successfully opened
    
    print("❌ Cannot open camera!")
    # Print error message
    
    exit()
    # Terminate program if camera fails

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# Set camera capture width to 640 pixels

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# Set camera capture height to 480 pixels
# 640×480 is VGA resolution - good balance of quality and performance

print("✓ Camera opened!\n", "="*60)
# Confirm camera is ready

print("CONTROLS:\n- Press 'q' to quit\n- Press 's' to save\n- Press SPACE for analysis")
# Display keyboard controls to user

print("="*60, "\n")
# Print separator line

try:
    # Try block for main program loop, allows clean exit on errors
    
    while True:
        # Infinite loop - runs until user quits
        
        ret, frame = cap.read()
        # Capture one frame from camera:
        # ret: Boolean indicating if frame was successfully captured
        # frame: The actual image data as numpy array (height, width, 3)
        
        if not ret: break
        # If frame capture failed, exit the loop
        
        t = time.time()
        # Get current time in seconds since epoch (for timing)
        
        if (t - last_analysis_time >= analysis_interval) and not is_analyzing:
            # Check if it's time for automatic analysis:
            # (t - last_analysis_time >= analysis_interval): 4 seconds have passed
            # and not is_analyzing: No analysis currently running
            
            last_analysis_time = t
            # Update last analysis time to now
            
            threading.Thread(target=analyze_frame, args=(frame.copy(),), daemon=True).start()
            # Start analysis in a separate thread:
            # target=analyze_frame: Function to run in thread
            # args=(frame.copy(),): Pass a copy of frame (prevents race conditions)
            # daemon=True: Thread dies when main program exits
            # .start(): Actually start the thread
        
        disp = frame.copy()
        # Create a copy of frame for display (don't modify original)
        
        cv2.rectangle(disp, (0,0), (disp.shape[1],50), (40,40,40), -1)
        # Draw dark gray header bar at top:
        # (0,0): Top-left corner
        # (disp.shape[1],50): Bottom-right (full width, 50 pixels tall)
        # (40,40,40): Dark gray color
        # -1: Filled rectangle
        
        cv2.putText(disp, "LIVE AI VISION", (20,35), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,255,0), 2)
        # Draw title text in header:
        # "LIVE AI VISION": Text to display
        # (20,35): Position
        # cv2.FONT_HERSHEY_DUPLEX: More stylized font
        # 0.8: Font scale
        # (0,255,0): Green color
        # 2: Thickness
        
        col, stat = ((0,165,255), "ANALYZING...") if is_analyzing else ((0,255,0), "READY")
        # Set status indicator based on analyzing flag:
        # If analyzing: Orange color (0,165,255) and "ANALYZING..." text
        # If ready: Green color (0,255,0) and "READY" text
        
        cv2.circle(disp, (disp.shape[1]-120, 25), 8, col, -1)
        # Draw status indicator circle in header:
        # (disp.shape[1]-120, 25): Position near top-right
        # 8: Radius in pixels
        # col: Color determined above
        # -1: Filled circle
        
        cv2.putText(disp, stat, (disp.shape[1]-100, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        # Draw status text next to indicator circle:
        # stat: Text determined above ("ANALYZING..." or "READY")
        # Position near top-right
        # (255,255,255): White color
        # 1: Thin thickness
        
        draw_text(disp, current_description, (20, disp.shape[0]-80))
        # Draw AI description at bottom of frame:
        # Uses custom draw_text function for multi-line with background
        # Position: 20 pixels from left, 80 pixels from bottom
        
        cv2.imshow('Live AI Vision - BLIP', disp)
        # Display the frame in a window:
        # 'Live AI Vision - BLIP': Window title
        # disp: The image to display
        
        key = cv2.waitKey(1) & 0xFF
        # Wait 1 millisecond for keyboard input:
        # Returns -1 if no key pressed, or ASCII code of key
        # & 0xFF: Mask to get only last 8 bits (handles platform differences)
        
        if key == ord('q'):
            # Check if 'q' key was pressed
            
            print("\nQuitting...")
            # Print exit message
            
            break
            # Exit the main loop
            
        elif key == ord('s'):
            # Check if 's' key was pressed (save screenshot)
            
            fn = f"capture_{int(t)}.jpg"
            # Create filename with timestamp: capture_1234567890.jpg
            
            cv2.imwrite(fn, frame)
            # Save the original frame (without overlays) to file
            
            print(f"✓ Saved: {fn}")
            # Confirm save with filename
            
        elif key == ord(' ') and not is_analyzing:
            # Check if SPACE key was pressed and not currently analyzing
            
            print("Forcing analysis...")
            # Inform user that manual analysis started
            
            threading.Thread(target=analyze_frame, args=(frame.copy(),), daemon=True).start()
            # Start immediate analysis in background thread

except KeyboardInterrupt:
    # Catch Ctrl+C interrupt from user
    
    print("\nStopped by user")
    # Print clean exit message
    
finally:
    # This block always runs, even after break or exception
    
    cap.release()
    # Release camera resources (turn off camera, free memory)
    
    cv2.destroyAllWindows()
    # Close all OpenCV windows
    
    print("\n✓ Done!\n", "="*60)
    # Print completion message with separator