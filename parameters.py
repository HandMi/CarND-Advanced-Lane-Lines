SOBEL_KERNEL_SIZE=5
SOBEL_THRESHOLD=(30, 255)
MAGNITUDE_THRESHOLD=(25, 360)
DIRECTIONAL_THRESHOLD=(0.3, 1.2)


dark_yellow = (18, 60, 90)
light_yellow = (35,255,255)
dark_white = (0, 0, 200)
light_white = (255,25,255)

# Choose the number of sliding windows
nwindows = 10
# Set the width of the windows +/- margin
window_margin = 50
# Set minimum number of pixels found to recenter window
window_minpix = 50

# Pixel to meter in y direction
M_TO_PIX_Y=3.05/58.0

# Pixel to meter in x direction
M_TO_PIX_X=3.7/658.0

# Camera Position
CAMERA_X=640*M_TO_PIX_X

# Minimum Curvature
CURVATURE_MARGIN=50.0

# Maximum relative deviation of new lane radius of curvature
CURVATURE_MARGIN_REL=0.5

# Maximum deviation of new lane base position in ms
POSITION_MARGIN=1.0

# Max number of frames without lane detection
ERROR_FRAMES=20