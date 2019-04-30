from collections import deque
import numpy as np
import parameters as params

mpx = params.M_TO_PIX_X
mpy = params.M_TO_PIX_Y
y_base = 720*mpy

    
def compute_curvature_and_position(fit):
    A = fit[0]*mpx/(mpy**2)
    B = fit[1]*(mpx/mpy)
    C = fit[2]*mpx
    y_evals = np.linspace(0,y_base,11)
    curvature =  (2*A)/ (1 + (2*A*y_evals + B)**2)**1.5
    radius_of_curvature = len(y_evals)/sum(curvature)
    line_base_pos = A*y_base**2+B*y_base+C-params.CAMERA_X
    return radius_of_curvature, line_base_pos
    
# Define a class to receive the characteristics of each line detection
class lane():
    def __init__(self):
        # was the line detected in the last iteration?
        self.degradation = 0 
        # last n fits of the line
        self.lane_history = deque(maxlen=10)
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #radius of curvature of the line in some units
        self.radius_of_curvature = 0.0
        #distance in meters of vehicle center from the line
        self.line_base_pos = 0.0
        #difference in fit coefficients between last and new fits
        self.error_frames = 0

    def update_fit(self,pts_x,pts_y):
        current_fit = np.polyfit(pts_y,pts_x,2)
        self.update_lane(current_fit)

    def check_plausability(self,radius,position):
        if (np.abs(radius)>params.CURVATURE_MARGIN) and position<3.0:
            return True
        else:
            return False

    def check_relative_plausability(self,radius,position):
        if self.check_plausability(radius,position) and np.abs(self.radius_of_curvature)/(np.abs(1/self.radius_of_curvature-1/radius)<params.CURVATURE_MARGIN_REL) and (np.abs(self.line_base_pos-position)<params.POSITION_MARGIN):
            return True
        else:
            return False

    def set_curvature_and_position(self,radius,position):
        self.radius_of_curvature = radius
        self.line_base_pos = position

    def compute_average_fit(self):
        self.best_fit=sum(self.lane_history)/len(self.lane_history)

    def update_lane(self,fit):
        radius_of_curvature, line_base_pos = compute_curvature_and_position(fit)
        if (self.error_frames==10) or (self.degradation ==1 and self.error_frames>len(self.lane_history)):
            self.degradation = 0
        if (self.degradation == 0):
            self.lane_history.clear()
            if (self.check_plausability(radius_of_curvature, line_base_pos)):
                self.error_frames = 0
                self.set_curvature_and_position(radius_of_curvature, line_base_pos)
                self.best_fit=fit
                self.lane_history.clear()
                self.lane_history.append(fit)
                self.degradation = 1
            else:
                self.error_frames+=1
        elif (self.degradation == 1):
            self.lane_history.append(fit)
            self.compute_average_fit()
            if (self.check_relative_plausability(radius_of_curvature,line_base_pos)):
                radius_of_curvature, line_base_pos = compute_curvature_and_position(self.best_fit)
                self.set_curvature_and_position(radius_of_curvature, line_base_pos)
                self.degradation=2
            else:
                radius_of_curvature, line_base_pos = compute_curvature_and_position(self.best_fit)
                self.set_curvature_and_position(radius_of_curvature, line_base_pos)
                self.error_frames+=1
        elif (self.degradation == 2):
            if (self.check_relative_plausability(radius_of_curvature,line_base_pos)):
                self.error_frames = 0
                self.lane_history.append(fit)
                self.compute_average_fit()
                radius_of_curvature, line_base_pos = compute_curvature_and_position(self.best_fit)
                self.set_curvature_and_position(radius_of_curvature, line_base_pos)
            else:
                self.error_frames+=1