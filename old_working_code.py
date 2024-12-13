# imports
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import copy
import glob
import pickle
import cv2
import re
import numpy as np
import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import yolov5
from scipy.stats import multivariate_normal # while cropping the obstacles
from math import sqrt, exp
from scipy.optimize import linear_sum_assignment # required in associate function. 
from tqdm import tqdm
# load & visualize image. images are not shown. because we convert images from RGB to BGR.
# and matplotlib has problem in displaying images in BGR.
# def visualize_images_x(img):
#     cv2.imshow("image", img)
#     cv2.waitKey(0)

# def visualize_images(input_images):
#     fig=plt.figure(figsize=(150,75))
#     for i in range(len(input_images)):
#         fig.add_subplot(1, len(input_images), i+1)
#         plt.imshow(input_images[i])
#     plt.show()

# # Get sorted list of image files
# images_files = sorted(glob.glob("dataset/nvidia_ai_challenge_images/*.jpg"), key=lambda x: float(re.findall(r"(\d+)", x)[0]))

# images_rgb = []  # List for RGB images
# images_bgr = []  # List for BGR images (if needed)

# # Load images 57 and 58
# index = 57
# for img_path in images_files[index:index+2]:
#     # Read image as BGR
#     img_rgb = cv2.imread(img_path)
#     visualize_images_x(img_rgb)
#     images_rgb.append(img_rgb)  # Add BGR image to separate list
#     # Convert to RGB for display
#     img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
#     images_bgr.append(img_bgr)  # Add RGB image for display
#     # visualize_images(img_bgr)




#######################################################################
# yolo

global stored_obstacles
global idx

class Obstacle():
    def __init__(self, idx, box, features=None,  age=1, unmatched_age=0):
        """
        Init function. The obstacle must have an id and a box.
        """
        self.idx = idx
        self.box = box
        self.features = features
        self.age = age
        self.unmatched_age = unmatched_age

class Yolo_implmentation:
    def __init__(self):


        self.model = yolov5.load('models/yolov5s.pt')

        self.model.conf = 0.5
        self.model.iou = 0.4

        self.classesFile = "models/coco.names"
        with open(self.classesFile,'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        # hungarian
        self.encoder = torch.load("models/model640.pt", map_location=torch.device('cpu'))
        self.encoder = self.encoder.eval()

        self.MIN_HIT_STREAK = 1
        self.MAX_UNMATCHED_AGE = 1

        # for testing_main_function
        # self.stored_obstacles=[]
        # self.idx=0


    def generate_random_color(self, idxx):
        """
        Random function to convert an id to a color
        Do what you want here but keep numbers below 255
        """
        blue = idxx*5 % 256
        green = idxx*12 %256
        red = idxx*23 %256
        return (red, green, blue)

    def draw_boxes(self, image, boxes, categories, mot_mode=False):
        
        # iterates through all bounding boxes
        for i, box in enumerate(boxes):

            # get class name from category ID using COCO names
            label = self.classes[int(categories[i])]
            
            # if mot_mode == True generates unique color for each object
            # else uses red color
            color = self.generate_random_color(i*10) if mot_mode==True else (255,0,0)

            # draws bounding box on rectangle
            # rectangle ( image, top-left corner, bottom_right, box color, line thickness )
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness=7)

            # add class name above each box
            cv2.putText(image, str(label), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness=3)
        return image

    def get_yolo_model_results(self, img):

        # pass input image to YOLO model. gets raw detection results. results is of length 1. 
        results = self.model(img)
        
        # get all predicted objects in an image.
        # each predicted object contains 6 values. 
        # First 4 are bounding box coordinates. [x1, y1, x2, y2] top-left corner, bottom-right corner
        # 5th value is confidence score.
        # 6th value is label. this label is from coco.names
        predictions = results.pred[0]

        # get bounding box coordinates for all detections. 
        boxes = predictions[:, :4].tolist()
        
        # convert floating point coordinates to integers.
        boxes_int = [[int(v) for v in box] for box in boxes]

        # get all confidence scores. 
        scores = predictions[:,4].tolist()

        # get all class IDs
        categories = predictions[:,5].tolist()

        # convert class IDs to integers.
        categories_int = [int(c) for c in categories]

        # draws detection boxes and labels on image 
        img_out = self.draw_boxes(img, boxes_int, categories_int, mot_mode=True)

        # img_out: image with drawn boxes and labels.
        # boxes_int: list of bounding box coordinates
        # categories_int: list of object categories
        # list of confidence scores. 
        return img_out, boxes_int, categories_int, scores

    def visualize_images(self, input_images):
        for i, img in enumerate(input_images):
            # Create named window for each image
            cv2.namedWindow(f'Image {i}', cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f'Image {i}', 800, 600)  # Set window size
            cv2.imshow(f'Image {i}', img)
        
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # cropping the obstacles
    # takes a image or frame of a video, and coordinates for cropping of objects. 
    def crop_frames(self, frame, boxes):
        try:
            # converts image to PIL
            # resizes it to 128 x 128
            # converts it back to PyTorch tensor
            transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(), torchvision.transforms.Resize((128,128)), torchvision.transforms.ToTensor()])
            
            # stores original cropped images
            crops = []

            # stores transformed pytorch tensor versions
            crops_pytorch = []

            # for each box
            for box in boxes:

                # crops the frame using coordinates (height, width)
                crop = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                crops.append(crop)

                # transform the crop and adds it to crops_torch. 
                crops_pytorch.append(transforms(crop))

            return crops, torch.stack(crops_pytorch)
        except:
            return [],[]

    # purpose of gaussian mask is to create a weight distribution that follows a bell curve shape.
    # highest in center and gradually decreasing towards edges. 
    # this is used to: give more importance to features in center of image. 
    # reduce influence of edge
    # create smooth attention or focus mechanisms
    def get_gaussian_mask(self):
        # creates two coordinate grids ranging from 0 to 1 with 128 points each. 
        # this a coordinate system for 128 x 128 image.
        # each point (x,y) represents a pixel position. 
        # values are normalized between 0 and 1. (center is 0.5, 0.5)
        x, y = np.mgrid[0:1.0:128j, 0:1.0:128j]

        # transforms coordinates into pairs. flatens the grid and then combines them. 
        # each row becomes (x,y) coordinate pair. 
        # result is (128*128, 2) = 16384 coordinate pairs. 
        xy = np.column_stack([x.flat, y.flat])

        # defines gaussian distribution params
        # centers the peak of bell curve in middle of image
        mu = np.array([0.5,0.5])
        # controls how quickly values fall off from center.
        # smaller sigma = steeper fall = more focused center
        # larger sigma = gentle fall = more spread out weights
        sigma = np.array([0.22,0.22])
        # creates a diagonal matrix of variances (sigma 2) 
        # using diagonal matrix means x and y directions are independent. 
        # vale of sigma**2 = 0.0484 means ~95% of weights are within 2*0.022 = 0.44 units from center. 
        covariance = np.diag(sigma**2)

        # f(x,y) = (1/2πσ²) * exp(-((x-μx)² + (y-μy)²)/(2σ²))
        # (y-μy)²)/(2σ²)): measures squared distance from center.
        # exp(-((x-μx)²: creates characteristic bell shape curve. 
        # result is highest (1.0) at center and decreases exponentially with distance.
        z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
        z = z.reshape(x.shape)

        z = z / z.max()
        z  = z.astype(np.float32)

        mask = torch.from_numpy(z)

        # resulting mask will look like. 
        # 1. center pixel = 1.0 (white)
        # 2. smoothly decreasing valyes towards edges ( getting darker)
        # 3. corner pixels = 0.0 ( black)

        # when this mask is applied to image using multiplication
        # center pixels retain most of original values. 
        # edge pixels are dampened. 
        # creates circular effect. 
        return mask
    
    # called 4th in total_cost
    # calculates cosine similarity between two sets of vectors. 
    # purpose of this function is to measure how similar vectors are by their orientation, regardless of magnitude.
    # commonly used in image similarity. 
    def cosine_similarity(self, a, b, data_is_normalized=False):
        
        # checks if data is normalized.
        if not data_is_normalized:

            # converts a to numpy array. 
            # calculates magnitude (length) of each vector in a using np.linalg.norm
            # divides each vector by its magnitude to normalize it. 
            # axis = 1 means apply operation row by row.

            # a = [0, 0, 1]
            # calculate magnitude = sqrt (0**2 + 0**2 + 1**2) = sqrt(1) = 1
            # normalized a = [0/1, 0/1, 1/1] = [0, 0, 1]
            a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)

            # b = sqrt( 0**2 + 1**2 + 1**2) = sqrt(2)= square root 2
            # normalized b = [0/sqrt(2), 1/sqrt(2), 1/sqrt(2)] = [0, 1/sqrt(2), 1/sqrt(2)]
            b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
        
        # calculates dot product between normalized vectors in a and transpose of b
        # result is cosine similarity matrix between all pairs of vectors from a & b. 
        # values range from -1 to 1. 1 means vectors point in same direction. 
        # 0 means vectors are perpendicular. -1 means vectors point in opposite directions. 

        # dot_product = (0x0) + (0x1/sqrt(2)) + (1x1/sqrt(2))
        # = 0 + 0 + 1/sqrt(2) = 1/sqrt(2) = 0.707.
        # this results means vectors have similarity of about 0.707 which indicates they are similar but not perfectly aligned. 
        return np.dot(a, b.T)

    
    def get_features(self, processed_crops):
        features = []
        if len(processed_crops)>0:
            features = self.encoder.forward_once(processed_crops)
            features = features.detach().cpu().numpy()
            if len(features.shape)==1:
                features = np.expand_dims(features,0)
        return features  

    # called 1st in total_cost
    def box_iou(self, box1, box2, w = 1280, h=360):
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1) #abs((xi2 - xi1)*(yi2 - yi1))
        # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1) #abs((box1[3] - box1[1])*(box1[2]- box1[0]))
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1) #abs((box2[3] - box2[1])*(box2[2]- box2[0]))
        union_area = (box1_area + box2_area) - inter_area
        # compute the IoU
        iou = inter_area/float(union_area)
        return iou

    def check_division_by_0(self, value, epsilon=0.01):
        if value < epsilon:
            value = epsilon
        return value

    # called 2nd in total_cost
    def sanchez_matilla(self, box1, box2, w = 1280, h=360):
        Q_dist = sqrt(pow(w,2)+pow(h,2))
        Q_shape = w*h
        distance_term = Q_dist/self.check_division_by_0(sqrt(pow(box1[0] - box2[0], 2)+pow(box1[1] -box2[1],2)))
        shape_term = Q_shape/self.check_division_by_0(sqrt(pow(box1[2] - box2[2], 2)+pow(box1[3] - box2[3],2)))
        linear_cost = distance_term*shape_term
        return linear_cost
    
    # called 3rd in total_cost
    def yu(self, box1, box2):
        w1 = 0.5
        w2 = 1.5
        a= (box1[0] - box2[0])/self.check_division_by_0(box1[2])
        a_2 = pow(a,2)
        b = (box1[1] - box2[1])/self.check_division_by_0(box1[3])
        b_2 = pow(b,2)
        ab = (a_2+b_2)*w1*(-1)
        c = abs(box1[3] - box2[3])/(box1[3]+box2[3])
        d = abs(box1[2]-box2[2])/(box1[2]+box2[2])
        cd = (c+d)*w2*(-1)
        exponential_cost = exp(ab)*exp(cd)
        return exponential_cost

    # called in associate function
    def total_cost(self, old_box, new_box, old_features, new_features, iou_thresh = 0.3, linear_thresh = 10000, exp_thresh = 0.5, feat_thresh = 0.2):
        iou_cost = self.box_iou(old_box, new_box)
        linear_cost = self.sanchez_matilla(old_box, new_box, w= 1920, h=1080)
        exponential_cost = self.yu(old_box, new_box)
        feature_cost = self.cosine_similarity(old_features, new_features)[0][0]

        if (iou_cost >= iou_thresh and linear_cost >= linear_thresh and exponential_cost>=exp_thresh and feature_cost >= feat_thresh):
            return iou_cost

        else:
            return 0

    def associate(self, old_boxes, new_boxes, old_features, new_features):
        """
        old_boxes will represent the former bounding boxes (at time 0)
        new_boxes will represent the new bounding boxes (at time 1)
        Function goal: Define a Hungarian Matrix with IOU as a metric and return, for each box, an id
        """
        if len(old_boxes)==0 and len(new_boxes)==0:
            return [], [], []
        elif(len(old_boxes)==0):
            return [], [i for i in range(len(new_boxes))], [] #Weird trick
        elif(len(new_boxes)==0):
            return [], [], [i for i in range(len(old_boxes))]# Weird trick 
            
        # Define a new IOU Matrix nxm with old and new boxes
        iou_matrix = np.zeros((len(old_boxes),len(new_boxes)),dtype=np.float32)
        
        # Go through boxes and store the IOU value for each box 
        # You can also use the more challenging cost but still use IOU as a reference for convenience (use as a filter only)
        for i,old_box in enumerate(old_boxes):
            for j,new_box in enumerate(new_boxes):
                iou_matrix[i][j] = self.total_cost(old_box, new_box, old_features[i].reshape(1,1024), new_features[j].reshape(1,1024))

        #print(iou_matrix)
        # Call for the Hungarian Algorithm
        hungarian_row, hungarian_col = linear_sum_assignment(-iou_matrix)
        hungarian_matrix = np.array(list(zip(hungarian_row, hungarian_col)))

        # Create new unmatched lists for old and new boxes
        matches, unmatched_detections, unmatched_trackers = [], [], []

        #print(hungarian_matrix)
        
        # Go through old boxes, if no matched detection, add it to the unmatched_old_boxes
        for t,trk in enumerate(old_boxes):
            if(t not in hungarian_matrix[:,0]):
                unmatched_trackers.append(t)
        
        # Go through new boxes, if no matched tracking, add it to the unmatched_new_boxes
        for d, det in enumerate(new_boxes):
            if(d not in hungarian_matrix[:,1]):
                    unmatched_detections.append(d)
        
        # Go through the Hungarian Matrix, if matched element has IOU < threshold (0.3), add it to the unmatched 
        for h in hungarian_matrix:
            if(iou_matrix[h[0],h[1]]<0.3):
                unmatched_trackers.append(h[0]) # Return INDICES directly
                unmatched_detections.append(h[1]) # Return INDICES directly
            else:
                matches.append(h.reshape(1,2))
        
        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)
        

        return matches, unmatched_detections,unmatched_trackers

    # imitates main function running for single image at a time. 
    def process_single_image(self, input_image):
        global stored_obstacles
        global idx
        # 1 — Run Obstacle Detection & Convert the Boxes
        final_image = copy.deepcopy(input_image)
        h, w, _ = final_image.shape

        _, out_boxes, _, _ = self.get_yolo_model_results(input_image)
        crops, crops_pytorch = self.crop_frames(final_image, out_boxes)
        features = self.get_features(crops_pytorch)
        
        # print("----> New Detections: ", out_boxes)
        # Define the list we'll return:
        new_obstacles = []

        old_obstacles = [obs.box for obs in stored_obstacles] # Simply get the boxes
        old_features = [obs.features for obs in stored_obstacles]
        
        matches, unmatched_detections, unmatched_tracks = self.associate(old_obstacles, out_boxes, old_features, features)

        # Matching
        for match in matches:
            obs = Obstacle(stored_obstacles[match[0]].idx, out_boxes[match[1]], features[match[1]], stored_obstacles[match[0]].age +1)
            new_obstacles.append(obs)
            # print("Obstacle ", obs.idx, " with box: ", obs.box, "has been matched with obstacle ", stored_obstacles[match[0]].box, "and now has age: ", obs.age)
        
        # New (Unmatched) Detections
        for d in unmatched_detections:
            obs = Obstacle(idx, out_boxes[d], features[d])
            new_obstacles.append(obs)
            idx+=1
            # print("Obstacle ", obs.idx, " has been detected for the first time: ", obs.box)

        # Unmatched Tracks
        for t in unmatched_tracks:
            i = old_obstacles.index(stored_obstacles[t].box)
            # print("Old Obstacles tracked: ", stored_obstacles[i].box)
            if i is not None:
                obs = stored_obstacles[i]
                obs.unmatched_age +=1
                new_obstacles.append(obs)
                # print("Obstacle ", obs.idx, "is a long term obstacle unmatched ", obs.unmatched_age, "times.")

        # Draw the Boxes
        for i, obs in enumerate(new_obstacles):
            if obs.unmatched_age > self.MAX_UNMATCHED_AGE:
                new_obstacles.remove(obs)

            if obs.age >= self.MIN_HIT_STREAK:
                left, top, right, bottom = obs.box
                cv2.rectangle(final_image, (left, top), (right, bottom), self.generate_random_color(obs.idx*10), thickness=7)
                final_image = cv2.putText(final_image, str(obs.idx),(left - 10,top - 10),cv2.FONT_HERSHEY_SIMPLEX, 1, self.generate_random_color(obs.idx*10),thickness=4)

        stored_obstacles = new_obstacles

        return final_image, stored_obstacles

def main():
    # create instance of yolo implementation class.
    global stored_obstacles
    global idx
    yolo_obj = Yolo_implmentation()

    # get all .jpg files. sort them based on numeric part in theri filenames. regex part extracts numeric part from each filename
    # float() converts extracted number to allow proper numeric sorting. 
    images_files = sorted(glob.glob("dataset/nvidia_ai_challenge_images/*.jpg"), key=lambda x:float(re.findall("(\d+)",x)[0]))
    
    # creates empty list to store loaded images.
    images = []

    # start loading images from index 57.
    index = 57
    print(len(images_files))
    # for each image. 
    # read image in BGR format 
    # convert image from BGR to RGB format. 
    # add RGB image to images list. 
    # for img in images_files[index:index+2]: used for testing code. 
    for img in images_files:
        images.append(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB))

    print(len(images))
    # visualization method to display loaded images. 
    # yolo_obj.visualize_images(images)
    
    # yolo_images = []
    # yolo_boxes = []
    # yolo_categories = []
    # yolo_scores = []
    # pics = copy.deepcopy(images)

    # for img in pics:
    #     result, pred_bboxes, pred_categories, pred_scores = yolo_obj.get_yolo_model_results(img)
    #     yolo_boxes.append(pred_bboxes)
    #     yolo_images.append(result)
    #     yolo_categories.append(pred_categories)
    #     yolo_scores.append(pred_scores)

    # # yolo_obj.visualize_images(yolo_images)
    # # print("called all functions")

    # # hungarian process
    # # process
    # # 1. gets gaussian mask. 
    # # 2. crops detected objects from image
    # # 3. applies gaussian mask to transformed crops. then visualize normal cropped features and not masked pytorch features 
    # # 4. 
    # # purpose
    # # 1. extracts individual objects from detection boxes. 
    # # 2. standardizes their sizes.
    # # 3. applies attention mechanism (Gaussian mask) to focus on central parts. 
    
    # gaussian_mask = yolo_obj.get_gaussian_mask()
    
    # crops_1, crops_pytorch_1 = yolo_obj.crop_frames(pics[0], yolo_boxes[0])
    # crops_pytorch_1 = gaussian_mask * crops_pytorch_1
    # # yolo_obj.visualize_images(crops_1)
    
    # crops_2, crops_pytorch_2 = yolo_obj.crop_frames(pics[1], yolo_boxes[1])
    # crops_pytorch_2 = gaussian_mask * crops_pytorch_2
    # # yolo_obj.visualize_images(crops_2)

    # features_1 = yolo_obj.get_features(crops_pytorch_1)
    # features_2 = yolo_obj.get_features(crops_pytorch_2) 

    # matches, unmatched_detections, unmatched_trackers = yolo_obj.associate(yolo_boxes[0], yolo_boxes[1], features_1, features_2)
    # print("executed")
    
    # testing code
    # out_imgs = []
    idx = 0
    stored_obstacles = []

    # for i in range(len(images)):
    #     out_img, stored_obstacles = yolo_obj.process_single_image(images[i])
    #     out_imgs.append(out_img)
    # yolo_obj.visualize_images(out_imgs)

    result_video = []
    out_img=[]
    for img in tqdm(images):
        out_img, stored_obstacles = yolo_obj.process_single_image(img)
        if out_img is not None:
            result_video.append(out_img)

    out = cv2.VideoWriter('output_demo.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 15, (out_img.shape[1],out_img.shape[0]))

    for img in result_video:
        out.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    out.release()


################### break 1. see google docs for explaination till now. 

if __name__ == "__main__":
    main()