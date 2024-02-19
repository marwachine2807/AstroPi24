import math
from orbit import ISS
import time
from datetime import timezone 
from datetime import datetime, timedelta
from exif import Image
import cv2
from picamera import PiCamera 
from logzero import logger

SLEEP_TIME = 14 #sleep seconds
CONST_TIME_MINUTES = 9 #10 minutes of runtime + extra buffer time
R = 3671 #radius of the earth in km 
H = 408 #average altitude of the iss

cam = PiCamera()

cam.resolution = (4056, 3040)

def get_time(image):
    #time as image is taken
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        time_str = img.get("datetime_original")
        time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
    return time

def get_time_difference(image_1, image_2):
    #difference in time between two photos 
    time_1 = get_time(image_1)
    time_2 = get_time(image_2)
    time_difference = time_2 - time_1
    return time_difference.seconds

def convert_to_cv(image_1, image_2):
    image_1_cv = cv2.imread(image_1, 0)
    image_2_cv = cv2.imread(image_2, 0)
    return image_1_cv, image_2_cv

def calculate_features(image_1_cv, image_2_cv, feature_number):
    orb = cv2.ORB_create(nfeatures = feature_number)
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1_cv, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2_cv, None)
    return keypoints_1, keypoints_2, descriptors_1, descriptors_2

def calculate_matches(descriptors_1, descriptors_2):
    #matching areas 
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches):
    match_img = cv2.drawMatches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches[:100], None)
    resize = cv2.resize(match_img, (1600,600), interpolation = cv2.INTER_AREA)
    cv2.imshow('matches', resize)
    cv2.waitKey(0)
    cv2.destroyWindow('matches')

def find_matching_coordinates(keypoints_1, keypoints_2, matches):
    #coordinates of matching areas
    coordinates_1 = []
    coordinates_2 = []
    for match in matches:
        image_1_idx = match.queryIdx
        image_2_idx = match.trainIdx
        (x1,y1) = keypoints_1[image_1_idx].pt
        (x2,y2) = keypoints_2[image_2_idx].pt
        coordinates_1.append((x1,y1))
        coordinates_2.append((x2,y2))
    return coordinates_1, coordinates_2

def calculate_mean_distance(coordinates_1, coordinates_2):
    #average distance between matching points
    all_distances = 0
    merged_coordinates = list(zip(coordinates_1, coordinates_2))
    for coordinate in merged_coordinates:
        x_difference = coordinate[0][0] - coordinate[1][0]
        y_difference = coordinate[0][1] - coordinate[1][1]
        distance = math.hypot(x_difference, y_difference)
        all_distances = all_distances + distance
    return all_distances / len(merged_coordinates)

def calculate_speed_in_kmps(feature_distance, GSD, time_difference):
    #speed calculation
    distance = feature_distance * GSD / 100000
    speed = distance / time_difference
    return speed

def imgtime(image_1, image_2): 
    #time between photos
    time_difference = get_time_difference(image_1, image_2) # Get time difference between images
    return time_difference

def imgdist(image_1, image_2):
    #distance between photo featurs 
    image_1_cv, image_2_cv = convert_to_cv(image_1, image_2) # Create OpenCV image objects
    keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 1000) # Get keypoints and descriptors
    matches = calculate_matches(descriptors_1, descriptors_2) # Match descriptors
    coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)
    average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)
    return average_feature_distance

def convert(angle):
    #dms to decimal degrees
    decimalangle = angle.degrees
    return decimalangle

def lat():
        #lat where image taken
        coords = ISS.coordinates()
        lat = convert(coords.latitude)
        return lat

def lon():
    #long where image taken
        coords = ISS.coordinates()
        lon = convert(coords.longitude)
        return lon

def haversine(lat1, lon1, lat2, lon2):
        #Convert latitude and longitude from degrees to radians
        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)
        
        # Haversine formula to find angle at which the iss travels on an arc
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return c

def photo(photonum): 
    img = (f"issimg{photonum}.jpg")
    cam.capture(img)
    return img

endtime = datetime.now(timezone.utc) + timedelta(minutes = CONST_TIME_MINUTES) 

photonum = 1

image_1 = photo(photonum)



photonum = photonum + 1

lat1 = lat()
lon1 = lon()

#append speeds to a list
estimates = []
estimates2 = []

if __name__ == '__main__':
    # datetime.now(timezone.utc) in endtime
    while datetime.now(timezone.utc) < endtime and photonum < 39:
        begin = datetime.now(timezone.utc)
        
        try:
            #Finds the angle at which is passed in a time period of 10 seconds (since its path is of a circle)
            time.sleep(SLEEP_TIME)
            image_2 = photo(photonum)
            lat2 = lat()
            lon2 = lon()
            c = haversine(lat1, lon1, lat2, lon2)
            timediff = imgtime(image_1, image_2)
            speed1 = None
            try:
                distance = imgdist(image_1, image_2)
                speed1 = calculate_speed_in_kmps(distance, 12648, timediff)
            except Exception:
                print('Exception in lines 161-162')
            w = c / timediff #angular velocity w
            speed2 = w * (R + H) #tangential velocity v=w*r (circular motion) 
            if speed1 == None:
                speed1 = speed2
            estimates.append(speed1)
            estimates2.append(speed2)
            
            sum_estimates = sum(estimates)
            length_estimates =len(estimates)
            sum_estimates2 = sum(estimates2)
            length_estimates2 =len(estimates2)

            #take mean of list items and log into result.txt

            image_1 = image_2
            lat1 = lat2
            lon1 = lon2
            photonum += 1


            estimate_kmps = sum_estimates / length_estimates  # Replace with your estimate
            estimate2_kmps = sum_estimates2 / length_estimates2
            
            average_estimate = (estimate_kmps + estimate2_kmps) / 2 

            # Format the estimate_kmps to have a precision
            # of 5 significant figures
            estimate_kmps_formatted = "{:.4f}".format(average_estimate)

            # Create a string to write to the file
            output_string = estimate_kmps_formatted

            # Write to the file
            file_path = "result.txt"  # Replace with your desired file path
            with open(file_path, 'w') as file:
                file.write(output_string)

            print("Data written to", file_path)
        except Exception as e: #if error it will log 
            logger.error(f"Error in {e.__class__.__name__}, {e}")
            print('Catching exception')
