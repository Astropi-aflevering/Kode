#Go here to the URL shown beneath and navigate to the online  test tool under the headline section:
# 'Accessing the Astro Pi Replay Tool online ' the online tool is called ' Astro Pi Replay Tool' 

# https://projects.raspberrypi.org/en/projects/mission-space-lab-creator-guide/2
# Import the Camera class from the picamera-zero module
#der skal sættes et standard format for billederne og filerne, som gør at de kan indeholde tidspunkter, lokationer og type af kamera
from exif import Image
from datetime import datetime
import cv2
import math

from picamzero import Camera

# Create an instance of the Camera class
cam = Camera()

# Capture an image

cam.take_photo("image1.jpg")
cam.take_photo("image2.jpg")
cam.take_photo("image2.jpg")
cam.take_photo("image2.jpg")
cam.take_photo("image2.jpg")
#der skal tages billeder af jorden fra rummet det
#de her billeder er dem vi skal se nærmere på omlidt

cam.take_photo("image2.jpg")


image_1 = 'image1.jpg'
image_2 = 'image2.jpg'
#her har man navngivet billederne, så man blot kan kalde dem for image_1 og image_2

def get_time(image):
#billderne skal åbnes og herefter blive konverteret til et objekt-
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        for data in img.list_all():
            print(data)
#Her har man nu mulighed for at se på al Exif dataen som er gemt i billedet
            time_str = img.get("datetime_original")
        time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
    return time
#Man skal gemme tidsdataen, så der kan blive lavet udregninger på det
#print(get_time('photo_0683.jpg'))
#nu skal vi finde tidsforskellen for de to billeder
def get_time_difference(image_1, image_2):
#man får tiden fra exif dataen og så trækker man dem fra hinanden for at finde tidsforskellen
    time_1 = get_time(image_1)
    time_2 = get_time(image_2)
    time_difference = time_2 - time_1
#man vil gerne have tiden i sekunder, så det gør koden nedenunder for os
    return time_difference.seconds
#    print(time_difference)
#nu skal vi have fundet de "matchende" egenskaber ved de to billeder
#her skal vi bruge cv2 pakken
#billederne skal konverteres til OpenCV objekter, så de kan blive processed, derfor skal man tilføje en funktion som
#tager de to billeder og retunerer dem som objekter
def convert_to_cv(image_1, image_2):
    image_1_cv = cv2.imread(image_1, 0)
    image_2_cv = cv2.imread(image_2, 0)
    return image_1_cv, image_2_cv
#Disse "OpenCV" objekter kan blive brugt af ORB algoritmen. Denne algoritme vil opfange "keypoints" på et eller flere billeder
#hvis billederne er tæt på at være ens, vil de samme "keypoints" blive opdaget, selvom de har rykket sig eller ændret sig lidt
#ORB kan faktisk også laver "descriptors" til disse "keypoints", de vil indeholde informationer som position, størrelse, rotation og lysstyrke
#Ved at sammenligne disse "descriptors" mellem "keypoints" kan man beregne ændringerne fra det ene billede til det andet
#vi skal nu derfor finde "keypointsne" og "descriptors" for de 2 billeder
def calculate_features(image_1, image_2, feature_number):
    orb = cv2.ORB_create(nfeatures = feature_number)
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1_cv, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2_cv, None)
    return keypoints_1, keypoints_2, descriptors_1, descriptors_2
#Den nemmeste måde at sammenligne "descriptors" for de 2 billeder er ved at bruge brute force
#Brute force er en algoritme som prøver alle mulige kombinationer ved at matche en "descriptor fra det ene billede med alle de andre "descriptors" fra det andet billede
#Her bliver der derfor skrevet en funktion som tager de set af descriptors og prøver at finde matches vha. brute force
def calculate_matches(descriptors_1, descriptors_2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches
#Man kan også få spillet til at vise matches det er gjort i nedenstående kode
def display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches):
    match_img = cv2.drawMatches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches[:100], None)
#Her bliver billederne give en ny størrelse og sat op side om side med linjer mellem matches    
    resize = cv2.resize(match_img, (1600,600), interpolation = cv2.INTER_AREA)
    cv2.imshow('matches', resize)
#Man skal også kunne lukke billedet ned igen, derfor har man gjort at hvis man trykker på "0" lukker det ned    
    cv2.waitKey(0)
    cv2.destroyWindow('matches')

#nu skal vi finde koordinaterne til matches
#først laver vi en funktion som tager de to sæt af keypoints og lister af matches som "arguments"
def find_matching_coordinates(keypoints_1, keypoints_2, matches):
#nu skal man lave 2 tomme lister som skal indeholde koordinaterne for hver af de matchende egenskaber for billederne    
    coordinates_1 = []
    coordinates_2 = []
#listerne for matches indeholder mange objekter.
#man kan køre gennem listen gentagene gange for at finde koordinaterne til match på billedet
#Derfor tilføjer vi et loop, for at hente koordinaterne til hver match
    for match in matches:
        image_1_idx = match.queryIdx
        image_2_idx = match.trainIdx
        (x1,y1) = keypoints_1[image_1_idx].pt
        (x2,y2) = keypoints_2[image_2_idx].pt
#så skal vi have tilføjet koordinaterne til de to koordinatlister
        coordinates_1.append((x1,y1))
        coordinates_2.append((x2,y2))
#de to lister kan nu blive retuneret
    return coordinates_1, coordinates_2

def calculate_mean_distance(coordinates_1, coordinates_2):
    all_distances = 0
    merged_coordinates = list(zip(coordinates_1, coordinates_2))
    for coordinate in merged_coordinates:
        x_difference = coordinate[0][0] - coordinate[1][0]
        y_difference = coordinate[0][1] - coordinate[1][1]
        distance = math.hypot(x_difference, y_difference)
        all_distances = all_distances + distance
    return all_distances / len(merged_coordinates)
    #print(coordinates_1[0])
    #print(coordinates_2[0])
    #print(merged_coordinates[0])
def calculate_speed_in_kmps(feature_distance, GSD, time_difference):
    distance = feature_distance * GSD / 100000
    speed = distance / time_difference
    return speed


#Her kører programmet
image_1_cv, image_2_cv = convert_to_cv(image_1, image_2) # Create OpenCV image objects

time_difference = get_time_difference(image_1, image_2) # Get time difference between images
image_1_cv, image_2_cv = convert_to_cv(image_1, image_2) # Create OpenCV image objects
keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 1000) # Get keypoints and descriptors
matches = calculate_matches(descriptors_1, descriptors_2) # Match descriptors
#display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches) # Display matches

coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)
average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)

speed = calculate_speed_in_kmps(average_feature_distance, 12648, time_difference)


#timedif=get_time_difference('image1.jpg', 'image2.jpg')


# Format the estimate_kmps to have a precision
# of 5 significant figures
#timedifFormatted = "{:.3f}".format(timedif)

#file_path = "result.txt"  # Replace with your desired file path
#with open(file_path, 'w') as file:
#    file.write(timedifFormatted)

estimate_kmps = speed  # Replace with your estimate

# Format the estimate_kmps to have a precision
# of 5 significant figures
estimate_kmps_formatted = "{:.3f}".format(estimate_kmps)

# Create a string to write to the file
output_string = estimate_kmps_formatted
#print(output_string)
# Write to the file
file_path = "result.txt"  # Replace with your desired file path
with open(file_path, 'w') as file:
    file.write(output_string)

#print("Data written to", file_path)
