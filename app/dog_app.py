import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Constants
MODEL_FILE = "../saved_models/weights.best.Resnet50.hdf5"
FACE_CASCADE_XML = "../haarcascades/haarcascade_frontalface_alt.xml"


# Define ResNet50 model
ResNet50_model = ResNet50(weights="imagenet")
my_model = load_model(MODEL_FILE)


# Define function to detect faces in an image using a pre-trained face detector
def detect_faces(img):
    """
    Uses a pre-trained face detector to detect faces in an image.

    Args:
        img (numpy.ndarray): An image as a numpy array.

    Returns:
        faces (list): A list of tuples representing the coordinates of each detected face.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray)


# Extract pre-trained face detector
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_XML)

dog_names = [
    "Affenpinscher",
    "Afghan hound",
    "Airedale terrier",
    "Akita",
    "Alaskan malamute",
    "American eskimo dog",
    "American foxhound",
    "American staffordshire terrier",
    "American water spaniel",
    "Anatolian shepherd dog",
    "Australian cattle dog",
    "Australian shepherd",
    "Australian terrier",
    "Basenji",
    "Basset hound",
    "Beagle",
    "Bearded collie",
    "Beauceron",
    "Bedlington terrier",
    "Belgian malinois",
    "Belgian sheepdog",
    "Belgian tervuren",
    "Bernese mountain dog",
    "Bichon frise",
    "Black and tan coonhound",
    "Black russian terrier",
    "Bloodhound",
    "Bluetick coonhound",
    "Border collie",
    "Border terrier",
    "Borzoi",
    "Boston terrier",
    "Bouvier des flandres",
    "Boxer",
    "Boykin spaniel",
    "Briard",
    "Brittany",
    "Brussels griffon",
    "Bull terrier",
    "Bulldog",
    "Bullmastiff",
    "Cairn terrier",
    "Canaan dog",
    "Cane corso",
    "Cardigan welsh corgi",
    "Cavalier king charles spaniel",
    "Chesapeake bay retriever",
    "Chihuahua",
    "Chinese crested",
    "Chinese shar-pei",
    "Chow chow",
    "Clumber spaniel",
    "Cocker spaniel",
    "Collie",
    "Curly-coated retriever",
    "Dachshund",
    "Dalmatian",
    "Dandie dinmont terrier",
    "Doberman pinscher",
    "Dogue de bordeaux",
    "English cocker spaniel",
    "English setter",
    "English springer spaniel",
    "English toy spaniel",
    "Entlebucher mountain dog",
    "Field spaniel",
    "Finnish spitz",
    "Flat-coated retriever",
    "French bulldog",
    "German pinscher",
    "German shepherd dog",
    "German shorthaired pointer",
    "German wirehaired pointer",
    "Giant schnauzer",
    "Glen of imaal terrier",
    "Golden retriever",
    "Gordon setter",
    "Great dane",
    "Great pyrenees",
    "Greater swiss mountain dog",
    "Greyhound",
    "Havanese",
    "Ibizan hound",
    "Icelandic sheepdog",
    "Irish red and white setter",
    "Irish setter",
    "Irish terrier",
    "Irish water spaniel",
    "Irish wolfhound",
    "Italian greyhound",
    "Japanese chin",
    "Keeshond",
    "Kerry blue terrier",
    "Komondor",
    "Kuvasz",
    "Labrador retriever",
    "Lakeland terrier",
    "Leonberger",
    "Lhasa apso",
    "Lowchen",
    "Maltese",
    "Manchester terrier",
    "Mastiff",
    "Miniature schnauzer",
    "Neapolitan mastiff",
    "Newfoundland",
    "Norfolk terrier",
    "Norwegian buhund",
    "Norwegian elkhound",
    "Norwegian lundehund",
    "Norwich terrier",
    "Nova scotia duck tolling retriever",
    "Old english sheepdog",
    "Otterhound",
    "Papillon",
    "Parson russell terrier",
    "Pekingese",
    "Pembroke welsh corgi",
    "Petit basset griffon vendeen",
    "Pharaoh hound",
    "Plott",
    "Pointer",
    "Pomeranian",
    "Poodle",
    "Portuguese water dog",
    "Saint bernard",
    "Silky terrier",
    "Smooth fox terrier",
    "Tibetan mastiff",
    "Welsh springer spaniel",
    "Wirehaired pointing griffon",
    "Xoloitzcuintli",
    "Yorkshire terrier",
]


def face_detector(img_path):
    """
    Return True if face is detected in image stored at img_path
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def path_to_tensor(img_path):
    """
    Read an image file at img_path and return a numpy array
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def dog_detector(img_path):
    """
    Return True if a dog is detected in the image stored at img_path
    """
    img = path_to_tensor(img_path)
    img = preprocess_input(img)
    prediction = np.argmax(ResNet50_model.predict(img))
    return (prediction <= 268) & (prediction >= 151)


def extract_Resnet50(tensor):
    return ResNet50_model.predict(preprocess_input(tensor))


def Resnet50_predict_breed(img_path):
    """
    Return the predicted dog breed for picture stored at img_path
    """
    # extract bottleneck features
    tensor = path_to_tensor(img_path)
    bottleneck_feature = extract_Resnet50(tensor)
    # obtain predicted vector
    predicted_vector = ResNet50_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


def dog_human_detector(img_path):
    """
    Return a string explaining whether the image contains a human
    face or dog, and the corresponding breed.
    """
    if face_detector(img_path):
        return f"This person resembles a {Resnet50_predict_breed(img_path)}"
    elif dog_detector(img_path):
        return f"This dog is a {Resnet50_predict_breed(img_path)}"
    else:
        return "Neither a dog nor a human face detected in this picture"
