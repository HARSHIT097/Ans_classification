import streamlit as st
from PIL import Image
#import cv2
#import re
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

dog_breed_labels = ['Affenpinscher', 'Afghan Hound', 'African Hunting Dog', 'Airedale', 'American Staffordshire Terrier', 'Appenzeller', 'Australian Terrier', 'Basenji', 'Basset', 'Beagle', 'Bedlington Terrier', 'Bernese Mountain Dog', 'Black-And-Tan Coonhound', 'Blenheim Spaniel', 'Bloodhound', 'Bluetick', 'Border Collie', 'Border Terrier', 'Borzoi', 'Boston Bull', 'Bouvier Des Flandres', 'Boxer', 'Brabancon Griffon', 'Briard', 'Brittany Spaniel', 'Bull Mastiff', 'Cairn', 'Cardigan', 'Chesapeake Bay Retriever', 'Chihuahua', 'Chow', 'Clumber', 'Cocker Spaniel', 'Collie', 'Curly-Coated Retriever', 'Dandie Dinmont', 'Dhole', 'Dingo', 'Doberman', 'English Foxhound', 'English Setter', 'English Springer', 'Entlebucher', 'Eskimo Dog', 'Flat-Coated Retriever', 'French Bulldog', 'German Shepherd', 'German Short-Haired Pointer', 'Giant Schnauzer', 'Golden Retriever', 'Gordon Setter', 'Great Dane', 'Great Pyrenees', 'Greater Swiss Mountain Dog', 'Groenendael', 'Ibizan Hound', 'Irish Setter', 'Irish Terrier', 'Irish Water Spaniel', 'Irish Wolfhound', 'Italian Greyhound', 'Japanese Spaniel', 'Keeshond', 'Kelpie', 'Kerry Blue Terrier', 'Komondor', 'Kuvasz', 'Labrador Retriever', 'Lakeland Terrier', 'Leonberg', 'Lhasa', 'Malamute', 'Malinois', 'Maltese Dog', 'Mexican Hairless', 'Miniature Pinscher', 'Miniature Poodle', 'Miniature Schnauzer', 'Newfoundland', 'Norfolk Terrier', 'Norwegian Elkhound', 'Norwich Terrier', 'Old English Sheepdog', 'Otterhound', 'Papillon', 'Pekinese', 'Pembroke', 'Pomeranian', 'Pug', 'Redbone', 'Rhodesian Ridgeback', 'Rottweiler', 'Saint Bernard', 'Saluki', 'Samoyed', 'Schipperke', 'Scotch Terrier', 'Scottish Deerhound', 'Sealyham Terrier', 'Shetland Sheepdog', 'Shih-Tzu', 'Siberian Husky', 'Silky Terrier', 'Soft-Coated Wheaten Terrier', 'Staffordshire Bullterrier', 'Standard Poodle', 'Standard Schnauzer', 'Sussex Spaniel', 'Tibetan Mastiff', 'Tibetan Terrier', 'Toy Poodle', 'Toy Terrier', 'Vizsla', 'Walker Hound', 'Weimaraner', 'Welsh Springer Spaniel', 'West Highland White Terrier', 'Whippet', 'Wire-Haired Fox Terrier', 'Yorkshire Terrier']

urllib.request.urlretrieve(
        "https://github.com/HARSHIT097/Ans_classification/blob/main/mobilnetV2-9000-images.h5?raw=true", "mobilnetV2-9000-images.h5")


### load model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
        """ Loads a saved model from a specified path. """
        print(f"Loading saved model from: {model_path}")

        model = tf.keras.models.load_model(model_path, custom_objects={"KerasLayer": hub.KerasLayer})
        return model
model = load_model('mobilnetV2-9000-images.h5')

#### prediction code
IMG_SIZE = 224
def process_single_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    # Convert the colour channel values from 0-225 values to 0-1 values
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Resize the image to our desired size (224, 244)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    return image

def process_image(image_path):
  """
  Takes an image file path and turns it into a Tensor.
  """
  # Read in image file
  image = tf.io.read_file(image_path)
  # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
  image = tf.image.decode_jpeg(image, channels=3)
  # Convert the colour channel values from 0-225 values to 0-1 values
  image = tf.image.convert_image_dtype(image, tf.float32)
  # Resize the image to our desired size (224, 244)
  image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
  return image

# Define the batch size, 32 is a good default, we will have 32 images in each batch
BATCH_SIZE = 32

# Create a function to turn data into batches
def create_data_batches(x, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=True):
  """
  x : array of images filepath
  y : array of images label
  batch_size : size of the batch we want to create
  valid_data, test_data : to specify the type of dataset we want to create

  Creates batches from pairs of image (x) and label (y).
  Shuffles the data if it's training data . Doesn't shuffle it if it's validation data.
  In test data we use only images (no labels)
  """
  # If the data is a test dataset, we don't have labels
  if test_data:
    # Get the slices of an array in the form of tensors, we only pass filepaths
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x)))
    # Preprocess each image object with our 'process_image' function
    data = data.map(process_image)
    # Turn our data into batches
    data_batch = data.batch(BATCH_SIZE)
    return data_batch

# Turn prediction probabilities into their respective label (easier to understand)
def get_pred_label(prediction_probabilities):
  """  Turns an array of prediction probabilities into a label. """
  return dog_breed_labels[np.argmax(prediction_probabilities)]

def top_possibilities(prediction_probabilities):
    top_lst = []
    for idx in list(np.argsort(prediction_probabilities)[::-1][:3]):
        top_lst.append(dog_breed_labels[idx])
    return top_lst
# title
st.markdown("<h1 style='text-align: center; color: black;'>Animal Breed Identification</h1>", unsafe_allow_html=True)

col1, col2 = st.beta_columns(2)
col1.subheader("Species")
selected_species = col1.selectbox(
     'Select the species you want to upload.',
     ('Dog', 'Buffalo', 'Camel', 'Goat'))

col2.subheader("Breed")
#col2.image(temp_img1, caption='Preprocessed Image.', use_column_width=True)

dog_breed_lst = ['Affenpinscher', 'Afghan Hound', 'African Hunting Dog', 'Airedale', 'American Staffordshire Terrier', 'Appenzeller', 'Australian Terrier', 'Basenji', 'Basset', 'Beagle', 'Bedlington Terrier', 'Bernese Mountain Dog', 'Black-And-Tan Coonhound', 'Blenheim Spaniel', 'Bloodhound', 'Bluetick', 'Border Collie', 'Border Terrier', 'Borzoi', 'Boston Bull', 'Bouvier Des Flandres', 'Boxer', 'Brabancon Griffon', 'Briard', 'Brittany Spaniel', 'Bull Mastiff', 'Cairn', 'Cardigan', 'Chesapeake Bay Retriever', 'Chihuahua', 'Chow', 'Clumber', 'Cocker Spaniel', 'Collie', 'Curly-Coated Retriever', 'Dandie Dinmont', 'Dhole', 'Dingo', 'Doberman', 'English Foxhound', 'English Setter', 'English Springer', 'Entlebucher', 'Eskimo Dog', 'Flat-Coated Retriever', 'French Bulldog', 'German Shepherd', 'German Short-Haired Pointer', 'Giant Schnauzer', 'Golden Retriever', 'Gordon Setter', 'Great Dane', 'Great Pyrenees', 'Greater Swiss Mountain Dog', 'Groenendael', 'Ibizan Hound', 'Irish Setter', 'Irish Terrier', 'Irish Water Spaniel', 'Irish Wolfhound', 'Italian Greyhound', 'Japanese Spaniel', 'Keeshond', 'Kelpie', 'Kerry Blue Terrier', 'Komondor', 'Kuvasz', 'Labrador Retriever', 'Lakeland Terrier', 'Leonberg', 'Lhasa', 'Malamute', 'Malinois', 'Maltese Dog', 'Mexican Hairless', 'Miniature Pinscher', 'Miniature Poodle', 'Miniature Schnauzer', 'Newfoundland', 'Norfolk Terrier', 'Norwegian Elkhound', 'Norwich Terrier', 'Old English Sheepdog', 'Otterhound', 'Papillon', 'Pekinese', 'Pembroke', 'Pomeranian', 'Pug', 'Redbone', 'Rhodesian Ridgeback', 'Rottweiler', 'Saint Bernard', 'Saluki', 'Samoyed', 'Schipperke', 'Scotch Terrier', 'Scottish Deerhound', 'Sealyham Terrier', 'Shetland Sheepdog', 'Shih-Tzu', 'Siberian Husky', 'Silky Terrier', 'Soft-Coated Wheaten Terrier', 'Staffordshire Bullterrier', 'Standard Poodle', 'Standard Schnauzer', 'Sussex Spaniel', 'Tibetan Mastiff', 'Tibetan Terrier', 'Toy Poodle', 'Toy Terrier', 'Vizsla', 'Walker Hound', 'Weimaraner', 'Welsh Springer Spaniel', 'West Highland White Terrier', 'Whippet', 'Wire-Haired Fox Terrier', 'Yorkshire Terrier']
dog_breed_lst = '\n'.join(dog_breed_lst)

buffalo_breed_list = 'Banni\nBargur\nBhadawari\nChilika'
if selected_species == 'Dog':
    s = dog_breed_lst
elif selected_species == 'Buffalo':
    s = buffalo_breed_list
else:
    s = 'List not updated'
txt = col2.text_area('Identifying from following breeds.', s)

# picture upload button
uploaded_file = st.file_uploader("Upload an image.", type="jpg", accept_multiple_files=False)

top_pos = 'Waiting for image'
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    #st.image(image, caption='Uploaded Image.', use_column_width=True)
    # prediction
    temp = image.save("temp_image.jpg")
    # Turn custom image into batch (set to test data because there are no labels)
    custom_data = create_data_batches(['temp_image.jpg'], test_data=True)

    # Make predictions on the custom data
    custom_preds = model.predict(custom_data)

    # Get custom image prediction labels
    predicted_labels = get_pred_label(custom_preds[0])

    st.latex(predicted_labels)
    top_pos = top_possibilities(custom_preds[0])

else:
    image = "default_image.jpg"

st.image(image, caption='Uploaded Image.', use_column_width=True)
st.subheader("Other possibilities:")
st.write( ', '.join(top_pos),  )

st.markdown("<h1 style='text-align: center; color: black;'> **** </h1>", unsafe_allow_html=True)
