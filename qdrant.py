from qdrant_client import QdrantClient,models
import cv2
import os
import sys
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import json
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct



client = QdrantClient("localhost", port=6333)


client.create_collection(
    collection_name="jas",
    vectors_config=VectorParams(size=25088,distance=models.Distance.COSINE,),
)

collection=client.get_collection(collection_name="jas")
print(collection)
  # Print the first 5 vectors
for i in range(5):
    try:
      point = collection.PointStruct(i)  # Retrieve data point by ID
      if point:
        vector = point.vector
        print(f"Sample Vector {i+1}: {vector}")
      else:
        print(f"Point with ID {i} not found in the collection.")
    except Exception as e:
      print(f"Error retrieving vector {i}: {e}")


global l1
count=0
l1 = []
array_of_arrays=() # New list to store distinct embeddings

# Function to detect faces using MTCNN
def detect_faces(image):
    detector = MTCNN()
    detections = detector.detect_faces(image)
    return detections


# Store embeddings with image information

# Function to extract and save face regions with embeddings
def extract_and_save_faces_with_embeddings(image_path, model, save_dir="extracted_faces/",distance_threshold=0.1):
    global count
    global array_of_arrays
    image = cv2.imread(image_path)
    detections = detect_faces(image)

    if detections:
        for i, detection in enumerate(detections):
            x, y, w, h = detection['box']  # Extract bounding box
            face_roi = image[y:y+h, x:x+w]

            img = face_roi.copy()
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32)
            img = preprocess_input(img)
            img = np.expand_dims(img, axis=0)

            try:
                features = model.predict(img)
                embedding = features.flatten()
                filename, ext = os.path.splitext(os.path.basename(image_path))
                png=".png"
                output_filename = f"{count}{png}"
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, output_filename)
                cv2.imwrite(save_path, face_roi)
                l1.append(embedding.tolist()) 
                print(l1)
                print(embedding)
                array_of_arrays = np.array(l1)
                print(array_of_arrays[i])
                operation_info = client.upsert(
                    collection_name="jas",
                    wait=True,
                    points=[
                        PointStruct(id=count, vector=embedding, payload={"image": image_path}),
                    ],
                )
                count=count+1
                print("jekl")
                print(operation_info)

                print(f"Inserted embedding for data item: {i+1}")
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")


# Directory containing images
image_dir = "test/"
save_dir = "extracted_faces/"

# Load the pre-trained VGG16 model
model = VGG16(weights='imagenet', include_top=False)

# Process each image in the directory
for filename in os.listdir(image_dir):
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
        image_path = os.path.join(image_dir, filename)
        extract_and_save_faces_with_embeddings(image_path, model, save_dir,distance_threshold=0.1)

print("Finished processing images. Extracted faces and embeddings saved to", save_dir)
def perform_similarity_search(client, l1, top_k):
    """Performs a similarity search using the reference image and displays top k similar images."""
    if l1:
        reference_embedding = l1[2]
        search_results = client.search(
            collection_name="jas",
            query_vector=reference_embedding,
            limit=top_k # Retrieve top k most similar
        )
        print(search_results)
        similar_image_paths = []
        for result in search_results:
            payload = result.payload
            similar_image_path = payload["image"]
            similar_image_paths.append(similar_image_path)
            print(f"{similar_image_path}")  # Print the image path directly
            print(payload)
        print(f"\nTop {top_k} Similar Images:")
        for i, image_path in enumerate(similar_image_paths):
            print(f"{i+1}. {image_path}")

            # Display the images using a suitable library (e.g., OpenCV or matplotlib)
            # Implement image display logic here (outside this function for modularity)
            image = cv2.imread(image_path)  # Using OpenCV as an example
            cv2.imshow("Similar Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# Assuming you have already populated l1 with embeddings
perform_similarity_search(client, l1, top_k=7)

# Connect to Qdrant and get the "jas" collection
  # collection_name = "jas"
  # collection = client.get_collection(collection_name)
  # if not collection:
  #   print(f"Collection '{collection_name}' not found in Qdrant.")
  #   exit()

  # # Print the first 5 vectors
  # for i in range(5):
  #   try:
  #     vector = collection.get(i)["vector"]
  #     print(f"Sample Vector {i+1}: {vector}")
  #   except KeyError:
  #     print(f"Error retrieving vector {i}. Skipping.")
  
# distinct=[]
# if(len(distinct)==0):
#     distinct.append(l1[0]) 
# for i in range(1,16):
#     # for j in range(1,16): 
#         input1=l1[i]
#         input2=l1[1]
#         dot_product = np.dot(input1, input2)
#         # Calculate the magnitudes of each vector
#         magnitude_a = np.linalg.norm(input1)
#         magnitude_b = np.linalg.norm(input2)
#         # Compute the cosine similarity
#         cosine_similarity = dot_product / (magnitude_a * magnitude_b)

#         # print(i-1,cosine_similarity)
#         threshold=0.4
#         for j in range(len(distinct)):
#             # if(len(distinct)==0):
#             #     distinct.append(l1[0]) 
#             if(cosine_similarity<threshold):
#                 distinct.append(l1[i]) 
#                 print(j)
                
# print(len(distinct))
