import os
import xml.etree.ElementTree as ET
from PIL import Image
import pandas as pd

XML_FILE = "landmarks_raw/Conus_striolatus_landmarks.xml"
IMG_DIR = "images_uncropped/Conus_striolatus"
OUTPUT_DIR = "images_cropped/Conus_striolatus_cropped"
CSV_FILE = "landmarks_raw/Conus_striolatus_landmarks_cropped.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)
landmark_data = []

tree = ET.parse(XML_FILE)
root = tree.getroot()

for image_tag in root.iter("image"):
    img_file = image_tag.attrib["file"]
    img_path = os.path.join(IMG_DIR, img_file)

    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue

    img = Image.open(img_path)

    for i, box in enumerate(image_tag.findall("box")):
        left = int(box.attrib["left"])
        top = int(box.attrib["top"])
        width = int(box.attrib["width"])
        height = int(box.attrib["height"])

        cropped_img = img.crop((left, top, left + width, top + height))
        cropped_filename = f"{os.path.splitext(img_file)[0]}_specimen{i+1}.jpg"
        cropped_path = os.path.join(OUTPUT_DIR, cropped_filename)
        cropped_img.save(cropped_path)

        specimen_landmarks = {"image": cropped_filename}
        for part in box.findall("part"):
            name = part.attrib["name"]
            x = int(part.attrib["x"]) - left
            y = int(part.attrib["y"]) - top
            specimen_landmarks[f"x{name}"] = x
            specimen_landmarks[f"y{name}"] = y

        landmark_data.append(specimen_landmarks)

df = pd.DataFrame(landmark_data)
df.to_csv(CSV_FILE, index=False)
print(f"Landmarks saved to {CSV_FILE}")