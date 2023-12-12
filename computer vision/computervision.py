import os
import azure.ai.vision as cvsdk
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Load environment variables
load_dotenv()
endpoint = os.getenv('CV_ENDPOINT')
key = os.getenv('CV_KEY')

# Create a Vision Service
service_options = cvsdk.VisionServiceOptions(endpoint, key)

# Select an image to analyze
img_filename = "sample.jpg"
vision_source = cvsdk.VisionSource(filename=img_filename)

# Set image analysis options and features
analysis_options = cvsdk.ImageAnalysisOptions()
analysis_options.features = (
    cvsdk.ImageAnalysisFeature.CAPTION |
    cvsdk.ImageAnalysisFeature.DENSE_CAPTIONS |
    cvsdk.ImageAnalysisFeature.TAGS |
    cvsdk.ImageAnalysisFeature.OBJECTS
)

# Specify the language of the returned data
analysis_options.language = "en"

# Select gender-neutral captions
analysis_options.gender_neutral_caption = True

# Get the Image Analysis results
image_analyzer = cvsdk.ImageAnalyzer(service_options, vision_source, analysis_options)
result = image_analyzer.analyze()

if result.reason == cvsdk.ImageAnalysisResultReason.ANALYZED:
    # Print caption
    if result.caption is not None:
        print(f"\nCaption: '{result.caption.content}' (Confidence {result.caption.confidence:.4f})")

    # Print dense captions
    if result.dense_captions is not None:
        print("\nDense Captions:\n")
        for caption in result.dense_captions:
            print(f" '{caption.content}' (Confidence: {caption.confidence:.4f})")

    # Print tags
    if result.tags is not None:
        print("\nTags:\n")
        for tag in result.tags:
            print(f" '{tag.name}' (Confidence {tag.confidence:.4f})")

    # Print objects
    if result.objects is not None:
        img = Image.open(img_filename)
        img_height, img_width, img_ch = np.array(img).shape

        draw = ImageDraw.Draw(img)

        line_width = 3
        font_size = 18
        color = (0, 255, 0)

        print("\nObjects:\n")
        for detected_object in result.objects:
            print(f" '{detected_object.name}', {detected_object.bounding_box} (Confidence: {detected_object.confidence:.4f})")

            if detected_object.confidence > 0.5:
                left = detected_object.bounding_box.x
                top = detected_object.bounding_box.y
                height = detected_object.bounding_box.h
                width = detected_object.bounding_box.w

                shape = [(left, top), (left + width, top + height)]
                draw.rectangle(shape, outline=color, width=line_width)

                font = ImageFont.truetype("arial.ttf", font_size)
                draw.text((left, top - 20), f"{detected_object.name} ({detected_object.confidence * 100:.2f}%)", fill=color,
                          font=font)

        img.show()
        img.save("result.png", "PNG")
        print("Image saved!")

else:
    error_details = cvsdk.ImageAnalysisErrorDetails.from_result(result)
    print("Analysis failed.")
    print(f" Error reason: {error_details.reason}")
    print(f" Error code: {error_details.error_code}")
    print(f" Error message: {error_details.message}")
