import os
from tqdm import tqdm
from Design2Code.data_utils.screenshot import take_screenshot
from gpt4v_utils import cleanup_response, encode_image, gpt_cost, extract_text_from_html, index_text_from_html
from gpt4v_utils import gemini_encode_image
import json
import google.generativeai as genai
import argparse
import retry
import shutil 
import base64
from PIL import Image
from io import BytesIO

@retry.retry(tries=2, delay=2)
def gemini_call(gemini_client, encoded_image, prompt):
    generation_config = genai.GenerationConfig(
        temperature=0.,
        candidate_count=1,
        max_output_tokens=4096,
    )
    
    response = gemini_client.generate_content([prompt, encoded_image], generation_config=generation_config)
    response.resolve()
    response = response.text
    response = cleanup_response(response)

    return response
def gemini_call_image(gemini_client, encoded_image, prompt):
    generation_config = genai.GenerationConfig(
        temperature=0.,
        candidate_count=1,
        max_output_tokens=4096,
    )

    # Call the API to generate content
    response = gemini_client.generate_content([prompt, encoded_image], generation_config=generation_config)
    response.resolve()

    # Assuming the response has an image data attribute
    image_data = response.image_data

    image = Image.open(BytesIO(base64.b64decode(image_data)))

    return image

def sketch_generation(gemini_client, image_file):
    sketch_prompt = "Generate a sketch for the following image. Only return image not include text."
    description_prompt = "Generate a description for the following image"
    
    ## encode image 
    image = gemini_encode_image(image_file)

    ## call Gemini
    sketch = gemini_call(gemini_client, image, sketch_prompt)
    description = gemini_call(gemini_client, image, description_prompt)

    return description, sketch
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, default='dataset/image2code')
    parser.add_argument('--output_dir', type=str, default='dataset/sketch2code')
    parser.add_argument('--file_name', type=str, default='all')
    args = parser.parse_args()

    ## load API Key
    with open("../api_key.json", "r") as f:
        api_key = json.load(f)
    
    ## set up gemini client
    genai.configure(api_key=api_key["gemini_api_key"])
    gemini_client = genai.GenerativeModel('gemini-pro-vision')

    ## specify file directory
    input_data_dir = "../" + args.input_data 
    output_data_dir = "../" + args.output_dir
    
    
    ## create cache directory if not exists
    os.makedirs(output_data_dir, exist_ok=True)

    # get the list of predictions already made
    existing_predictions = [item for item in os.listdir(output_data_dir) if item.endswith(".png")]
    print ("#existing predictions: ", len(existing_predictions))
    
    test_files = []
    if args.file_name == "all":
      test_files = [item for item in os.listdir(input_data_dir) if item.endswith(".png") and "_marker" not in item and item not in existing_predictions]
    else:
      test_files = [args.file_name]

    counter = 0
    for filename in tqdm(test_files):
        print (filename)
        try:
            description, sketch = sketch_generation(gemini_client, os.path.join(input_data_dir, filename))
            with open(os.path.join(output_data_dir, filename.replace(".png", ".txt")), "w") as f:
                f.write(description)
            with open(os.path.join(output_data_dir, filename), "w") as f:
                f.write(sketch)
            
            counter += 1
        except:
            continue 
    print ("#new predictions: ", counter)
            