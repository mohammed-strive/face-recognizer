import argparse
import logging
import os
import requests
import zipfile


model_dict = {
    "20170511-185253": "0B5MzpY9kBtDVOTVnU3NIaUdySFE",
}

def download_and_extract_model(model_name, data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    file_id = model_dict[model_name]
    destination = os.path.join(data_dir, (model_name+".zip"))

    if not os.path.exists(destination):
        print("Downloading model to %s" % destination)
        download_file_from_google_drive(file_id, destination)
        with zipfile.ZipFile(destination, 'r') as zip:
            print("Extracting model to %s" % data_dir)
            zip.extractall(data_dir)

def download_file_from_google_drive(file_id, destination):
    URL = "https://google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, val in response.cookies.items():
        if key.startswith("download_warning"):
            return val
        
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as fh:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: fh.write(chunk)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--model-dir', type=str, action="store",
        dest="model_dir", help="path to model protobuf graph")

    args = parser.parse_args()

    download_and_extract_model("20170511-185253", args.model_dir)