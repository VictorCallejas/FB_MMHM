from google.cloud import vision

import io
import pandas as pd

from multiprocessing import Pool
import multiprocessing as mp
from tqdm import *

def getVisionData(image_path):
    
    with io.open("../../data/raw/"+image_path, 'rb') as image_file:
        content = image_file.read()
    
    client = vision.ImageAnnotatorClient()
    
    image = vision.Image(content=content)
    
    web_det = client.web_detection(image=image).web_detection
    object_ann = client.object_localization(image=image).localized_object_annotations
    label_det = client.label_detection(image=image).label_annotations
    text_det = client.text_detection(image=image).text_annotations
    face_ann = client.face_detection(image=image).face_annotations
    
    best_guess_labels = []
    object_labels = []

    #Best guess labels
    if len(web_det.best_guess_labels) > 0:
        best_guess_labels = [bgl.label for bgl in web_det.best_guess_labels]
        
    web_thresh = 0.5
    web_ents = set()
    if len(web_det.web_entities)>0:
        for webent in web_det.web_entities:
            if webent.score > web_thresh:
                web_ents.add(webent.description)
        
    #web_entities
    web_entities = list(web_ents)
    
    #object labels   
    if len(object_ann)>0:
        object_labels = list(set([obj.name for obj in object_ann]))

    label_thresh = 0.6
    labels = set()
    if len(label_det)>0:
        for label in label_det:
            if label.score > label_thresh:
                labels.add(label.description)

    #labels        
    labels = list(labels)
    
    #texts
    if len(text_det) != 0:
        texts = text_det[0].description.split("\n")
    else:
        texts = []
    
    face_thresh = 3
    expressions = set()
    if len(face_ann)>0:
        for ann in face_ann:
            if ann.joy_likelihood >= face_thresh: expressions.add("joy")
            if ann.sorrow_likelihood >= face_thresh: expressions.add("sorrow")
            if ann.anger_likelihood >= face_thresh: expressions.add("anger")
            if ann.surprise_likelihood >= face_thresh: expressions.add("surprise")
            if ann.under_exposed_likelihood >= face_thresh: expressions.add("under exposed")
            if ann.blurred_likelihood >= face_thresh: expressions.add("blurred")

     #facial expressions
    expressions = list(expressions)
 
    #client.close()
    
    return {
        "image_path":image_path,
        "best_guess_labels":best_guess_labels,
        "web_entities":web_entities,
        "object_labels":object_labels,
        "image_labels":labels,
        "texts":texts,
        "expressions":expressions
    }

def main():

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="../../data/interim/google_vision_cred.json" # Google Vision API KEY

    test1 = pd.read_json('../../data/raw/test_seen.jsonl', lines=True)
    test2 = pd.read_json('../../data/raw/test_unseen.jsonl', lines=True)
    train = pd.read_json('../../data/raw/train.jsonl', lines=True)
    dev1 = pd.read_json('../../data/raw/dev_seen.jsonl', lines=True)
    dev2 = pd.read_json('../../data/raw/dev_unseen.jsonl', lines=True)

    frames = [test1, test2, train, dev1, dev2]

    concat_df = pd.concat(frames)

    image_list = concat_df['img'].unique()

    tot_len = len(image_list)
    print (tot_len)

    json_data_list = []
    
    with tqdm(total=len(image_list)) as pbar:
        with Pool(processes=10) as p:
            for vision_data in (p.imap_unordered(getVisionData, image_list)):
                if vision_data is not None:
                    json_data_list.append(vision_data) 
                pbar.update(1)

    p.close()
    p.join()

    
    annotations = pd.DataFrame(json_data_list)

    annotations.to_csv('../../data/external/google_vision.csv')

if __name__ == "__main__":
    main()