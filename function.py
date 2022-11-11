from ipywidgets import widgets
from IPython.display import display, HTML
import ipywidgets as widgets
from ipywidgets import Button
from IPython.display import display, HTML
from ipyfilechooser import FileChooser
import zipfile as zipfile
import shutil
import os
import json
import io
import base64



def getVideoList():
    video_folder = os.listdir('Data_Storage/Videos')
    list_of_video = {}
    for video in video_folder:
        # remove hidden file especially for macOS
        if not video.endswith('.mp4'):
            list_of_video[video] = video
    return list_of_video


# read action and frame object from file
def readCaptionFile(filename,videoName):
    # Opening JSON file
    f = open(filename)

    # returns JSON object as 
    # a dictionary
    data = json.load(f)

    # Iterating through the json
    # Closing file
    f.close()
    return data[videoName]["actions"]

# convert frame to time
def convertFrameToTime(frame):
    seconds = int(frame/25)
    minutes = "00"
    if seconds >= 60:
        minutes = str(seconds // 60)
        seconds = seconds % 60
    if len(minutes) == 1:
        minutes = "0" + minutes
    seconds = str(seconds)
    #may need handle hour
    if len(seconds) == 1:
        seconds = "0" + seconds 
    return (minutes + ":" + seconds + ".000")

# read reference text from txt file
def readReferenceFile(refFile):
    referenceDict = {}
    with open(refFile) as f:
        lines = f.readlines()
    for i in lines:
        x = i.split()
        referenceDict[str(x[0])] = x[1]
    return referenceDict

# create caption file
def formatCaptionFile(captionList, reference, captionPath):
    start = "WEBVTT\n\n"
    captions = []
    for i in captionList:
        text = reference[str(i[0])]
        lines = convertFrameToTime(i[1]) + " --> " + convertFrameToTime(i[2]) + "\n" + text + "\n\n"
        captions.append(lines)
    f = open(captionPath, "w")
    f.write(start)
    f.writelines(captions)
    f.close()


# def play_video(video_src,caption_src):
#     video = io.open(video_src, 'r+b').read()
#     encoded = base64.b64encode(video)
#     return(HTML(data='''<video width="650" height="360" controls>
#         <source src="data:video/mp4;base64,{0}" type="video/mp4" />
#         <track kind="captions" src={1} srclang="en" label="English" default>
#         </video>'''.format(encoded.decode('ascii'),caption_src)))

def play_video(fc):
    
    src = "./Data_Storage/Videos/" + fc.selected_filename
    video = io.open(src, 'r+b').read()
    encoded = base64.b64encode(video)
    #location of reference are place at root
    ref = readReferenceFile('./Data_Storage/all_labels.txt')
    # may need change the caption path to dynamic
    
    caption_location = './Data_Storage/Captions' + fc.selected_filename + ".vtt"
    # model result file should be some directory, here using root 
    captionList = readCaptionFile('./Data_Storage/smarthome_CS_51.json',fc.selected_filename)
    formatCaptionFile(captionList, ref, caption_location)
    return(HTML(data='''<video width="650" height="360" controls>
        <source src="data:video/mp4;base64,{0}" type="video/mp4" />
        <track kind="captions" srclang="en" label="English" default>
        </video>'''.format(encoded.decode('ascii'), caption_location)))
        
def submit_upload(button, fc, output):
    fn = fc.selected_filename
    identified_filetype = ""
    if fn.endswith('.zip'):
        fn = fc.selected_filename.replace('.zip', '')
        with zipfile.ZipFile(fc.selected_path+"\\"+fc.selected_filename, 'r') as zipObj:
            zip_content = zipObj.namelist()
            for zip_content_item in zip_content:
                if zip_content_item.endswith('.mp4'):
                    zipObj.extract(zip_content_item, 'Data_Storage\\Videos')
                    if identified_filetype != "":
                        identified_filetype += ", Video"
                    else:
                        identified_filetype += "Video"
                elif zip_content_item.endswith('.csv') or zip_content_item.endswith('.xlsx') or zip_content_item.endswith('.txt'):
                    zipObj.extract(zip_content_item, 'Data_Storage\\Dataset')
                    if identified_filetype != "":
                        identified_filetype += ", Dataset"
                    else:
                        identified_filetype += "Dataset"
                elif zip_content_item.endswith('.file'):
                    zipObj.extract(zip_content_item, 'Data_Storage\\Pretrained_models')
                    if identified_filetype != "":
                        identified_filetype += ", Pretrained Model"
                    else:
                        identified_filetype += "Pretrained"
                elif zip_content_item.endswith('.json') or zip_content_item.endswith('.vtt'):
                    zipObj.extract(zip_content_item, 'Data_Storage\\Captions')
                    if identified_filetype != "":
                        identified_filetype += ", Caption"
                    else:
                        identified_filetype += "Caption"
                else:
                    zipObj.extract(zip_content_item, 'Data_Storage')
                    if identified_filetype != "":
                        identified_filetype += ", Others"
                    else:
                        identified_filetype += "Others"


    else:
        identified_filetype = ""
        if fn.endswith('.mp4'):  
            shutil.copyfile(fc.selected_path+"\\"+fc.selected_filename, 'Data_Storage\\Videos\\'+fc.selected_filename)
            identified_filetype += ", Videos" 
        elif fn.endswith('.csv') or fn.endswith('.xlsx') or fn.endswith('.txt'):
            shutil.copyfile(fc.selected_path+"\\"+fc.selected_filename, 'Data_Storage\\Dataset\\'+fc.selected_filename)
            identified_filetype += "Dataset"    
        elif fn.endswith('.file'):
            shutil.copyfile(fc.selected_path+"\\"+fc.selected_filename, 'Data_Storage\\Pretrained_Models\\'+fc.selected_filename)
            identified_filetype += "Pretrained_Model"
        elif fn.endswith('.json') or fn.endswith('.vtt'):
            shutil.copyfile(fc.selected_path+"\\"+fc.selected_filename, 'Data_Storage\\Captions\\'+fc.selected_filename)
            identified_filetype += ", Captions"
        else:
            shutil.copyfile(fc.selected_path+"\\"+fc.selected_filename, 'Data_Storage\\'+fc.selected_filename)
            identified_filetype += ", Others"
    with output:
        print("File type identified to be/include: " + identified_filetype)

def display_inference_model(fc, output):
    rs = fc.selected_path+fc.selected_filename
    with output:
        print(rs + "has been selected.")
    