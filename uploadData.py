import shutil
import zipfile as zipfile
import json

class UploadData:
    
    def upload_data(fn, fc):
        identified_filetype = ""
        identified_file = ""
        if fn.endswith('.zip'):
            fn = fc.selected_filename.replace('.zip', '')
            with zipfile.ZipFile(fc.selected_path+"\\"+fc.selected_filename, 'r') as zipObj:
                zip_content = zipObj.namelist()
                for zip_content_item in zip_content:
                    identified_file += zip_content_item + ","
                    if zip_content_item.endswith('.mp4'):
                        zipObj.extract(zip_content_item, 'Data_Storage/Videos')
                        if identified_filetype != "":
                            identified_filetype += ", Video"
                        else:
                            identified_filetype += "Video"
                    elif zip_content_item.endswith('.csv') or zip_content_item.endswith('.xlsx') or zip_content_item.endswith('.txt'):
                        zipObj.extract(zip_content_item, 'Data_Storage/Dataset')
                        if identified_filetype != "":
                            identified_filetype += ", Dataset"
                        else:
                            identified_filetype += "Dataset"
                    elif zip_content_item.endswith('.file'):
                        zipObj.extract(zip_content_item, 'Data_Storage/Pretrained_models')
                        if identified_filetype != "":
                            identified_filetype += ", Pretrained Model"
                        else:
                            identified_filetype += "Pretrained"
                    elif zip_content_item.endswith('.json') or zip_content_item.endswith('.vtt'):
                        zipObj.extract(zip_content_item, 'Data_Storage/Captions')
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
            identified_file = fc.selected_filename
            if fn.endswith('.mp4'):  
                shutil.copyfile(fc.selected_path+"/"+fc.selected_filename, 'Data_Storage/Videos/'+fc.selected_filename)
                identified_filetype = "Videos" 
            elif fn.endswith('.csv') or fn.endswith('.xlsx') or fn.endswith('.txt'):
                shutil.copyfile(fc.selected_path+"/"+fc.selected_filename, 'Data_Storage/Dataset/'+fc.selected_filename)
                identified_filetype = "Dataset"    
            elif fn.endswith('.file'):
                shutil.copyfile(fc.selected_path+"/"+fc.selected_filename, 'Data_Storage/Pretrained_Models/'+fc.selected_filename)
                identified_filetype = "Pretrained_Model"
            elif fn.endswith('.json') or fn.endswith('.vtt'):
                shutil.copyfile(fc.selected_path+"/"+fc.selected_filename, 'Data_Storage/Captions/'+fc.selected_filename)
                identified_filetype = "Captions"
            else:
                shutil.copyfile(fc.selected_path+"/"+fc.selected_filename, 'Data_Storage/'+fc.selected_filename)
                identified_filetype = "Others"

        return identified_filetype, identified_file