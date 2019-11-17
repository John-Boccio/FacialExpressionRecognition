# Facial Expression Recognition
Implementation of Facial Expression Recognition for senior capstone project at Stony Brook University.

### Config.json setup
Within the project, there is a configuration file named `config.json`. This file provides various parameters that will be used within the project.

`config.json` Parameters:
```
{
  """
  Name of the project
  """
  "name": "facial_expression_recognition",  

  # Parameters for the data loading defined here
  "data_loader": {      

    """
    If true, the dataset objects will be look for pickled objects within metadata/ for the datasets being used. 
    If none are found, the datasets will be parsed and pickled after creation for future use. 
    If pickle is set to false, the datasets used will be parsed and not pickled.
    """
    "pickle": true,     

    """
    CK+ dataset parameters. 
    image_dir is the full path to the image directory of the dataset.
    emotion_dir is the full path to the emotion labels of the dataset.
    """
    "CK": {     # Parameters for CK+ dataset
      "image_dir": "/Users/johnboccio/Datasets/CK/cohn-kanade-images/",
      "emotion_dir": "/Users/johnboccio/Datasets/CK/Emotion/"
    },

    """
    FER2013 dataset parameters.
    csv_path is the full path to the fer2013.csv file.
    """
    "FER2013": {
      "csv_path": "/Users/johnboccio/Datasets/fer2013/fer2013.csv"
    },

    """
    ExpW dataset parameters.
    image_dir is the full path to the image directory.
    label_path is the full path to the label.lst file.
    """
    "expW": {
      "image_dir": "/Users/johnboccio/Datasets/ExpW/data/image/",
      "label_path": "/Users/johnboccio/Datasets/ExpW/data/label/label.lst"
    }

  }

}
```
    
