### Table extraction from images

#### Top-level directory layout
    .  
    ├── png_image/              # Contains input images
    ├── constants.py            # Project-wide constants  
    ├── image_processing.py     # Functions to assist processing of images
    ├── img2table.py            # Main file. Everything unfolds here.
    ├── ReadMe.md               # You are here!
    ├── utils.py                # Utility functions  

#### Dependencies
- **Google Vision Library**  
You can follow [this guide](https://cloud.google.com/vision/docs/libraries#client-libraries-install-python).  
Please make sure you place the json file provided through the link in the root directory of this project and modify the constant `JSON_FILE` in file `constants.py`
- OpenCV
- Numpy 

#### How to run
- Run the following command in terminal:  
```python image2table.py```  
**NOTE**  
To change the input file being used, modify the variable `INPUT_FILE` in the file `constants.py`

#### Contributors
- [Pankaj Baranwal](https://github.com/Pankaj-Baranwal)
- [Sanjeev Dubey](https://github.com/getsanjeev)
- [Utkarsh Mittal](https://github.com/utkarshmttl)
