import cv2 as cv
import numpy as np

from torch import load,tensor,float16,cuda,autocast,no_grad,from_numpy,unsqueeze,sigmoid
from torchvision.transforms import ToTensor,Normalize
from skimage import transform,filters

device = "cuda" if cuda.is_available() else "cpu"

def predict(model,image,Threshold = 0.5):
  with autocast(device_type=device, dtype=float16, enabled=True):
    with no_grad():
        size = image.shape

        image = transform.resize(image, (256, 256))
        image = from_numpy(image.transpose((2, 0, 1))).type(float16).to(device)
        
        tf = Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        image = tf(image)

        image = unsqueeze(image,0)

        pred = transform.resize(sigmoid(model(image)).to('cpu').squeeze(0).squeeze(0).numpy(),(size[0],size[1]))

        pred[pred>Threshold] = 1
        pred[pred<Threshold] = 0
        
        return pred


def main(path):
    # WORKS FOR NOW. NEEDS TO BE REFINED
    model = load(path)

    cap = cv.VideoCapture(0)

    # Assessing if the camera is open
    frames_count = 0

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    transform = ToTensor()

    # Reading
    while True:        
        
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Computing the predicted segmentation        
        pred = predict(model,frame)

        # SOME MORPHOLOGICAL TRANSFORMATIONS SHOULD BE APPLIED TO THE PREDICTION e.g Putting to 0 the connected components associated to a 
        # a low number of pixels to remove some predictions...

        background = np.copy(frame)

        # Blurring the image taken from the camera
        background = filters.gaussian(background,sigma = 10, channel_axis = -1)
        
        background = np.clip(background, 0.0, 1.0)
        background = (background * 255).astype(np.uint8)

        # Putting the pixels associated to the person detected to 0
        background[pred == 1] = 0

        # Putting the pixels associated to the background to 0        
        frame[pred == 0] = 0

        # Combining the two pictures
        frame = frame + background
     
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
          
        cv.imshow('frame', frame)
    
        # Closing
        if cv.waitKey(1) == ord('q'):
            break

if __name__ == '__main__':
    main("./model/model_full.pt")