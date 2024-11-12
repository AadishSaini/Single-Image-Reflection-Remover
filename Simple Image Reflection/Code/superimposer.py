import cv2 

class superimpose:
    def __init__(self, blur_func="gauss"):
        self.blur_func = blur_func
    
    def gen(self, base, reflection_layer, alpha=0.3):
        base, reflection_layer = self.resize(base, reflection_layer)

        if(self.blur_func == "gauss"):
            gaus = cv2.GaussianBlur(reflection_layer, (7,7), 0)
        else:
            gaus = cv2.medianBlur(reflection_layer)
        res = cv2.addWeighted(gaus, alpha, base, 1 - alpha, 0)
        return res

    def resize(self, img1, img2):

        # Get dimensions of each image
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Determine the smallest dimensions
        min_height = min(h1, h2)
        min_width = min(w1, w2)

        # Resize both images to the smallest dimensions
        img1 = cv2.resize(img1, (min_width, min_height), interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, (min_width, min_height), interpolation=cv2.INTER_AREA)

        # Save the resized images
        return [img1, img2]


