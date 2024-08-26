import Augmentor

def image_augmentation(string):         
  p = Augmentor.Pipeline(string)        

# se definesc parametrii operațiilor de augmentare
  p.flip_left_right(0.5)         
  p.rotate(0.7, 10, 10)          
  p.skew(0.4, 0.5)               
  p.zoom(probability = 0.2, min_factor = 1.1, max_factor = 1.5)      
  p.sample(1000)                 
  

