import numpy as np
import pandas as pd
import torch 
import random
import matplotlib.pyplot as plt

class utils():
    
    
    def __init__(self):
        
        self.move_mapping = {
                (-1.0, 0.0, 0.0, 0.0): 1, 
                (-1.0, 0.0, 0.0, 1.0): 1,   # Right
                
                (0.0, -1.0, 0.0, 0.0): 2, 
                (0.0, -1.0, 0.0, 1.0): 2,   # Forward
                
                (1.0, 0.0, 0.0, 0.0): 3, 
                (1.0, 0.0, 0.0, 1.0): 3,    # Left
                
                (0.0, 1.0, 0.0, 0.0): 4, 
                (0.0, 1.0, 0.0, 1.0): 4,    # Backwards
                
                (0.0, 0.0, 1.0, 0.0): 5, 
                (0.0, 0.0, 1.0, 1.0): 5,    # Up
                
                (0.0, 0.0, -1.0, 0.0): 6, 
                (0.0, 0.0, -1.0, 1.0): 6,   # Down
                
                (0.0, 0.0, 0.0, 1.0): 7,    # Close
                (0.0, 0.0, 0.0, 0.0): 8     # Open  
            }
        
        self.flipped_map = {
                1: (-1.0, 0.0, 0.0, 0.0),  # Right
                2: (0.0, -1.0, 0.0, 0.0),  # Forward
                3: (1.0, 0.0, 0.0, 0.0),    # Left
                4: (0.0, 1.0, 0.0, 0.0),    # Backwards
                5: (0.0, 0.0, 1.0, 0.0),    # Up
                6: (0.0, 0.0, -1.0, 0.0),  # Down
                7: (0.0, 0.0, 0.0, 1.0),   # Close
                8: (0.0, 0.0, 0.0, 0.0)    # Open
            }
    
    def move2int(self, move_array):
        
        val_tuple = tuple(move_array.tolist())
        ret_int_val = self.move_mapping.get(val_tuple, 1) 
        
        return ret_int_val
        
    def int2move(self, val, closed=False):
        
        move = np.array(self.flipped_map.get(val))
        
        if closed:
            move[3] = 1
             
        return move
    
    def load_data(self): 
           
        image_input = torch.load('imageTensorExtrav4.pt')
        move_inputs_y = torch.load('moveTensorExtrav4.pt')
        arm_input = torch.load('armTensorExtrav4.pt')
        
        return image_input, move_inputs_y, arm_input
    
    def display_data(self, move_inputs_y, image_input):
        
        random_index = random.randint(0, move_inputs_y.size(0))
        print("Index: ", random_index)
        print("Move: ", move_inputs_y[random_index])
        plt.imshow(image_input[random_index].permute(1,2,0).int().detach().numpy())
