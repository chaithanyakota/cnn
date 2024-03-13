import numpy as np

class MaxPool2: 
    # Max pooling layer using a pool size of 2
    
    
    def iterate_regions(self, image): 
        '''
            - image is a 2d numpy array
        '''
        
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2
        
        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j
                
    def forward(self, input): 
        '''
            - input is a 3d numpy array with dimensions (h, w, num_filters)
        '''
        
        h, w, num_filters = input.shape
        
        output = np.zeros((h // 2, w // 2, num_filters))
        
        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))

        return output