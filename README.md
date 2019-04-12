# EEG-ANALYSIS-FOR-MOTOR-IMAGERY-APPLICATION
In this project I have implemented two different feature extraction techniques for EEG classificatio- Common Spatial Pattern(CSP) and Discrete Wavelet Transforms.
The success of machine learning algorithms is considerably dependent on the features provided to them. I have implemented these techniques to observe the differences in classification accuracy as one changes the features.
I have also implemented an additional feature extraction technique which is a modification of CSP called Filter-Band CSP (FB-CSP). FBCSP uses a combination of spatial patterns from multiple frequency bands. This combination is formed using the mutual information of the features from multiple bands and their respective labels. For more information please follow the resources below. The objective of is to identify right hand motion vs right leg motion.

Note: The parameters for the classifiers havent been cross validated yet. 

One of the problems faced while decoding EEG signals is that classifiers do not generalise well. This means that one cannot train classifiers on one subject and test on another. The eventual goal of this research is to experiment with the idea of subject independent feature decoding. 

The Folder named Notebooks has some examples of this implementation. 

Resources: 

Dataset : http://bnci-horizon-2020.eu/database/data-sets (the one by David Steryl) 

Description: https://lampx.tugraz.at/~bci/database/002-2014/description.pdf

CSP:https://ieeexplore.ieee.org/document/5662067 
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2918755/
    
DWT:https://link.springer.com/article/10.1007/s13246-015-0333-x

FBCSP: https://ieeexplore.ieee.org/document/4634130 (used the concept of MIBIF for this implementation)



