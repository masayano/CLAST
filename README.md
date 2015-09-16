# CLAST
CUDA implemented large-scale alignment search tool

# HOW TO USE ?
Please read files in "doc".

# VERSION

0.1.0 Feb.5,  2014: first version.  

0.1.1 Jan.19, 2015:  
    (1) more dense alignment is now available.  
    (2) florting point value is now available for tRAM, qRAM, tVRAM, and qVRAM in command line parameter

0.1.2 Apr.26, 2015:  
    Add "CDeviceHitList_alignmentHits.cuh.7.0" for CUDA 7.0.  
    Now CLAST can be built on CUDA 7.0 if it will be renamed "CDeviceHitList_alignmentHits.cuh".
    But it does not work. Please use CUDA 5.5 or CUDA 4.0.
