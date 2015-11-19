# CLAST
CUDA implemented large-scale alignment search tool

# HOW TO USE ?

## Build

### Dependenies.

CUDA (version 4.0~)  
"boost library"  
"thrust library (version ~1.7)" <- very important  
NVIDIA GPU (newer than Fermi architecture)

### Go to "clast" directory.

### Edit "Makefile".

You only need to edit CUDA_PATH and "GENCODE_FLAGS".

### Do "make".

### CLAST is now in your current directory.

# CAUTION

   Result may contain odd result due to GPU memory error.

# FORM OF RESULT FILE

   Result file is sepalated by tab.

0: queryLabel
1: query side start index
2: query side hit length
3: query strand ("+" or "-")
4: targetLabel
5: target side start index
6: target side hit length
7: identity ("match num" / "query side hit length" * 100 %)
8: score
9: E-value

# FOR LARGE REFERENCE SEQUENCES

If your database contains large (refer "-tRAM" and "-tVRAM" option) sequences,  
you need to preprocess your database by "preprocessDB" before execute CLAST.  
You can learn how to use it by exeute it without any option.


# LICENSE

GNU GPL

# VERSION

0.1.0 Feb.5,  2014: first version.  

0.1.1 Jan.19, 2015:  
    (1) more dense alignment is now available.  
    (2) florting point value is now available for tRAM, qRAM, tVRAM, and qVRAM in command line parameter

0.1.2 Apr.26, 2015:  
    Add "CDeviceHitList_alignmentHits.cuh.7.0" for CUDA 7.0.  
    Now CLAST can be built on CUDA 7.0 if it will be renamed "CDeviceHitList_alignmentHits.cuh".  
    But it does not work.  
    Please use CUDA 5.5 or CUDA 4.x, and Fermi or Kepler architecture GPU.

0.1.3 Nov.19, 2015:
    Remove "CDeviceHitList_alignmentHits.cuh.7.0".
    Remove "doc/READ_ME*".
    Remove "Makefile.*"
    Edit "README.md".
