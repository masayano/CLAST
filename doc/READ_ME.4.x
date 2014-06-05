                       READ ME for CUDA 4.x

######################## < HOW TO BUILD > #########################

0. Install CUDA toolkit and CUDA SDK
   (CLAST was developed on CUDA ver4.0~4.2).

 * "boost library" install protocol:

   Visual C++: Installer for Visual C++ 7.1~10.0 can be available in "boostpro".
   Cygwin gcc: You can install boost when install Cygwin by its installer.
               Select the boost and boost-dev of "Devel" category.
   Unix like : Almost all distribution may have boost package.
               "$yum install boost boost-dev" or
               "apt-get install libboost*-dev" or
               "cd /usr/ports/devel/boost; make" etc.

   If you need more detail, see "http://www.boost.org/".

1. Put "clast/clast" directory into "~/NVIDIA_GPU_Computing_SDK/C/src".

2. Go to "~/NVIDIA_GPU_Computing_SDK/C/src/clast".

3. Do "mv Makefile.4.x Makefile".

4. Do "sudo make" or "make".

5. CLAST is now in "~/NVIDIA_GPU_Computing_SDK/C/bin/linux(or other)/release".

ex: CLAST needs NVIDIA Fermi/Kepler architecture GPU.

########################## < CAUTION > ############################

   Result may contain odd result due to GPU memory error.

#################### < FORM OF RESULT FILE > ######################

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

################ < FOR LARGE REFERENCE SEQUENCES > ################

   If your database contains large (refer "-tRAM" and "-tVRAM" option) sequences,
   you need to preprocess your database by "preprocessDB" before execute CLAST.
   You can learn how to use it by exeute it without any option.


########################### < LICENSE > ###########################

GNU GPL
