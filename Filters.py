# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy as np

def help_message():
   print("Usage: [Question_Number] [Input_Options] [Output_Options]")
   print("[Question Number]")
   print("1 Histogram equalization")
   print("2 Frequency domain filtering")
   print("3 Laplacian pyramid blending")
   print("[Input_Options]")
   print("Path to the input images")
   print("[Output_Options]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "[path to input image] " + "[output directory]")                                # Single input, single output
   print(sys.argv[0] + " 2 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, three outputs
   print(sys.argv[0] + " 3 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, single output
   
# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================

def CDF_Equalize_hist(channel):
   hist,bins=np.histogram(channel,256,[0,256])
   cdf=hist.cumsum()
   cdf_normalized_transform = np.ma.masked_equal(cdf,0)        # Avoiding 0's so that they don;t result in errors, no issues since we are restoring in the end
   cdf_normalized_transform = (cdf_normalized_transform - cdf_normalized_transform.min())*255/(cdf_normalized_transform.max()-cdf_normalized_transform.min())
   cdf = np.ma.filled(cdf_normalized_transform,0).astype('uint8')
   img2=cdf[channel]
   return img2

def histogram_equalization(img_in):

   # Write histogram equalization here
   # Histogram equalization result 
   component_blue,component_green,component_red=cv2.split(img_in)
   hb=CDF_Equalize_hist(component_blue)
   hg=CDF_Equalize_hist(component_green)
   hr=CDF_Equalize_hist(component_red)   
   img_out=cv2.merge((hb,hg,hr))
   
   return True, img_out
   
def Question1():

   # Read in input images
   input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   
   # Histogram equalization
   succeed, output_image = histogram_equalization(input_image)
   
   # Write out the result
   output_name = sys.argv[3] + "output1.png"
   cv2.imwrite(output_name, output_image)

   return True
   
# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================

def low_pass_filter(img_in):
	
   # Write low pass filter here
   # Low pass filter result
   img_in=cv2.cvtColor(img_in,cv2.COLOR_RGB2GRAY)
   f=np.fft.fft2(img_in)
   fshift = np.fft.fftshift(f)
   #print img_in.shape
   rows, cols = img_in.shape
   crow,ccol = rows/2 , cols/2
   mask = np.zeros((rows,cols),np.uint8)
   mask[crow-10:crow+10, ccol-10:ccol+10] = 1
   fshift = fshift*mask
   f_ishift = np.fft.ifftshift(fshift)
   img_out = np.fft.ifft2(f_ishift)
   img_out = np.abs(img_out)
   
   return True, img_out

def high_pass_filter(img_in):

   # Write high pass filter here
   # High pass filter result
   img_in=cv2.cvtColor(img_in,cv2.COLOR_RGB2GRAY)
   f=np.fft.fft2(img_in)
   fshift = np.fft.fftshift(f)
   rows, cols = img_in.shape
   #print 'lpf'
   crow,ccol = rows/2 , cols/2
   mask = np.ones((rows,cols),np.uint8)
   mask[crow-10:crow+10, ccol-10:ccol+10] = 0
   fshift = fshift*mask
   f_ishift = np.fft.ifftshift(fshift)
   img_out = np.fft.ifft2(f_ishift)
   img_out = np.abs(img_out)
   
   return True, img_out
   
def deconvolution(img_in):
   
   # Write deconvolution codes here
   # Deconvolution result
   gk = cv2.getGaussianKernel(21,5)
   f1=np.fft.fft2(np.float32(img_in),(img_in.shape[0],img_in.shape[1]))
   fshift1 = np.fft.fftshift(f1)
   f2=np.fft.fft2(np.float32(gk),(img_in.shape[0],img_in.shape[1]))
   fshift2 = np.fft.fftshift(f2)

   fshift3=fshift1/fshift2
   f_ishift3 = np.fft.ifftshift(fshift3)
   img_out = np.fft.ifft2(f_ishift3)
   img_out = np.abs(img_out)
    
   return True, img_out*255

def Question2():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH);

   # Low and high pass filter
   succeed1, output_image1 = low_pass_filter(input_image1)
   succeed2, output_image2 = high_pass_filter(input_image1)
   
   # Deconvolution
   succeed3, output_image3 = deconvolution(input_image2)
   
   # Write out the result
   output_name1 = sys.argv[4] + "output2LPF.png"
   output_name2 = sys.argv[4] + "output2HPF.png"
   output_name3 = sys.argv[4] + "output2deconv.png"
   cv2.imwrite(output_name1, output_image1)
   cv2.imwrite(output_name2, output_image2)
   cv2.imwrite(output_name3, output_image3)
   
   return True
   
# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================

def laplacian_pyramid_blending(img_in1, img_in2):

   # Write laplacian pyramid blending codes here
   # Blending result

   img_in1=img_in1[:,:img_in1.shape[0]]
   original_apple=img_in1.copy()

   gauss_apple=[]
   gauss_apple.append(original_apple)

   for i in xrange(0,6):    #Gaussian pyramid, 6 levels
      down_layer=cv2.pyrDown(gauss_apple[i])
      gauss_apple.append(down_layer)

   laplacian_apple=[]
   laplacian_apple.append(gauss_apple[5])

   for i in xrange(4,-1,-1):   #getting the laplacian pyramid
      size=(gauss_apple[i].shape[1],gauss_apple[i].shape[0])
      enlarge=cv2.pyrUp(gauss_apple[i+1],dstsize=size)
      result=cv2.subtract(gauss_apple[i],enlarge)
      laplacian_apple.append( result)


   img_in2=img_in2[:img_in1.shape[0],:img_in1.shape[0]]
   original_orange=img_in2.copy()

   gauss_orange=[]
   gauss_orange.append(original_orange)

   for i in xrange(0,6):
      down_layer=cv2.pyrDown(gauss_orange[i])
      gauss_orange.append(down_layer)

   laplacian_orange=[gauss_orange[5]]
   for i in xrange(4,-1,-1):
      size=(gauss_orange[i].shape[1],gauss_orange[i].shape[0])
      enlarge=cv2.pyrUp(gauss_orange[i+1],dstsize=size)
      result=cv2.subtract(gauss_orange[i],enlarge)
      laplacian_orange.append(result)

   add=[]
   for lapple,lorange in zip(laplacian_apple,laplacian_orange):
    r,c,z=lapple.shape
    merge=np.hstack((lapple[:,0:c/2],lorange[:,c/2:]))
    add.append(merge)

   img_out=add[0]
   for i in xrange(1,6):
       size=(add[i].shape[1],add[i].shape[0])
       img_out=cv2.pyrUp(img_out,dstsize=size)
       img_out=cv2.add(img_out,add[i])


   return True, img_out

def Question3():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR);
   
   # Laplacian pyramid blending
   succeed, output_image = laplacian_pyramid_blending(input_image1, input_image2)
   
   # Write out the result
   output_name = sys.argv[4] + "output3.png"
   cv2.imwrite(output_name, output_image)
   
   return True

if __name__ == '__main__':
   question_number = -1
   
   # Validate the input arguments
   if (len(sys.argv) < 4):
      help_message()
      sys.exit()
   else:
      question_number = int(sys.argv[1])
	  
      if (question_number == 1 and not(len(sys.argv) == 4)):
         help_message()
	 sys.exit()
      if (question_number == 2 and not(len(sys.argv) == 5)):
	 help_message()
         sys.exit()
      if (question_number == 3 and not(len(sys.argv) == 5)):
	 help_message()
         sys.exit()
      if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
	 print("Input parameters out of bound ...")
         sys.exit()

   function_launch = {
   1 : Question1,
   2 : Question2,
   3 : Question3,
   }

   # Call the function
   function_launch[question_number]()
