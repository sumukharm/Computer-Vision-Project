import cv2
import numpy


def add_pixels(image,row,col,scalar):
    for i in xrange(0,row):
        for j in xrange(0,col):
            image[i,j]=image[i,j]+scalar           #Since it is a numpy array, BGR can be modified in one shot i.e array([158,161,165])+2 = array([160,163,167])
            
def subtract_pixels(img,row,col,scalar):
    for i in xrange(0,row):
        for j in xrange(0,col):
            img[i,j]=img[i,j]-scalar

def multiply_pixels(img,row,col,scalar):
    for i in xrange(0,row):
        for j in xrange(0,col):
            img[i,j]=img[i,j]*scalar

def divide_pixels(img,row,col,scalar):
    for i in xrange(0,row):
        for j in xrange(0,col):
            img[i,j]=img[i,j]/scalar

def display(img,s):
    cv2.imshow(s,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    #Display
    Image_name=raw_input("Please enter the image file name to be read and displayed : ")
    img=cv2.imread(Image_name, 1)
    row=len(img)
    col=len(img[0])
    display(img,'Actual_image_Display')

    #Add_Scalar
    img_add=cv2.imread(Image_name,1)        #This is another read which will be used as addition
    scalar=input("Enter the value you want to add to each pixel : ")
    add_pixels(img_add,row,col,scalar)
    display(img_add,'Addition_operation')
    cv2.imwrite('Addition_result.jpg',img_add)

    #Subtract_scalar
    img_sub=cv2.imread(Image_name,1)
    scalar=input("Enter the value you want to Subtract from each pixel : ")
    subtract_pixels(img_sub,row,col,scalar)
    display(img_sub,'Subtraction_operation')
    cv2.imwrite('Subtraction_result.jpg',img_sub)

    #Multiply_Scalar
    img_mul=cv2.imread(Image_name,1)
    scalar=input("Enter the value you want to multiply to each pixel : ")
    multiply_pixels(img_mul,row,col,scalar)
    display(img_mul,'Multiply_operation')
    cv2.imwrite('Multiplication_result.jpg',img_mul)

    #Divide_Scalar
    img_div=cv2.imread(Image_name,1)
    scalar=input("Enter the value you want to divide each pixel by : ")
    divide_pixels(img_div,row,col,scalar)
    display(img_div,'Division_operation')
    cv2.imwrite('Division_result.jpg',img_div)

    #Resize by 1/2s
    print "here you go : resizing by 1/2"
    resize_img=cv2.resize(img,(img.shape[0]/2,img.shape[1]/2),interpolation = cv2.INTER_AREA)
    display(resize_img,'Resized Image')
    cv2.imwrite('Resize_result.jpg',resize_img)


if "__name__==main":
    main()
