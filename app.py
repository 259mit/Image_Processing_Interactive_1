import streamlit as st
from pyngrok import ngrok
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import re
from numpy.lib.stride_tricks import as_strided
import skimage.measure

def main():
    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('Welcome','Image Analysis','Image Processing', 'Transformation', 'Basics of CNN')
    )
    
    if selected_box == 'Welcome':
        welcome() 
    if selected_box == 'Image Analysis':
        imageanalysis()
    if selected_box == 'Image Processing':
        imageprocessing()
    #if selected_box == 'Tools':
     #   perspective()
    if selected_box == 'Transformation':
        transformations()
    if selected_box == 'Basics of CNN':
        conv_pool()

# Functions:
def read_image(img):
  a = cv2.imread(img)
  a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
  return a

def image_shape(img):
  a = img.shape
  return a

#Slider for roatation, scale, 
# Box for x and y
def rotate_image(img, degrees=0, scale=1, x=0, y=0):
  '''
  degrees, scale, move by x, move by y
  '''
  (rows, cols) = img.shape[:2]
  a = cv2.getRotationMatrix2D((cols / 2, rows / 2), degrees, scale)
  b = np.float32([[1, 0, x], [0, 1, y]])
  a = cv2.warpAffine(img, a, (cols, rows))
  a = cv2.warpAffine(a, b, (cols, rows))
  cv2.imwrite('rotated.jpg', a)
  return a

#Box for x and y
def show_edges(img, x_kernel, y_kernel):
  edges = cv2.Canny(img, x_kernel, y_kernel)
  cv2.imwrite('edge.jpg', edges)
  return edges

#Slider for kernel till 20
def Gaussian_blur(img, kernel, sigmaX=0):
  blur = cv2.GaussianBlur(img, (kernel, kernel),sigmaX)
  #cv2.imwrite('blurjpg', blur)
  return blur

#Change colour space
def tohsv(img):
  hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  return hsv_img

#Image threshold
# button for inv and slider for thresh till 256
def thresh(img, thresha, inv=False):
  if inv == False:
    ret,t_img = cv2.threshold(img, thresha, 255, cv2.THRESH_BINARY)
  if inv == True:
    ret,t_img = cv2.threshold(img, thresha, 255, cv2.THRESH_BINARY_INV)
  t_img = t_img.astype(np.float64)
  return t_img

#Image resize
# Input box for x and y
def resize(img, x, y):
  res = cv2.resize(img, (x, y), interpolation = cv2.INTER_NEAREST)
  return res

#Histogram equalization
def histe(img):
  equ = cv2.equalizeHist(img)
  inte = img.ravel()
  bins = 256
  range = [0,256]
  #plt.hist(img.ravel(),256,[0,256])
  return equ, inte, bins, range

#Shuffle points
def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect

#Four point transform
def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	return warped

#Get contours and transform perspective
#Value for kernelx and kernely
def perspective_transform(img, kernelx = 10, kernely = 10):
  edged = cv2.Canny(img, kernelx, kernely)
  thresh = cv2.adaptiveThreshold(edged,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
  contours=cv2.findContours(thresh.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  contours = imutils.grab_contours(contours)
  contours = sorted(contours,key=cv2.contourArea, reverse = True)[:10]
  screenCnt = None
  for c in contours:
      peri = cv2.arcLength(c, True)
      approx = cv2.approxPolyDP(c, 0.018 * peri, True)
      if len(approx) == 4:
          screenCnt = approx
          break
  screenCnt1 = screenCnt.reshape(4,2)
  imgt = four_point_transform(img, screenCnt1)
  return imgt

def morphol(img, r1, r2, g1, g2, b1, b2, k):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  lower = np.array([r1, g1, b1])
  upper = np.array([r2, g2, b2])
  mask = cv2.inRange(hsv, lower, upper)
  res = cv2.bitwise_and(img, img, mask = mask)
  kernel = np.ones((k, k), np.uint8)
  opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
  closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
  gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
  return opening, closing, gradient
def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size, 
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)


def convolve(array, kernel):
    ks = kernel.shape[0] # shape gives the dimensions of an array, as a tuple
    final_length = array.shape[0] - ks + 1
    return np.array([(array[i:i+ks]*kernel).sum() for i in range(final_length)])


def convolve2Dimage(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output


def welcome():
  st.title('An Interactive Image Processing Tutorial')
  st.subheader('Mithesh Ramachandran')
  st.markdown('Interactive tutorial to understand basics of parameter tuning in image processing')
  st.subheader('Light theme recommended!')


im1 = read_image('Blue_Origin.jpg')
im2 = read_image('numplateip.jpg')

def imageanalysis():
  st.header('Image Analysis - Histogram Equalization')
  im1 = read_image('Blue_Origin.jpg')
  st.image(im1, use_column_width=True)
  st.text('Grayscale: ')
  im1 = cv2.imread('Blue_Origin.jpg', 0)
  st.image(im1, use_column_width=True)
  st.text('Image credits: The Verge')
  st.write('Image shape is: ', im1.shape)
  st.markdown('Histogram Equalization')
  st.text('Histogram equalization is a technique for adjusting image intensities to enhance contrast.')
  st.text('Let f be a given image represented as a mr by mc matrix of integer pixel intensities ranging from 0 to L − 1. L is the number of possible intensity values, often 256.')
  st.text('Let p denote the normalized histogram of f with a bin for each possible intensity.')
  st.latex(r'''
  p_n = \frac{number  of  pixels  with  intensity  'n'}{total  number  of  pixels} \     n = 0, 1, ..., L − 1.
  '''
  )
  st.text('The histogram equalized image g will be defined by: ')
  st.latex(r'''
  T(k) = floor((L - 1 )\sum_{n=0}^{f_i,j} p_n)
  '''
  )
  st.text('where floor() rounds down to the nearest integer. This is equivalent to transforming the pixel intensities, k, of f by the function')
  st.latex(r'''
  g_i,j = floor((L - 1 )\sum_{n=0}^{k} p_n)
  '''
  )
  st.markdown('Image Histogram is: ')
  equ, inte, bins, range = histe(im1)
  histr = cv2.calcHist([im1],[0],None,[256],[0,256])
  st.bar_chart(histr)
  st.text("View histogram equalized image")
  if st.button('Histogram Equalization'):
      st.image(equ,use_column_width=True)
      st.text('Equalized Histogram is: ')
      histr2 = cv2.calcHist([equ],[0],None,[256],[0,256])
      st.bar_chart(histr2)
      st.text('This method usually increases the global contrast of images when its usable data is represented by close contrast values.')
      st.text(' This allows for areas of lower local contrast to gain a higher contrast.')

def imageprocessing():
  st.header('Image Processing')
  #st.text('Image Threshold')
  #im1 = cv2.imread('/content/Blue_Origin.jpg', 0)
  #threshv = st.slider('Change Threshold value',min_value = 0,max_value = 255)
  #st.image(thresh(im1, thresha=threshv), use_column_width=True)
  st.markdown('Gaussian Blur')
  kernel = st.slider('Change Blur Gaussian Kernel value',min_value = 1,max_value = 21, step=2)
  st.image(Gaussian_blur(im1, kernel))
  st.markdown('Edge Detection')
  x_kernel = st.slider('Change X Kernel value',min_value = 0,max_value = 300)
  y_kernel = st.slider('Change Y Kernel value',min_value = 0,max_value = 300)
  st.image(show_edges(im1, x_kernel, y_kernel))
  st.text('The Canny edge detector is an edge detection operator that uses a multi-stage algorithm to detect a wide range of edges in images.')
  st.text('The process of Canny edge detection algorithm can be broken down to five different steps: ')
  st.text('1. Apply Gaussian filter to smooth the image in order to remove the noise')
  st.text('2. Find the intensity gradients of the image')
  st.text('3. Apply gradient magnitude thresholding or lower bound cut-off suppression to get rid of spurious response to edge detection')
  st.text('4. Apply double threshold to determine potential edges')
  st.text('5. Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.')
  st.markdown('Morphological Transformation')
  red = st.slider('Select a range of Red',0.0, 255.0, (125.0, 135.0)) 
  green = st.slider('Select a range of Green',0.0, 255.0, (50.0, 255.0)) 
  blue = st.slider('Select a range of Blue',0.0, 100.0, (50.0, 255.0))
  k = st.slider('Select a kernel for Morphological transformation',min_value = 1,max_value = 21, step=2)
  r1, r2 = red
  g1, g2 = green
  b1, b2 = blue 
  opening, closing, gradient = morphol(im1, r1, r2, g1, g2, b1, b2, k)
  if st.button('Opening'):
      st.image(opening)
      st.text('Opening Morphological Transformation')
  if st.button('Closing'):
      st.image(closing)
      st.text('Closing Morphological Transformation')
  if st.button('Gradient'):
      st.image(gradient)
      st.text('Gradient Morphological Transformation')

#def perspective():
#  st.header('Perspective correction')
#  st.image(im2, use_column_width=True)
#  st.text('Perspective transformation using Keystone correction')
#  st.image(perspective_transform(im2))
#  st.text('Try for your own image: ')
#  uploaded_file = st.file_uploader('upload your image', type=['png', 'jpeg'])
#  if uploaded_file is not None:
#    image = read_image(uploaded_file)
#    st.image(perspective_transform(image), caption='Uploaded Image.', use_column_width=True)

def transformations():
  st.image(im1)
  st.text('Resize')
  st.write('Current image shape is: ', im1.shape[:2])
  X = st.text_input("Enter Width", "1200")
  Y = st.text_input("Enter Height", "800")
  st.image(resize(im1, int(X), int(Y)))
  st.text('Rotation and Translation: ')
  degrees = st.slider('Change Angle of rotation',min_value = 0,max_value = 360)
  scale = st.text_input("Enter scale", "1")
  x = st.text_input("Move horizontally by", "10")
  y = st.text_input("Move vertically by", "10")
  st.image(rotate_image(im1, degrees=degrees, scale=int(scale), x=int(x), y=int(y)))

def conv_pool():
  st.title('Convolution and Pooling')
  st.markdown('In Convolutional Neural Networks, we usually deal with images.')
  st.markdown('An image is basically an array of pixel values. Depending of Grayscale or')
  st.markdown('colour, it might also have channels (eg. R,G,B)')
  st.markdown('In this tutorial we will first explore basic concepts required')
  st.markdown('for building CNNs, first with an array, then images and then a set of images.')
  st.header('Pooling')
  st.markdown('Pooling is basically reducing the size of an array of size nxn. The process involves a Kernel or a filter, which')
  st.markdown('is of dimension mxm, m<n. A subset of mxm is taken from the array, starting from array[0,0],')
  st.markdown('Then the maximum or average value is selected from this subset. Then the subset is moved in the array, along both array[i,j],')
  st.markdown('such that all the elements from the array are represented in a subset.')
  st.markdown('The max or average of each subset represents an element from a new array.')
  st.markdown('')
  st.image('pool1.png')
  st.text('Image credits: Arden Dertat')
  st.subheader('Here\'s an exercise to better understand pooling')
  k = st.slider('Change size of the array',min_value = 4,max_value = 10, step = 2)
  array1 = np.random.randint(1, 10, (k,k))
  st.dataframe(pd.DataFrame(array1))
  array2 = np.flip(array1.T, axis=1)
  st.markdown('Lets see the 2x2 kernel in action for this array')
  st.markdown('We\'re using a 2x2 kernel as it one of the most commonly used kernels. You can however use am mxm kernel.')
  a = np.zeros((k,k))
  p = 2
  lista = []
  ic = 0
  pl = st.empty()
  for i in range(0, k, p):
    for j in range(0, k, p):
      for m in range(1, p):
        a[i,j] = 1
        a[i+m,j] = 1
        a[i,j+m] = 1
        a[i+m,j+m] = 1
        plt.matshow(a)
        plt.hlines(y=np.arange(0, k)+0.5, xmin=np.full(k, 0)-0.5, xmax=np.full(k, k)-0.5, color="black")
        plt.vlines(x=np.arange(0, k)+0.5, ymin=np.full(k, 0)-0.5, ymax=np.full(k, k)-0.5, color="black")
        plt.axis("off")
        #plt.figure(figsize=(5,5))
        st.set_option('deprecation.showPyplotGlobalUse', False)
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        ic = ic + 1
        fig.savefig('test'+str(ic)+'.jpg', dpi=100)
        #pl.pyplot()
        #plt.imshow(a, interpolation='nearest')
        #plt.show()
        pl.image('test'+str(ic)+'.jpg')
        a = np.zeros((k,k))
        time.sleep(1)
  poolar = skimage.measure.block_reduce(array1, (2,2), np.max)
  st.markdown('The resulting array after applying MAX pooling would look like this!')
  st.dataframe(pd.DataFrame(poolar))
  st.markdown('---')
  st.subheader('But what if you want to capture more information?')
  st.subheader('Stride: ')
  st.markdown('In the above example you saw that the kernel moved 2 steps in both x and y direction.')
  st.markdown('This also meant that in any of the kernel operations, no element appeared twice.')
  st.markdown('But sometimes if you do need to capture more information, you might reduce the step to 1')
  st.markdown('This causes a comparative increase in dimension size to moving two steps, but the resultant array')
  st.markdown('is still smaller than the input array. This step size is called stride. The default value is usually the kernel size.')
  st.subheader('Padding: ')
  st.markdown('Padding is like adding a border to the array. This allows the kernel to cover more area, hence')
  st.markdown('capturing more features')
  st.subheader('Let\'s take a look at another example: ')
  k = st.slider('Change size of the array',min_value = 4,max_value = 8, step = 2)
  v = st.slider('Change size of the Padding',min_value = 0,max_value = 2)
  array4 = np.random.randint(1, 10, (k,k))
  array4 = np.pad(array4, pad_width=v, mode='constant', constant_values=0)
  st.dataframe(pd.DataFrame(array4))
  array5 = np.flip(array4.T, axis=1)
  st.markdown('Lets see the 2x2 kernel in action for this array')
  k2=k+2*v
  c = np.zeros((k2,k2))
  p = 2
  listb = []
  ic = 0
  pl = st.empty()
  for i in range(0, k2-1, p-1):
    for j in range(0, k2-1, p-1):
      for m in range(1, p):
        c[i,j] = 1
        c[i+m,j] = 1
        c[i,j+m] = 1
        c[i+m,j+m] = 1
        plt.matshow(c)
        plt.hlines(y=np.arange(0, k2)+0.5, xmin=np.full(k2, 0)-0.5, xmax=np.full(k2, k2)-0.5, color="black")
        plt.vlines(x=np.arange(0, k2)+0.5, ymin=np.full(k2, 0)-0.5, ymax=np.full(k2, k2)-0.5, color="black")
        plt.axis("off")
        #plt.figure(figsize=(5,5))
        st.set_option('deprecation.showPyplotGlobalUse', False)
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        ic = ic + 1
        fig.savefig('test2'+str(ic)+'.jpg', dpi=100)
        #pl.pyplot()
        #plt.imshow(a, interpolation='nearest')
        #plt.show()
        pl.image('test2'+str(ic)+'.jpg')
        c = np.zeros((k2,k2))
        time.sleep(0.5)
  poolar2 = pool2d(array4, kernel_size=2, stride=1, padding=0, pool_mode='max')
  st.markdown('The resulting array after applying MAX pooling WITH STRIDE 1/ PADDING would look like this!')
  st.dataframe(pd.DataFrame(poolar2))
  st.markdown('Noticed a pattern in the output array size? Adding a stride increased the dimension by 1.')
  st.markdown('Now let\'s see the mathematical formula behind this!')
  st.latex(r'''
  Output_, size = \frac{n - k}{stride + 1}
  '''
  )
  st.markdown('---')
  st.header('Convolutions')
  st.markdown('You would have come across Convolution in mathematics and signal processing, especially in Fourier and Laplace transforms')
  st.markdown('Convolution is basically multiplying a larger array with a smaller array (Kernel), to produce an array that better represents')
  st.markdown('or captures some of the features of the original array.')
  st.markdown('Convolution is similar to what we did above, just that instead of calculating max value of each sub-array (after fitting the kernel) of the array,')
  st.markdown('we will multiply that selected sub-array with another array (same dimesions of the kernel')
  st.markdown('To avoid confusion, let\'s just refer this as a filter here.')
  st.image('conv1.png')
  st.markdown('')
  st.markdown('Since we know how kernels/ filters/ padding and stride work, let\'s just jump straight to an example.')
  q = st.slider('Change size of the array for Convolution',min_value = 4,max_value = 8, step = 2)
  rq = st.slider('Change size of the padding for Convolution',min_value = 0,max_value = 2)
  array6 = np.random.randint(1, 10, (q,q))
  edgekernel=np.array([[-1, -1, -1], [-1, 4, -1],[-1, -1, -1]])
  horkernel = np.array([[-1, -1, -1], [2, 2, 2],[-1, -1, -1]])
  verkernel = np.array([[-1, 2, -1], [-1, 2, -1],[-1, 2, -1]])
  edgekernels=np.array([[str(-1), str(-1), str(-1)], [str(-1), 4, str(-1)],[str(-1), str(-1), str(-1)]])
  horkernels = np.array([[str(-1), str(-1), str(-1)], [2, 2, 2],[str(-1),str(-1), str(-1)]])
  verkernels = np.array([[str(-1), 2, str(-1)], [str(-1), 2, str(-1)],[str(-1), 2, str(-1)]])
  if st.button('Vertical Filter'):
      st.dataframe(pd.DataFrame(array6))
      st.text('Vertical Filter')
      st.dataframe(pd.DataFrame(verkernels))
      st.text('Output: ')
      st.dataframe(pd.DataFrame(convolve2Dimage(array6, verkernel, padding=rq, strides=1)))
  if st.button('Horizontal Filter'):
      st.dataframe(pd.DataFrame(array6))
      st.text('Horizontal Filter')
      st.dataframe(pd.DataFrame(horkernels))
      st.text('Output: ')
      st.dataframe(pd.DataFrame(convolve2Dimage(array6, horkernel, padding=rq, strides=1)))
  if st.button('Edge Filter'):
      st.dataframe(pd.DataFrame(array6))
      st.text('Edge Filter')
      st.dataframe(pd.DataFrame(edgekernels))
      st.text('Output: ')
      st.dataframe(pd.DataFrame(convolve2Dimage(array6, edgekernel, padding=rq, strides=1)))
  if st.button('Try your own Filter'):
      st.dataframe(pd.DataFrame(array6))
      st.text('Your filter')
      #st.dataframe(pd.DataFrame(edgekernels))
      st.text('In progress, stay tuned for updates every week')
      #st.dataframe(pd.DataFrame(convolve2Dimage(array6, edgekernel, padding=rq, strides=1)))
  st.markdown('---')
  st.subheader('Now let\'s try for images')
  st.markdown('Let\'s start with grayscale images:')
  im1 = cv2.imread('Blue_Origin.jpg', 0)
  st.image(im1, use_column_width=True)
  if st.button('Vertical Filter Image'):
      st.image(convolve2Dimage(im1, verkernel, padding=0, strides=1), clamp = True)
  if st.button('Horizontal Filter Image'):
      st.image(convolve2Dimage(im1, horkernel, padding=0, strides=1), clamp = True)
  if st.button('Edge Filter Image'):
      st.image(convolve2Dimage(im1, edgekernel, padding=0, strides=1), clamp = True)
  st.markdown('---')
  st.subheader('Let\'s talk deep learning')
  st.markdown('When training a deep neural network on many images, a single filter might not be enough.')
  st.markdown('We need multiple filters or a set of filters usually 32 or 64, to help extract maximum features from an image.')
  st.markdown('When training a DNN, we usually have colour (RGB) images that have 3 channels.')
  st.markdown('When dealing with channels in an image, The filters also have channels, Therefore resultant array from 1 filter is a 1 channel array (h x w x 1)')
  st.image('andrewngconv.png')
  st.text('Image: Andrew Ng')
  st.markdown('For Convolutions: ')
  st.image('conv2.png')
  st.markdown('This is how convolution works with RGB images. Notice that the image height and width do not change, but the depth layers (Channels) are replaced by number of filters')
  st.markdown('So an image array of size h x w x 3, with 32 conv filters applied will become an array of size h x w x 32')
  st.markdown('---')
  st.markdown('Similarly for pooling: ')
  st.image('pool2.png')
  st.markdown('As you can see pooling does not change the depth, but changes the height and width of an image array.')
  st.markdown('Convolution and Pooling in Neural Networks...')
  st.markdown('In deep learning, the pooling layers are always preceeded by a convolution layer')
  st.markdown('If you keep adding Convolutional and pooling layers, you might end up with an array of depth p, height and width as 3x3, 2x2 or 1x1')
  st.markdown('This is then flattened by adding a fully connected layer, where you get a 1xj array, where j = hxwxp.')
  st.markdown('---')
  st.text('Coming up: Other convolutions, Interactive kernels and Python codes for this exercise and CNNs')


if __name__ == "__main__":
    main()
