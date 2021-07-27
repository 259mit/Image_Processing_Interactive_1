import streamlit as st
from pyngrok import ngrok
import cv2
import imutils
import numpy as np

def main():
    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('Welcome','Image Analysis','Image Processing', 'Transformation')
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



def welcome():
  st.title('An Interactive Image Processing Tutorial')
  st.subheader('Mithesh Ramachandran')
  st.markdown('Interactive tutorial to understand basics of parameter tuning in image processing')


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


if __name__ == "__main__":
    main()