import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
import cv2
from PIL import Image
import io
import base64

def line_eqa(a,b):
    slope = (b[1] - a[1]) / (b[0] - a[0])
    intercept = a[1] - slope * a[0]
    return slope,intercept

def distance(a,b):
    return ((b[0] - a[0])**2 + (b[1] - a[1])**2)**0.5
    



def shortest_distance_detector(bboxes,height):

    fig=plt.figure()
    ax=plt.axes()
    ax.set_facecolor(color='white')
    ay=fig.add_subplot(1,1,1)

    letters=[]
    numbers=[]
    letters_x_values=[]
    letters_y_values=[]
    
    numbers_x_values=[]
    numbers_y_values=[]

    for i in bboxes:
        try:
            a=int(i[-1])
            numbers.append(a)
            numbers_x_values.append(i[0])
            numbers_y_values.append(height-i[1])
        except Exception:
            letters.append(i[-1])
            letters_x_values.append(i[0])
            letters_y_values.append(height-i[1])
    ay.scatter(letters_x_values, letters_y_values,color='0',linewidths=2)
    ay.scatter(numbers_x_values, numbers_y_values, color='w')
    for i in range(len(letters_x_values)):
        ay.annotate(letters[i], xy=(letters_x_values[i], letters_y_values[i]), xytext=(letters_x_values[i]+10, letters_y_values[i]+10),fontsize=17)
    for i in range(len(numbers_x_values)):
        ay.annotate(numbers[i], xy=(numbers_x_values[i], numbers_y_values[i]), xytext=(numbers_x_values[i]-10, numbers_y_values[i]-10),fontsize=17)

    com=list(combinations(letters,2))
    center_points=[]
    for i in com:
        m,b=line_eqa([letters_x_values[letters.index(i[0])],letters_y_values[letters.index(i[0])]],[letters_x_values[letters.index(i[1])],letters_y_values[letters.index(i[1])]])
        x_center = (letters_x_values[letters.index(i[0])] + letters_x_values[letters.index(i[1])]) / 2
        y_center = m * x_center + b
        center_points.append([x_center,y_center])
    
    dis=[]
    final=[]
    plot=[]
    for i in range(len(numbers)):
        tem_dis=[]
        for j in center_points:
                tem_dis.append(distance(j,[numbers_x_values[i],numbers_y_values[i]]))
        final.append(com[np.argmin(tem_dis)][0]+com[np.argmin(tem_dis)][1]+"="+str(numbers[i]))
        plot.append(com[np.argmin(tem_dis)])
        del center_points[np.argmin(tem_dis)]
        del com[np.argmin(tem_dis)]


    for i in plot:
        x_values=[]
        y_values=[]
        m,b=line_eqa([letters_x_values[letters.index(i[0])],letters_y_values[letters.index(i[0])]],[letters_x_values[letters.index(i[1])],letters_y_values[letters.index(i[1])]])
        x_values = [letters_x_values[letters.index(i[0])], letters_x_values[letters.index(i[1])]]
        y_values = [letters_y_values[letters.index(i[0])], letters_y_values[letters.index(i[1])]]
        for i in range(letters_x_values[letters.index(i[0])]+1, letters_x_values[letters.index(i[1])]):
                                     x_values.append(i)
                                     y_values.append(m*i + b)
        ay.plot(x_values, y_values)
    ax.axis('off')
    for side in ['top','bottom','left','right']:
        ax.spines[side].set_visible(False)
    ay.axis('off')
    for side in ['top','bottom','left','right']:
        ay.spines[side].set_visible(False)
    fig.canvas.draw()
    img=np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
    img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    pil_im=Image.fromarray(img)
    buff=io.BytesIO()
    pil_im.save(buff,format='JPEG')
    
    with open("predg.jpeg", 'wb') as f:
        f.write(buff.getvalue())
        
    img_str=base64.b64encode(buff.getvalue()).decode('utf-8')
    
    return final,img_str
