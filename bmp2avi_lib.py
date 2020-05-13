# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys
import os
import shutil
import datetime
import time
#import Image

#縦に連結
def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    # list check
    if not im_list:
        return
    w_min = min(im.shape[1] for im in im_list)
    try:
        im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    except cv2.error as e:
        print( "cv2.error : vconcat_resize_min" )
        print(e.args)
        return
    
    return cv2.vconcat(im_list_resize)

    #im_v_resize = vconcat_resize_min([im1, im2, im1])
    #cv2.imwrite('data/dst/opencv_vconcat_resize.jpg', im_v_resize)
# 横に連結    
def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

    #im_h_resize = hconcat_resize_min([im1, im2, im1])
    #cv2.imwrite('data/dst/opencv_hconcat_resize.jpg', im_h_resize)

# x and y are the distance from the center of the background image
def cvpaste(img, imgback, x, y, angle=0.0, scale=1.0):  
    # x and y are the distance from the center of the background image 

    r = img.shape[0]
    c = img.shape[1]
    rb = imgback.shape[0]
    cb = imgback.shape[1]
    hrb=round(rb/2)
    hcb=round(cb/2)
    hr=round(r/2)
    hc=round(c/2)

    # Copy the forward image and move to the center of the background image
    imgrot = np.zeros((rb,cb,3),np.uint8)
    imgrot[hrb-hr:hrb+hr,hcb-hc:hcb+hc,:] = img[:hr*2,:hc*2,:]

    # Rotation and scaling
    M = cv2.getRotationMatrix2D((hcb,hrb),angle,scale)
    imgrot = cv2.warpAffine(imgrot,M,(cb,rb))
    # Translation
    M = np.float32([[1,0,x],[0,1,y]])
    imgrot = cv2.warpAffine(imgrot,M,(cb,rb))

    # Makeing mask
    imggray = cv2.cvtColor(imgrot,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(imggray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of the forward image in the background image
    img1_bg = cv2.bitwise_and(imgback,imgback,mask = mask_inv)

    # Take only region of the forward image.
    img2_fg = cv2.bitwise_and(imgrot,imgrot,mask = mask)

    # Paste the forward image on the background image
    imgpaste = cv2.add(img1_bg,img2_fg)

    return imgpaste
def shift_xy(image, shift_x, shift_y):
    h, w = image.shape[:2]
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src.copy()
    dest[:,0] += shift_x # シフトするピクセル値
    dest[:,1] += shift_y # シフトするピクセル値
    affine = cv2.getAffineTransform(src, dest)
    return cv2.warpAffine(image, affine, (w, h))
def rotate(image, angle):
    h, w = image.shape[:2]
    affine = cv2.getRotationMatrix2D((0,0), angle, 1.0)
    return cv2.warpAffine(image, affine, (w, h))
#
# # 比較明合成    
def Lighten(bg_img, fg_img):
    result = np.zeros(bg_img.shape)
    
    is_BG_lighter = bg_img > fg_img
    
    result[is_BG_lighter] = bg_img[is_BG_lighter]
    result[~is_BG_lighter] = fg_img[~is_BG_lighter]
    
    return result
# 切り抜き
def small_image( img, xcc, ycc, size = 24 ) : 
    # img[top : bottom, left : right]
    # サンプル1の切り出し、保存
    hs = int(size/2) ; top = ycc-hs ; bottom=ycc+hs ; left = xcc-hs ; right = xcc+hs
    height = img.shape[0] 
    width = img.shape[1]  
    #h, w = image.shape[:2]
    #plane = img.shape[2] ; print(width,height, plane)
    move_x = 0
    move_y = 0
    if top < 0 :
        move_y = top
        top = top - move_y
        bottom = bottom - move_y
    if bottom >= height :
        move_y = bottom -(height-1)
        top = top - move_y
        bottom = bottom - move_y
    if left < 0 :
        move_x = left
        left = left - move_x
        right = right - move_x
    if right >= width :
        move_x = right -(width-1)
        left = left - move_x
        right = right - move_x
    img1  = img[ top : bottom, left : right ]
    if  abs(move_x) < 0.001 and abs(move_y) < 0.001 :
        return img1 
    #img2 = np.zeros((size, size, 3), np.uint8)
    return shift_xy(img1, -move_x, -move_y)
 
   
def small_image_write( img, xcc, ycc, filename_img ) : 
    # img[top : bottom, left : right]
    # サンプル1の切り出し、保存
    img1 = small_image(img, xcc, ycc) 
    try:
        cv2.imwrite(filename_img, img1)
    except cv2.error as e:
        print( "cv2.error : small_image_write" )
        print(e.args)
        return

def small_image_write2( img2, img1, img0, xcc, ycc, filename_img ) : 
    # img[top : bottom, left : right]
    # サンプル1の切り出し、保存
    # RGB分離
    img_blue_c0, img_green_c0, img_red_c0 = cv2.split(img0) 
    img_blue_c1, img_green_c1, img_red_c1 = cv2.split(img1) 
    img_blue_c2, img_green_c2, img_red_c2 = cv2.split(img2)
    img = cv2.merge((img_blue_c1, img_green_c1, img_red_c2)) 
    try:
        cv2.imwrite(filename_img, img)
    except cv2.error as e:
        print( "cv2.error : small_image_write2" )
        print(e.args)
        return

def small_image_lighten( img_lighten, img, detect_frame, j, xc, yc, size ) :
            if detect_frame-4 == j :
                img0 = small_image(img, xc, yc, size)
                img_lighten = img0
            if detect_frame-2 == j :
                img0 = small_image(img, xc, yc, size)
                img_lighten = Lighten(img0, img_lighten )
            if detect_frame-0 == j :
                img0 = small_image(img, xc, yc, size)
                img_lighten = Lighten(img0, img_lighten )
            if detect_frame+2 == j :
                img0 = small_image(img, xc, yc, size)
                img_lighten = Lighten(img0, img_lighten )
            if detect_frame+4 == j :
                img0 = small_image(img, xc, yc, size)
                img_lighten = Lighten(img0, img_lighten )
            #return img_lighten

                #cv2.imwrite(filename_img, img_lighten)
                #small_image_write(img, xc[0], yc[0], filename_img)

def small_image_concat( img_lighten, img, detect_frame, j, xc, yc, size, filename_img ) :
            if detect_frame-4 == j :
                img0 = small_image(img, xc, yc, size)
                img_lighten = img0
            if detect_frame-2 == j :
                img0 = small_image(img, xc, yc, size)
                img_lighten = vconcat_resize_min([img_lighten, img0] )
            if detect_frame-0 == j :
                img0 = small_image(img, xc, yc, size)
                img_lighten = vconcat_resize_min([img_lighten, img0] )
            if detect_frame+2 == j :
                img0 = small_image(img, xc, yc, size)
                img_lighten = vconcat_resize_min([img_lighten, img0] )
            if detect_frame+4 == j :
                img0 = small_image(img, xc, yc, size)
                img_lighten = vconcat_resize_min([img_lighten, img0] )
                cv2.imwrite(filename_img, img_lighten)

def bmp2avi(path_dir, target_dir, remake=False, disp=True):
    if not os.path.isdir(path_dir):
        print( "Error 1"+path_dir )
        return
    if not path_dir.endswith("/"):
        path_dir += "/"

    if not os.path.isdir(target_dir):
        print( "Error 2")
        return
    if not target_dir.endswith("/"):
        target_dir += "/"

    files = os.listdir(path_dir)
    if len(files) == 0:
        print( "Error 3")
        return 
    if not os.path.isfile(path_dir+files[0]):
        print( "Error 4")
        return
    if not files[0].endswith(".BMP"):
        print( "Error 5")
        return

    #print( "Check:"+path_dir+"log.txt")
    xc=[]
    yc=[]
    detect_frame=0
    j=0
    lockon_num     = " 0"
    lockon_num_pre = " 0"
    if os.path.exists(path_dir+"log.txt" ) :
        f=open(path_dir+"log.txt","r")
        strlist = f.readlines()
        for line in strlist:
            j = j+1
            lockon_num_pre = lockon_num
            lockon_num = line.split( "](" )[1][20:22]
            if lockon_num==" 1" and lockon_num_pre == " 0" :
                xc.append(int(line.split( "](" )[1][0:3]))
                yc.append(int(line.split( "](" )[1][4:7]))
                if detect_frame == 0 : detect_frame = j
        
        for i in range(0,len(xc)):
            if disp : print( "detectFrame:"+str(detect_frame)+ " (xc,yc)["+str(i)+"]=("+str(xc[i])+","+str(yc[i])+")")
        
        f.close()

    # 2度目のavi化は行わない
    #print( "Check:"+path_dir+"avi_make_ended.txt")
    if os.path.exists(path_dir+"avi_make_ended.txt" ) and remake==False :
        if disp : print( "avi_make_ended.txt exist!!")
        return

    try:
        img1 = cv2.imread(path_dir+files[0])
        height , width , layers =  img1.shape
    except AttributeError as e:
        # BMP fileが壊れている場合
        print( "AttributeError" )
        print(e.args)
        return 

    filename     = target_dir+"/"+files[0][4:].split(".")[0]+'_00.avi'
    filename_log = target_dir+"/"+files[0][4:].split(".")[0]+'_00t.txt'
    filename_img = target_dir+"/"+files[0][4:].split(".")[0]+'_00s.png'
    if os.path.exists( filename ) :
        os.remove( filename )
        #return
    if os.path.exists( filename_log ) :
        os.remove( filename_log )
        #return
    if os.path.exists( filename_img ) :
        os.remove( filename_img )
        #return
   
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    #fourcc = cv2.VideoWriter_fourcc(*'H264')
    #fourcc = cv2.cv.CV_FOURCC('D', 'I', 'B', ' ')
    #fourcc = cv2.cv.CV_FOURCC('I', '4', '2', '0')
    #fourcc = cv2.cv.CV_FOURCC('D', 'I', 'V', 'X')
    video = cv2.VideoWriter(filename, fourcc,30,(width,height))
    #video = cv2.VideoWriter(filename, cv2.cv.CV_FOURCC('H', 'Y', 'M', 'T'),30,(width,height))
    #video = cv2.VideoWriter(filename, cv2.cv.CV_FOURCC('U', 'L', 'Y', '0'),30,(width,height))
    #video = cv2.VideoWriter(filename, cv2.cv.CV_FOURCC('L', 'A', 'G', 'S'),30,(width,height))
    #video = cv2.VideoWriter(filename, cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'),30,(width,height))

    if disp : cv2.namedWindow('BMP2AVI')
    ft=[]
    for fn in files:
        ft.append( fn[4:] )
    dic = dict(zip(ft,files))

    # key値でソート
    j=0
    size=24
    #img_lighten = np.zeros((size*5, size, 1))
    for k, fn in sorted(dic.items()):
        if disp : print( k, fn )

        if fn.endswith(".BMP") :
            try:
                img = cv2.imread(path_dir+fn)
                #small_image_lighten(img_lighten, img, detect_frame,j,xc[0],yc[0],size)
                #cv2.imwrite(filename_img, img_lighten)
                xcc=xc[0] ; ycc=yc[0]
            
                if detect_frame < 2 :
                    img0 = small_image(img, xcc, ycc, size) 
                    img_lighten = img0  
                if detect_frame-2 == j :
                    img_2 = small_image(img, xcc, ycc, size)
                    img0 = small_image(img, xcc, ycc, size)
                    img_lighten = img0
                if detect_frame-1 == j :
                    img_1 = small_image(img, xcc, ycc, size)
                    img0 = small_image(img, xcc, ycc, size)
                    img_lighten = vconcat_resize_min([img_lighten, img0] )
                if detect_frame-0 == j :
                    img_0 = small_image(img, xcc, ycc, size)
                    small_image_write2(img_2, img_1, img_0, xcc, ycc, filename_img)
                    #small_image_write(img,xcc,ycc,filename_img)
                    img0 = small_image(img, xcc, ycc, size)
                    img_lighten = vconcat_resize_min([img_lighten, img0] )
                if detect_frame+2 == j :
                    img0 = small_image(img, xcc, ycc, size)
                    img_lighten = vconcat_resize_min([img_lighten, img0] )
                if detect_frame+4 == j :
                    img0 = small_image(img, xcc, ycc, size)
                    img_lighten = vconcat_resize_min([img_lighten, img0] )
                    #cv2.imwrite(filename_img, img_lighten)
            except UnboundLocalError as e:
                print( "UnboundLocalError" )
                print(e.args)
                continue
            except IndexError as e:
                print( "IndexError" )
                print(e.args)
                return

            for i in range(0,len(xc)):
                cv2.circle(img,(xc[i],yc[i]),25-i,(255-5*i,255-5*i,0))
            
            if disp : cv2.imshow('BMP2AVI', img)
            video.write(img)
            if disp : cv2.waitKey(20)
        if fn == "log.txt":
            shutil.copyfile(path_dir+fn, filename_log)
        j=j+1

    cv2.destroyAllWindows()
    video.release()

    print( "Output:",filename )
    
    #avi作成済み目印
    f=open(path_dir+"avi_make_ended.txt","w")
    f.close()
    #print( path_dir+"avi_make_ended.txt" )

#def fname(main_dir) :
#    files = os.listdir(main_dir)
#
import zipfile
def fish_dir_org(main_dir):
    fdir=[]
    fjpg=[]
    flog=[]
    zf = zipfile.ZipFile('sample.zip', 'w', zipfile.ZIP_DEFLATED)
    files = os.listdir(main_dir)
    for f in files :
        print( len(f), f )
        if len(f) == 5 :
            fdir.append(main_dir + f )
        elif f.endswith(".JPG") :
            fjpg.append( main_dir + f )
        else :
            flog.append( f )
            zf.write( main_dir + f )
    # ログファイル作成        
    zf.close()
    
    
    for fd in fdir :
        bmp2avi( fd )
    
#    for f in fjpg :
    
    print( zf )
#    print( fjpg )
#    print( flog )


# もし　small imageの名前をans listに合わせて変更、ans=-1, 0 , 1, 2 
def rename_small_image( fnfull, flg_list=[-1], target_dir='C:/Users/root/Documents/Python Scripts/gym/tmp' ): #path_dir, detect_frame, xc, yc):
    for id in  range(len(flg_list)):
        st = "_{:02d}".format(id)
        #fn='J:/MT/20200311/20200311_025809_540_00.avi'
        #path_dir =  os.path.dirname( fnfull )
        basefn   =  os.path.basename( fnfull ).replace('_00.avi', st)
        if  flg_list[id] == -1 :
            filename_img  = target_dir+"/"+basefn+'s.png'
        elif flg_list[id] == 0 :
            filename_img = target_dir+"/"+basefn+'s0.png'
        elif flg_list[id] == 1 :
            filename_img = target_dir+"/"+basefn+'s1.png'
        elif flg_list[id] == 2 :
            filename_img = target_dir+"/"+basefn+'s2.png'
        print(filename_img)
        if os.path.isfile( filename_img ):
            return
            
        fn0 = get_fn_small_image(fnfull, id, target_dir)
        if fn0 == '' :
            return
        if fn0 == filename_img :
            return
        os.rename( fn0, filename_img)
 
# もし　small imageがあれば、ファイル名、なければ ''
def get_fn_small_image( fnfull, id=0, target_dir='C:/Users/root/Documents/Python Scripts/gym/tmp' ): #path_dir, detect_frame, xc, yc):
    st = "_{:02d}".format(id)
    #fn='J:/MT/20200311/20200311_025809_540_00.avi'
    #path_dir =  os.path.dirname( fnfull )
    basefn   =  os.path.basename( fnfull ).replace('_00.avi', st)
    filename_img  = target_dir+"/"+basefn+'s.png'
    filename_img0 = target_dir+"/"+basefn+'s0.png'
    filename_img1 = target_dir+"/"+basefn+'s1.png'
    filename_img2 = target_dir+"/"+basefn+'s2.png'

    if os.path.exists( filename_img ) :        
       return filename_img
    if os.path.exists( filename_img0 ) :        
        return filename_img0
    if os.path.exists( filename_img1 ) :        
        return filename_img1
    if os.path.exists( filename_img2 ) :        
        return filename_img2
    return ''

# もし　small imageがあれば、True なければ False
def is_small_image( fnfull, target_dir='C:/Users/root/Documents/Python Scripts/gym/tmp' ): #path_dir, detect_frame, xc, yc):
    #fn='J:/MT/20200311/20200311_025809_540_00.avi'
    #path_dir =  os.path.dirname( fnfull )
    basefn   =  os.path.basename( fnfull )
    filename_img  = target_dir+"/"+basefn.split(".")[0]+'s.png'
    filename_img0 = target_dir+"/"+basefn.split(".")[0]+'s0.png'
    filename_img1 = target_dir+"/"+basefn.split(".")[0]+'s1.png'
    filename_img2 = target_dir+"/"+basefn.split(".")[0]+'s2.png'
    #print(filename_img0)

    if os.path.exists( filename_img ) :        
        #os.remove( filename_img )
        return True
    if os.path.exists( filename_img0 ) :        
        return True
    if os.path.exists( filename_img1 ) :        
        return True
    if os.path.exists( filename_img2 ) :        
        return True
    return False
 
# もし　id番目のsmall imageがあれば、True なければ False
def is_small_image_id( fnfull, id=0, target_dir='C:/Users/root/Documents/Python Scripts/gym/tmp' ): #path_dir, detect_frame, xc, yc):
    st = "_{:02d}".format(id)
    #fn='J:/MT/20200311/20200311_025809_540_00.avi'
    #path_dir =  os.path.dirname( fnfull )
    basefn   =  os.path.basename( fnfull ).replace('_00.avi', st)
    filename_img  = target_dir+"/"+basefn+'s.png'
    filename_img0 = target_dir+"/"+basefn+'s0.png'
    filename_img1 = target_dir+"/"+basefn+'s1.png'
    filename_img2 = target_dir+"/"+basefn+'s2.png'
    #print(filename_img0)

    if os.path.exists( filename_img ) :        
        #os.remove( filename_img )
        return True
    if os.path.exists( filename_img0 ) :        
        return True
    if os.path.exists( filename_img1 ) :        
        return True
    if os.path.exists( filename_img2 ) :        
        return True
    return False
 
def make_small_image( fnfull, target_dir='C:/Users/root/Documents/Python Scripts/gym/tmp' ): #path_dir, detect_frame, xc, yc):
    #fn='J:/MT/20200311/20200311_025809_540_00.avi'
    path_dir =  find_fish_dir( fnfull )
    detect_frame, xc, yc =  get_detect_loc( fnfull ) #5/11 list化
    #target_dir = os.path.dirname(fnfull)
    #print(fnfull, path_dir, target_dir, detect_frame, xc, yc)

    if not os.path.isdir(path_dir):
        print( "Error 1"+path_dir )
        return
    if not path_dir.endswith("/"):
        path_dir += "/"

    if not os.path.isdir(target_dir):
        print( "Error 2")
        return
    if not target_dir.endswith("/"):
        target_dir += "/"

    files = os.listdir(path_dir)
    if len(files) == 0:
        print( "Error 3")
        return 
    if not os.path.isfile(path_dir+files[0]):
        print( "Error 4")
        return
    if not files[0].endswith(".BMP"):
        print( "Error 5")
        return

    for i in range(len(xc)):
        if is_small_image_id(fnfull, i) :
            continue
        st = "_{:02d}".format(i)
        basefn   =  os.path.basename( fnfull ).replace('_00.avi', st)
        filename_img  = target_dir+"/"+basefn+'s.png'
        
        ft=[]
        for fn in files:
            ft.append( fn[4:] )
        dic = dict(zip(ft,files))

        # key値でソート
        j=0
        size=24
        if detect_frame[i] < 3:
            detect_frame[i] = 3
        for k, fn in sorted(dic.items()):
            #print( k, fn )

            if fn.endswith(".BMP") :
                try:
                    img = cv2.imread(path_dir+fn)
                    #xcc=xc[0] ; ycc=yc[0]
                    xcc=xc[i] ; ycc=yc[i]
                
                    if detect_frame[i] < 2 :
                        img0 = small_image(img, xcc, ycc, size) 
                        img_lighten = img0  
                    if detect_frame[i]-2 == j :
                        img_2 = small_image(img, xcc, ycc, size)
                        img0 = small_image(img, xcc, ycc, size)
                        img_lighten = img0
                    if detect_frame[i]-1 == j :
                        img_1 = small_image(img, xcc, ycc, size)
                        img0 = small_image(img, xcc, ycc, size)
                        img_lighten = vconcat_resize_min([img_lighten, img0] )
                    if detect_frame[i]-0 == j :
                        img_0 = small_image(img, xcc, ycc, size)
                        small_image_write2(img_2, img_1, img_0, xcc, ycc, filename_img)
                        break
                except UnboundLocalError as e:
                    print( "UnboundLocalError. (make_small_image)" )
                    print(e.args)
                    continue
                except IndexError as e:
                    print( "IndexError. (make_small_image)" )
                    print(e.args)
                    return
            j=j+1
        print('make NN data: '+filename_img)

def fish_dir(main_dir, t_dir, remake=False, disp=True):
    print( "fish_dir() "+main_dir , t_dir )
    if not os.path.isdir(main_dir):
        print( "Error 1 fish_dir()" )
        return
    if not main_dir.endswith("/"):
        main_dir += "/"

    if not os.path.isdir(t_dir):
        print( "Error 2 fish_dir()" )
        return
    if not t_dir.endswith("/"):
        t_dir += "/"

    fdir=[]
    fjpg=[]
    flog=[]
#    zf = zipfile.ZipFile('sample.zip', 'w', zipfile.ZIP_DEFLATED)
    files = os.listdir(main_dir)
    for f in files :
#        print( len(f), f )
        if len(f) == 5 :
            fdir.append(main_dir + f )
        elif f.endswith(".JPG") :
            fjpg.append( main_dir + f )
        else :
            flog.append( f )
#            zf.write( main_dir + f )
    # ログファイル作成        
#    zf.close()
    
    
    for fd in fdir :
        if disp : print( fd,t_dir )
        bmp2avi( fd, t_dir, remake, disp )

#検出座標取り出し
def get_detect_loc(cvvfn):
    xc=[]
    yc=[]
    detect_frame=[]
    j=0
    lockon_num     = " 0"
    lockon_num_pre = " 0"
    path_dir = os.path.dirname( cvvfn ) #self.cvv.base_dir
    basename = os.path.basename( cvvfn )
    log_file_name = path_dir+'/'+basename.replace(".avi","t.txt")
    #print(log_file_name, basename)
    if os.path.exists( log_file_name ) :
        with open( log_file_name, "r") as f:
            strlist = f.readlines()
            for line in strlist:
                j = j+1
                lockon_num_pre = lockon_num
                lockon_num = line.split( "](" )[1][20:22]
                if lockon_num==" 1" and lockon_num_pre == " 0" :
                    xc.append(int(line.split( "](" )[1][0:3]))
                    yc.append(int(line.split( "](" )[1][4:7]))
                    detect_frame.append(j)            
            for i in range(0,len(xc)):
                #print( "idx:"+str("detectFrame:"+str(detect_frame)+ " (xc,yc)["+str(i)+"]=("+str(xc[i])+","+str(yc[i])+")")
                #print(i,detect_frame[i], xc[i], yc[i])
                i
        #f.close()

    if  xc :
        return  detect_frame, xc, yc
    return [],[],[]
    #    return  detect_frame[0], xc[0], yc[0]
    #return 0,0,0

def base_dir_pre_f( base_dir):
    bbase_dir = os.path.dirname(base_dir)
    dd = base_dir[-8:]
    dt = datetime.date(int(dd[:4]),int(dd[4:6]),int(dd[6:]))
    # 翌日を求める timedelta
    td = datetime.timedelta(days=1)
    dt_pre = dt-td
    dd1 = dt_pre.strftime("%Y%m%d") 
    #print(dt,dt_pre,dd1)
    return bbase_dir+'/'+dd1

def find_fish_dir(fn):
    base_dir = os.path.dirname(fn)
    base_dir_pre = base_dir_pre_f(base_dir)
    base_fn  = os.path.basename(fn)
    fish_dir = base_dir + '/Fish1'
    fish_dir_list = []
    for f in  os.listdir( fish_dir ):
        if os.path.isdir(fish_dir+'/'+ f):
            fish_dir_list.append(fish_dir+'/'+ f)
    fish_dir = base_dir_pre + '/Fish1'
    for f in  os.listdir( fish_dir ):
        if os.path.isdir(fish_dir+'/'+ f):
            fish_dir_list.append(fish_dir+'/'+ f)
    dt = base_fn[:19]
    for d in fish_dir_list:
        for f in os.listdir(d):
            #print(d,f,dt)
            if f.find(dt) != -1:
                #print(d)
                return d
    return ''
    #print(base_dir, base_dir_pre, base_fn, fish_dir_list)

if __name__ == 'xx__main__':
           
    main_dir = './fishdata/fish1/'
    main_dir = 'J:/MT/20200315/fish1/'
    t_dir = './tmp/'
    remake = True
    disp = False
    fish_dir(main_dir, t_dir, remake, disp)

    #指定する画像フォルダ
    path_dir = './fishdata/fish1/26377/' #'C:/temp/tmp/14368/'
    target_dir ='./tmp/'
    #bmp2avi(path_dir, target_dir, True)

#---------------------------------------------------
# main
# 日付指定   
if __name__ == "__main__":
    fn='J:/MT/20200311/20200311_025809_540_00.avi'
    fn='J:/MT/20200508/20200508_191440_616_00.avi'
    fn='J:/MT/20190107/20190107_013333_230_00.avi'

    is_small_image_id(fn, 2)
    flg = get_fn_small_image(fn)
    rename_small_image(fn,2)

    fish_dir =  find_fish_dir(fn)
    detect_frame, xc, yc =  get_detect_loc(fn)
    make_small_image( fn )
    
    print( fish_dir, detect_frame, xc, yc, flg)

    sys.exit()

    dtnow = datetime.datetime.now()
    drange=1 #実行日数（戻り日数）
    if len( sys.argv )   >= 5:
        yyyy=int(sys.argv[1])
        mm  =int(sys.argv[2])
        dd  =int(sys.argv[3])
        drange =int(sys.argv[4])
    elif len( sys.argv ) == 4:
        yyyy=int(sys.argv[1])
        mm  =int(sys.argv[2])
        dd  =int(sys.argv[3])
    elif len( sys.argv ) == 3:
        yyyy=dtnow.year
        mm  =int(sys.argv[1])
        dd  =int(sys.argv[2])
    elif len( sys.argv ) == 2:
        yyyy=dtnow.year
        mm  =dtnow.month
        dd  =int(sys.argv[1])
    elif len( sys.argv ) == 1:
        yyyy=dtnow.year
        mm  =dtnow.month
        dd  =dtnow.day
        drange =7
    
    if yyyy < 2000 or yyyy > dtnow.year :
        print( "Year '%s' 範囲外" % yyyy)
        sys.exit()

    if mm < 1 or mm > 12 :
        print( "Month '%s' 範囲外" % mm)
        sys.exit()
    
    if dd < 1 or dd > 31 :
        print( "Day '%s' 範囲外" % dd)
        sys.exit()

    if drange < 1 or drange > 365 :
        print( "Drange '%s' 範囲外" % drange)
        sys.exit()
    
    for i in range(drange):
        dt = datetime.date(yyyy,mm,dd) -datetime.timedelta(days=i)
        print( dt)
        time.sleep(1)

        dir = dt.strftime("/%Y%m%d")
        TargetDrive   = "J:"
        BaseSoucePath = TargetDrive + "/MT"
        TargetPath1   = BaseSoucePath + dir
        TargetPath2   = BaseSoucePath + dir + "/Fish1"

        main_dir = TargetPath2   #'J:/MT/20200315/fish1/'
        t_dir = './tmp/'
        remake = True
        disp = False
        fish_dir(main_dir, t_dir, remake, disp)

