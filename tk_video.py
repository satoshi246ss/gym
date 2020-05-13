# -*- coding: utf-8 -*-
import sys
import os
import shutil
import datetime
import glob
import tkinter as tk
import tkinter.filedialog
import tkinter.ttk
from PIL import Image,ImageTk
import cv2
import numpy as np
import bmp2avi_lib
###import meteor1_eval

def rename_00002_avi(path):
    #path = os.path.dirname(fnfull)
    file_list = sorted([p for p in glob.glob(path+'/**') if os.path.isfile(p)])
    for f in file_list:
        basefn = os.path.basename(f)
        st = basefn.split('_')
        if st[0] == '00002' and len(st)==4 :
            dest = path +'/'+ st[1]+'_'+ st[2]+'_'+st[3].replace('.avi','_02.avi')
            if not os.path.isfile(dest):
                os.rename(f, dest)
                print('rename:'+f +'->'+dest)         
        if st[3] == '002.avi' and len(st)==4 :
            dest = path +'/'+ basefn.replace('_002.avi','_02.avi')
            if not os.path.isfile(dest):
                os.rename(f, dest)
                print('rename:'+f +'->'+dest)         

# fn:image file name
def get_datetime(fn):
    # fn: '20200503_013619_109_00.avi'
    # fn: '00002_20200423_200806_965.avi' 特殊構成
    # dt1 = datetime.datetime(year=2017, month=10, day=10, hour=15)
    try:
        st2 = fn.split('_')
        yy = st2[0][:4]
        mo = st2[0][4:6]
        dd = st2[0][-2:]
        hh = st2[1][:2]
        mm = st2[1][2:4]
        ss = st2[1][-2:]
        msec = st2[2]
        dt1 = datetime.datetime(int(yy),int(mo),int(dd),int(hh),int(mm),int(ss),int(msec+'000'))
    except IndexError as e:
        # BMP fileが壊れている場合
        print( "IndexError(get_datetime):"+fn )
        print(e.args)
        return 
    except ValueError as e:
        # BMP fileが壊れている場合
        print( "ValueError(get_datetime):"+fn+'  yy=',str(yy) )
        print(e.args)
        return 
    return dt1
    
def get_all_obs_files(fullfn):
    #fn = os.path.basename(fullfn)
    path = os.path.dirname(fullfn)
    #file_list = sorted(glob.glob(path+'/*'))
    file_list = sorted([p for p in glob.glob(path+'/**') if os.path.isfile(p)])
    return file_list

def get_same_obs_files(fullfn):
    if len(fullfn) == 0 :
        print('error fn empty. (get_same_obs_files): '+fullfn)
        return ''
        
    time_error = 10 #sec 　許容時間誤差
    fn = os.path.basename(fullfn)
    dt0 = get_datetime(fn)
 
    #path = os.path.dirname(fullfn)
    #file_list = sorted([p for p in glob.glob(path+'/**') if os.path.isfile(p)])
    obs_files=[]
    for f in get_all_obs_files(fullfn):
        fn = os.path.basename(f)
        dt1 = get_datetime(fn)
        if dt1 >= dt0 :
            td = dt1 -dt0
            if td.seconds <=  time_error :
                obs_files.append(f)
            else:
                break
        else :
            td = dt0 -dt1
            if td.seconds <=  time_error :
                obs_files.append(f)
        #print(f,dt0,dt1,td.seconds)
    return obs_files

class GUI:
    def __init__(self, targetpath):
        self.obs_path = targetpath
        #self.flg1 = tk.StringVar()  #グループA用変数
        #self.flg1.set('1')
        #fn="./tmp/20200322_224811_367_00.avi"
        nnTargetDir='./tmp' #"./tmp/data_20200501/train/" # for NN
        path ='./tmp/'  # for NN
        files = os.listdir(path)
        files_file = [f for f in files if os.path.isfile(os.path.join(path, f))]
        avi_files = []
        for f in files_file:
            if f.endswith(".avi"):
                avi_files.append(path+f) 
        #print(avi_files)

        self.cvv=CV2video(avi_files, nnTargetDir)
        self.root=tk.Tk()
        self.root.bind("<Key>",  self.get_key_input)
        self.ROOT_X = 1200
        self.ROOT_Y = 1080
        self.CANVAS_offsetX=10
        self.CANVAS_offsetY=100
        self.CANVAS_X=640
        self.CANVAS_Y=480
        size = 24 ; m = 12
        self.CANVAS2_X=size * m  #pixcel * 倍率
        self.CANVAS2_Y=size * m
        self.CANVAS3_X=size * m  #pixcel * 倍率
        self.CANVAS3_Y=size * m

        self.root.title(u"python tkinker opencv")
        self.root.geometry(str(self.ROOT_X) + "x" + str(self.ROOT_Y))
        self.root.resizable(width=0, height=0)

        self.next_file_flg = False
        self.cvv.count_num=0

        self.select_file()
        print('GUI init cvv.fn: '+self.cvv.fn )
        bmp2avi_lib.make_small_image(self.cvv.fn)
        self.cvv.imgfn = bmp2avi_lib.get_fn_small_image(self.cvv.fn)
        self.cvv.openFile()

        self.firstFrame()
        self.check_flg1_state()
        self.rename_imgdata()
        self.get_detect_loc()
        self.change_radio_button_bgcolor()
        self.label_idx_all.configure(text='/ '+str(len(self.cvv.flist)))
        ###self.get_nn_eval()

        self.afterMSec()

    # select file
    def select_file(self):
        rename_00002_avi(self.obs_path)
        self.obs_files = self.get_obs_file()
        #print('選択ファイル名：'+str(self.obs_files))
        avi_files = []
        for f in get_all_obs_files(self.obsfn):
            if f.endswith("00.avi"):
                avi_files.append(f) 
        #print(avi_files)
        self.cvv.set_filelist(avi_files)        
        self.cvv.set_base_dir( os.path.dirname(self.obsfn) )
        self.cvv.set_del_dir(self.obsfn[:2]+'/tmp')
        self.cvv.set_idx(0)

    def move_del_files(self, flist):
        for f in flist :
            print('move:'+str(f)+'->'+str(self.cvv.del_dir))
            shutil.move(f, self.cvv.del_dir)
            if f.endswith("00.avi"):
                self.cvv.remove_filelist(f)

    def get_flg1(self):
        return self.flg1

    def change_radio_button_bgcolor(self):    
        color1 = 'RosyBrown1'
        color2 = 'gray90'
        if self.flg1.get()=='0':
            self.rb1['bg'] = color1
            self.rb2['bg'] = color2
            self.rb3['bg'] = color2
        if self.flg1.get()=='1':
            self.rb1['bg'] = color2
            self.rb2['bg'] = color1
            self.rb3['bg'] = color2
        if self.flg1.get()=='2':
            self.rb1['bg'] = color2
            self.rb2['bg'] = color2
            self.rb3['bg'] = color1

        #ラジオボタン作成
    def create_radio_button(self) :
        #グループA用変数
        self.flg1 = tk.StringVar()
        self.flg1.set('0')

        #ラジオボタンを押した時の反応用関数
        def rb1_clicked():
            #print(self.flg1.get())
            self.change_radio_button_bgcolor()  
        
        step = 150 ; offset = 70
        #ラジオ1（グループA）
        self.rb1 = tk.Radiobutton(self.root,text='Meteor(0)',value=0,variable=self.flg1,
        command=rb1_clicked, font=("",18))
        #rb1.grid(row=1,column=1)
        self.rb1.place(x=step+offset, y=8) 
        #ラジオ2（グループA）
        self.rb2 = tk.Radiobutton(self.root,text='Plane(1)',value=1,variable=self.flg1,command=rb1_clicked, font=("",18))
        #rb2.grid(row=1,column=2)
        self.rb2.place(x=2*step+offset, y=8) 
    
        #ラジオ3（グループA）
        self.rb3 = tk.Radiobutton(self.root,text='Other(2)',value=2,variable=self.flg1,command=rb1_clicked, font=("",18))
        #rb3.grid(row=1,column=3)
        self.rb3.place(x=3*step+offset, y=8) 

        #ラジオ4（グループA）
        #rb4 = tk.Radiobutton(self.root,text='Cloud(3)',value=3,variable=self.flg1,command=rb1_clicked)
        #rb4.grid(row=1,column=4)
        #rb4.place(x=4*step, y=8) 

        #ラジオ5（グループA）
        #rb5 = tk.Radiobutton(self.root,text='Ghost(4)',value=4,variable=self.flg1,command=rb1_clicked)
        #rb5.grid(row=1,column=5)
        #rb5.place(x=5*step, y=8) 
        return self.flg1

    def change_radio_button2_bgcolor_clear(self):    
        color1 = 'gray90'
        color2 = 'gray60'
        self.rba0['bg'] = color1
        self.rba1['bg'] = color1
        self.rba3['bg'] = color1
        self.rba4['bg'] = color1
        self.rba8['bg'] = color1
        self.rba9['bg'] = color1
        self.rba10['bg'] = color1
        self.rba11['bg'] = color1
        self.rba13['bg'] = color1
        self.rba15['bg'] = color1
        self.rba16['bg'] = color1

    def update_radio_button2(self):
        for f in self.obs_files:
            if f[-6:-4]  =='00':
                self.change_radio_button2_bgcolor(0)
            elif f[-6:-4]=='01':
                self.change_radio_button2_bgcolor(1)
            elif f[-6:-4]=='_3':
                self.change_radio_button2_bgcolor(3)
            elif f[-6:-4]=='_4':
                self.change_radio_button2_bgcolor(4)
            elif f[-6:-4]=='_8':
                self.change_radio_button2_bgcolor(8)
            elif f[-6:-4]=='_9':
                self.change_radio_button2_bgcolor(9)
            elif f[-6:-4]=='10':
                self.change_radio_button2_bgcolor(10)
            elif f[-6:-4]=='11':
                self.change_radio_button2_bgcolor(11)
            elif f[-6:-4]=='13':
                self.change_radio_button2_bgcolor(13)
            elif f[-6:-4]=='15':
                self.change_radio_button2_bgcolor(15)
            elif f[-6:-4]=='16':
                self.change_radio_button2_bgcolor(16)

    def change_radio_button2_bgcolor(self, cam_id):    
        color1 = 'gray90'
        color2 = 'gray60'
        if cam_id == 0 :
            self.rba0['bg'] = color2
        elif cam_id == 1 :
            self.rba1['bg'] = color2
        elif cam_id == 3 :
            self.rba3['bg'] = color2
        elif cam_id == 4 :
            self.rba4['bg'] = color2
        elif cam_id == 8 :
            self.rba8['bg'] = color2
        elif cam_id == 9 :
            self.rba9['bg'] = color2
        elif cam_id == 10 :
            self.rba10['bg'] = color2
        elif cam_id == 11 :
            self.rba11['bg'] = color2
        elif cam_id == 13 :
            self.rba13['bg'] = color2
        elif cam_id == 15 :
            self.rba15['bg'] = color2
        elif cam_id == 16 :
            self.rba16['bg'] = color2

    #ラジオボタン作成
    def create_radio_button_cam(self) :
        #グループA用変数
        self.flg_cam = tk.IntVar()
        self.flg_cam.set(4)

        #ラジオボタンを押した時の反応用関数
        def rb_cam_clicked():
            print(self.flg_cam.get())
            #self.change_radio_button_bgcolor()  
        
        self.rbframe = tk.Frame(self.root, bd=2, relief="ridge")
        self.rbframe.place(x=1010, y=10)

        step = 150 ; offset = 70
        #ラジオ1（グループA）
        self.rba0 = tk.Radiobutton(self.rbframe,text='Fisheye(00)',value=0,variable=self.flg_cam, command=rb_cam_clicked, font=("",14))
        self.rba0.pack(anchor=tk.W)
        self.rba1 = tk.Radiobutton(self.rbframe,text='Fisheye2(01)',value=1,variable=self.flg_cam, command=rb_cam_clicked, font=("",14))
        self.rba1.pack(anchor=tk.W)
        #ラジオ2（グループA）
        self.rba4 = tk.Radiobutton(self.rbframe,text='Wide(4)',value=4,variable=self.flg_cam, command=rb_cam_clicked, font=("",14))
        self.rba4.pack(anchor=tk.W)
        #ラジオ3（グループA）
        self.rba8 = tk.Radiobutton(self.rbframe,text='Fine(8)',value=8,variable=self.flg_cam, command=rb_cam_clicked, font=("",14))
        self.rba8.pack(anchor=tk.W)
        self.rba9 = tk.Radiobutton(self.rbframe,text='SFine(9)',value=9,variable=self.flg_cam, command=rb_cam_clicked, font=("",14))
        self.rba9.pack(anchor=tk.W)
        self.rba13 = tk.Radiobutton(self.rbframe,text='Color(13)',value=13,variable=self.flg_cam, command=rb_cam_clicked, font=("",14))
        self.rba13.pack(anchor=tk.W)

        self.rba3  = tk.Radiobutton(self.rbframe,text='LrSpcam(3)',value=3,variable=self.flg_cam, command=rb_cam_clicked, font=("",14))
        self.rba3.pack(anchor=tk.W)
        self.rba15 = tk.Radiobutton(self.rbframe,text='NUV(15)',value=15,variable=self.flg_cam, command=rb_cam_clicked, font=("",14))
        self.rba15.pack(anchor=tk.W)
        self.rba16 = tk.Radiobutton(self.rbframe,text='NIR(16)',value=16,variable=self.flg_cam, command=rb_cam_clicked, font=("",14))
        self.rba16.pack(anchor=tk.W)
        self.rba10 = tk.Radiobutton(self.rbframe,text='Echelle Guide(10)',value=10,variable=self.flg_cam, command=rb_cam_clicked, font=("",14))
        self.rba10.pack(anchor=tk.W)
        self.rba11 = tk.Radiobutton(self.rbframe,text='Echelle(11)',value=11,variable=self.flg_cam, command=rb_cam_clicked, font=("",14))
        self.rba11.pack(anchor=tk.W)

        return self.flg_cam

    # キーの表示
    def get_key_input(self, event):
        key = event.keysym
        print(key)
        if(key=="F1"):
            self.flg1.set('0')
        if(key=="F2"):
            self.flg1.set('1')
        if(key=="F3"):
            self.flg1.set('2')
        self.change_radio_button_bgcolor() 

    def get_fn_target_img(self, fn, target_dir):
        basename = os.path.basename( fn )
        inlist=[]

        st = basename[-4:] #".avi"
        target_fn = target_dir +'/'+ basename
        inlist.append( target_fn.replace(st,"s.png") )
        inlist.append( target_fn.replace(st,"s0.png"))
        inlist.append( target_fn.replace(st,"s1.png"))
        inlist.append( target_fn.replace(st,"s2.png"))

        for path in inlist:
            if os.path.isfile(path):
                return path
        return bmp2avi_lib.make_small_image(fn)
        #return "./tmp/20200501s.png" #dummy

    def copy_nn_target_dir(self, fn, target_dir):
        basename = os.path.basename( fn )
        inlist=[]
        outlist=[]
        st = basename[-6:]

        target_fn = target_dir +"meteor/"+ basename
        inlist.append( target_fn.replace(st,"s.png") )
        inlist.append( target_fn.replace(st,"s0.png"))
        inlist.append( target_fn.replace(st,"s1.png"))
        inlist.append( target_fn.replace(st,"s2.png"))
        target_fn = target_dir +"plane/"+ basename
        inlist.append( target_fn.replace(st,"s.png") )
        inlist.append( target_fn.replace(st,"s0.png"))
        inlist.append( target_fn.replace(st,"s1.png"))
        inlist.append( target_fn.replace(st,"s2.png"))
        target_fn = target_dir +"other/"+ basename
        inlist.append( target_fn.replace(st,"s.png") )
        inlist.append( target_fn.replace(st,"s0.png"))
        inlist.append( target_fn.replace(st,"s1.png"))
        inlist.append( target_fn.replace(st,"s2.png"))

        for path in inlist:
            if os.path.isfile(path):
                outlist.append(path)
        for path in outlist:
            if os.path.isfile(path):
                os.remove(path)

        #print("outlist:",outlist)
        #print("inlist:",inlist)

        target_fn=""
        if basename[-5]=='0':
            target_fn = target_dir +"0_meteor/"+ basename
        if basename[-5]=='1':
            target_fn = target_dir +"1_plane/"+ basename
        if basename[-5]=='2':
            target_fn = target_dir +"2_other/"+ basename
        if len(target_fn)==0:
            return
        shutil.copyfile(fn, target_fn)
        print(fn,"->",target_fn)

    # 結果画像ファイル名にラベルを追加
    # 入力 "./tmp/20200323_193428_601_00.avi"
    # 例 20200323_193428_601_00s.png
    # 例 20200323_193428_601_00s0.png   s0:meteor  s1:plane  s2:other
    # 既に、ラベルがある場合は、書き換え
    # 結果画像ファイルがない場合は、メッセージ表示        
    def rename_imgdata(self):
        if not bmp2avi_lib.is_small_image(self.cvv.fn):
            print("結果画像ファイルがありません")
            return

        if self.flg1.get()=='0':
            ans=0
        if self.flg1.get()=='1':
            ans=1
        if self.flg1.get()=='2':
            ans=2
        bmp2avi_lib.rename_small_image(self.cvv.fn, self.cvv.flg1_list)
        return

        outfilename_non   = self.cvv.fn.replace(".avi","s.png")
        outfilename_meteor= self.cvv.fn.replace(".avi","s0.png")
        outfilename_plane = self.cvv.fn.replace(".avi","s1.png")
        outfilename_other = self.cvv.fn.replace(".avi","s2.png")

        outlist=[]
        path = outfilename_non
        if os.path.isfile(path):
            outlist.append(path)
        path = outfilename_meteor
        if os.path.isfile(path):
            outlist.append(path)
        path = outfilename_plane
        if os.path.isfile(path):
            outlist.append(path)
        path = outfilename_other
        if os.path.isfile(path):
            outlist.append(path)


        if len(outlist)==1:
            src = outlist[0]
            dest = outfilename_other
            if self.flg1.get()=='0':
                dest = outfilename_meteor
            if self.flg1.get()=='1':
                dest = outfilename_plane
            os.rename(src,dest)
            self.copy_nn_target_dir(dest, self.cvv.target_dir)
            return
        print("Error:結果画像ファイルが複数あります")
        #print(self.cvv.fn, basename, outlist)

    #検出座標取り出し
    def get_detect_loc(self):
        xc=[]
        yc=[]
        detect_frame=[]
        flg1=[]
        j=0
        lockon_num     = " 0"
        lockon_num_pre = " 0"
        path_dir = self.cvv.base_dir
        basename = os.path.basename( self.cvv.fn )
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
                        flg1.append(-1)
                for i in range(0,len(xc)):
                    print( "idx:"+str(self.cvv.idx) + " detectFrame:"+str(detect_frame[i])+ " (xc,yc)["+str(i)+"]=("+str(xc[i])+","+str(yc[i])+")")
            
            #f.close()
        if xc :
            self.cvv.xc_list = xc
            self.cvv.yc_list = yc
            self.cvv.xc = xc[0]
            self.cvv.yc = yc[0]
            self.cvv.detect_frame_list = detect_frame
            self.cvv.flg1_list = flg1
            self.cvv.xc_list_id = 0

    def get_nn_eval(self):
        nn_ans = meteor1_eval.nn_eval( self.net, self.cvv.imgfn )
        self.label_nn.configure(text=str(nn_ans))
        print(nn_ans)

    def update_label(self):
        #print(self.obsfn)
        if len(self.obsfn) < 4:
            return
        st = self.obsfn.split('/')
        self.label_date.configure(text=st[-2])
        st2 = st[-1].split('_')[1]
        self.label_time.configure(text=st2)

    def get_obs_file_form(self): #form type
        fTyp = [("","avi")]
        iDir = 'J:/MT/' #os.path.abspath(os.path.dirname(__file__))
        self.obsfn = tk.filedialog.askopenfilename(filetypes = fTyp,initialdir = iDir)
        return get_same_obs_files(self.obsfn)

    def get_obs_file(self): #cmd line type
        self.obsfn = ''
        flist = os.listdir(self.obs_path)
        for f in flist:
            #print(f[-7:])
            if f[-7:] == '_00.avi' :
                self.obsfn = self.obs_path+'/'+f
                break
        print(self.obsfn)
        return get_same_obs_files(self.obsfn)
 
    # 前回選択結果を反映
    def check_flg1_state(self):
        fn = bmp2avi_lib.get_fn_small_image(self.cvv.fn, self.cvv.xc_list_id)
        if fn == '':
            return
        elif fn[-5] == '0' or fn[-5] == '1' or fn[-5] == '2':
            self.flg1.set(fn[-5])
        return

        small_img_dir='C:/Users/root/Documents/Python Scripts/gym/tmp'
        basefn = os.path.basename(self.cvv.fn)
        outfilename_non   = small_img_dir+'/'+basefn.replace(".avi","s.png")
        outfilename_meteor= small_img_dir+'/'+basefn.replace(".avi","s0.png")
        outfilename_plane = small_img_dir+'/'+basefn.replace(".avi","s1.png")
        outfilename_other = small_img_dir+'/'+basefn.replace(".avi","s2.png")

        path = outfilename_non
        if os.path.isfile(path):
            #self.flg1.set('1')
            return
        path = outfilename_meteor
        if os.path.isfile(path):
            self.flg1.set('0')
            self.change_radio_button_bgcolor()
            return
        path = outfilename_plane
        if os.path.isfile(path):
            self.flg1.set('1')
            self.change_radio_button_bgcolor()
            return
        path = outfilename_other
        if os.path.isfile(path):
            self.flg1.set('2')
            self.change_radio_button_bgcolor()
            return

    # ボタン作成
    def create_button(self) :    
        #ボタンを押した時の反応用関数
        def button1_clicked():
            print('cb01')
            self.cvv.cap.release()
            self.cvv.cap2.release()
            self.cvv.flg1_list[self.cvv.xc_list_id] = int( self.flg1.get() )
            bmp2avi_lib.make_small_image(self.cvv.fn)
            self.rename_imgdata()
            # Meteor以外は、move file
            if not self.flg1.get() == '0' :
                self.move_del_files(self.obs_files)

            value = self.EditBox.get()
            self.cvv.set_idx( int(value) )
            self.label_idx_all.configure(text='/ '+str(len(self.cvv.flist)))
             
            print('cb02')
            #fn="./tmp/20200323_004207_221_00.avi"
            #self.cvv.set_fn(fn)
            self.change_radio_button2_bgcolor_clear()
            self.cvv.update_idx()
            self.obs_files = get_same_obs_files(self.cvv.fn)
            self.update_radio_button2()
            #print(self.cvv.fn, self.obs_files)
            self.cvv.openCamera()
            self.cvv.openCamera2(self.obs_files, '_4.avi')
            print('cb021')
            #self.cvv.imgfn = self.get_fn_target_img(self.cvv.fn, self.cvv.base_dir)
            if not bmp2avi_lib.is_small_image(self.cvv.fn) :
                print('cb022', self.cvv.fn)
                bmp2avi_lib.make_small_image(self.cvv.fn)
            print('cb023')
            self.get_detect_loc()
            self.cvv.imgfn = bmp2avi_lib.get_fn_small_image(self.cvv.fn, self.cvv.xc_list_id)
            #print(self.cvv.imgfn)
            self.cvv.openFile()
            self.button2.configure(text='Next Detect: '+str(self.cvv.detect_frame_list[self.cvv.xc_list_id])+' ' +str(self.cvv.xc_list_id +1)+'/'+str(len(self.cvv.xc_list)))
            self.check_flg1_state()
            self.change_radio_button_bgcolor()     
            #self.next_file_flg = True
            #print(self.flg1.get())
            print('cb03')
            
            self.EditBox.delete(0, tk.END)
            self.EditBox.insert(tk.END,self.cvv.idx+1)
            ###self.get_nn_eval()
        
        # 参照ボタン配置  
        button1 = tk.Button(self.root, text=u' OK ',width=10, command=button1_clicked, font=("",20))  
        #button1.grid(row=0, column=1)  
        button1.place(x=2, y=5) 

    def afterMSec(self):
        self.cvv.count_num+=1
        self.time_ms = self.cvv.count_num * self.time_ms_step
        self.label_count.configure(text=str(self.cvv.idx)+":"+  "{:3d}".format(self.cvv.frame_ID))
        self.pb.configure(value=self.cvv.frame_ID)
        if self.cvv.frame_ID == self.cvv.detect_frame_list[self.cvv.xc_list_id]:
            self.button2.configure(bg='red')
        else:
            self.button2.configure(bg='gray90')

        #cam0
        if self.time_ms >= self.cvv.frame_ID * self.cam0_fr_interval :
            # canvas
            try: 
                self.cvv.cameraFrame()

                image_rgb = cv2.cvtColor(self.cvv.frame, cv2.COLOR_BGR2RGB) # imreadはBGRなのでRGBに変換
                self.loop_img=Image.fromarray(image_rgb)
                #self.loop_img = Image.fromarray(self.cvv.frame)

                # canvas1 disp
                self.canvas_img = ImageTk.PhotoImage(self.loop_img)
                self.canvas.create_image(self.CANVAS_X / 2, self.CANVAS_Y / 2, image=self.canvas_img)
            except cv2.error :
                print( "cv2.error (afterMSec):" )
                return
            
            # canvas2 disp     target img
            try:
                image_rgb = cv2.cvtColor(self.cvv.img, cv2.COLOR_BGR2RGB) # imreadはBGRなのでRGBに変換
                self.loop_img2 = Image.fromarray(image_rgb)
                self.loop_img2 = self.loop_img2.resize( (self.CANVAS2_X, self.CANVAS2_Y))
                self.canvas_img2 = ImageTk.PhotoImage(self.loop_img2)
                self.canvas2.create_image(self.CANVAS2_X / 2, self.CANVAS2_Y / 2, image=self.canvas_img2)
            except cv2.error :
                print( "cv2.error (canvas2)" )
                return

            # canvas3 disp     roope
            image_rgb = cv2.cvtColor(self.cvv.frame, cv2.COLOR_BGR2RGB) # imreadはBGRなのでRGBに変換
            #cv2.imshow('BMP2AVI', image_rgb)
            #print(self.cvv.xc,self.cvv.yc)
            image_rgb = bmp2avi_lib.small_image(image_rgb, self.cvv.xc, self.cvv.yc, 48 )
            self.loop_img3 = Image.fromarray(image_rgb)
            self.loop_img3 = self.loop_img3.resize( (self.CANVAS3_X, self.CANVAS3_Y))
            self.canvas_img3 = ImageTk.PhotoImage(self.loop_img3)
            self.canvas3.create_image(self.CANVAS3_X / 2, self.CANVAS3_Y / 2, image=self.canvas_img3)

        # cam4
        if self.time_ms >= self.cvv.frame_ID2 * self.cam1_fr_interval :
            # canvas4
            try: 
                self.cvv.cameraFrame2()

                image_rgb2 = cv2.cvtColor(self.cvv.frame2, cv2.COLOR_BGR2RGB) # imreadはBGRなのでRGBに変換
                self.loop_img2 = Image.fromarray(image_rgb2)
                # canvas4 disp
                self.canvas_img4 = ImageTk.PhotoImage(self.loop_img2)
                self.canvas4.create_image(self.CANVAS_X / 2, self.CANVAS_Y / 2, image=self.canvas_img4)
            except cv2.error :
                print( "cv2.error (canvas4)" )
                return

        if not self.next_file_flg :
            self.root.after(self.time_ms_step, self.afterMSec)

    def set_flg1_list(self):
        st = self.flg1.get()
        self.cvv.flg1_list[self.cvv.xc_list_id] = int(st)

    # 複数検出切替 xc_list の変更
    def button2_clicked(self):
        print(self.cvv.flg1_list, self.cvv.xc_list)
        self.set_flg1_list()
        self.cvv.update_xc_list_id()
        self.cvv.xc = self.cvv.xc_list[self.cvv.xc_list_id]
        self.cvv.yc = self.cvv.yc_list[self.cvv.xc_list_id]
        self.button2.configure(text='Next Detect: '+str(self.cvv.detect_frame_list[self.cvv.xc_list_id])+' ' +str(self.cvv.xc_list_id +1)+'/'+str(len(self.cvv.xc_list)))
        self.cvv.imgfn = bmp2avi_lib.get_fn_small_image(self.cvv.fn, self.cvv.xc_list_id)
        self.cvv.openFile()
        self.check_flg1_state()
        self.change_radio_button_bgcolor()     
      

    #初期設定
    def firstFrame(self):
        self.time_ms = 0 #avi同期のための、時間
        self.time_ms_step = 5 #[msec]
        self.cam0_fr_interval = 1000.0 / 30.0 # cam0のフレーム間隔(ms) 1000/fr
        self.cam1_fr_interval = 1000.0 / 75.0 # cam0のフレーム間隔(ms) 1000/fr
        #cv2.namedWindow('BMP2AVI') 
        ###self.net = meteor1_eval.load_nn()
        self.first_frame = tk.Frame(self.root, bd=2, relief="ridge", bg="white",
                                    width=self.ROOT_X, height=self.ROOT_Y)
        self.first_frame.grid(row=0, column=0)
        self.create_radio_button()
        self.create_button()
        self.create_radio_button_cam()

        self.label_idx = tk.Label(self.first_frame, text=str("index"),font=("", 16))
        self.label_idx.place(x=10,y=60)
        self.EditBox = tk.Entry(self.first_frame,width=6,font=("", 20))
        self.EditBox.insert(tk.END,"1")
        self.EditBox.place(x=80, y=60)
        self.label_idx_all = tk.Label(self.first_frame, text=str("/ 0"),font=("", 16))
        self.label_idx_all.place(x=160,y=60)

        self.label_nn = tk.Label(self.first_frame, text=str("NN"),font=("", 16))
        self.label_nn.place(x=600,y=60)
        self.label_date = tk.Label(self.first_frame, text=str("YYYY/MM/DD"),font=("", 14))
        self.label_date.place(x=650,y=10)
        self.label_time = tk.Label(self.first_frame, text=str("hh:mm:ss"),font=("", 14))
        self.label_time.place(x=780,y=10)

        self.label_count = tk.Label(self.first_frame, text=str(self.cvv.count_num),font=("", 40))
        self.label_count.place(x=200,y=50,width=300)
        # プログレスバー (確定的)
        self.pb = tk.ttk.Progressbar(
            self.first_frame, 
            orient=tk.HORIZONTAL, 
            length=self.CANVAS2_X, 
            mode='determinate')
        self.pb.configure(maximum=80, value=0) # 90fr = 3.0sec detect = 約40fr
        self.pb.place(x=self.CANVAS_offsetX*2+self.CANVAS_X, y=self.CANVAS_offsetY+self.CANVAS2_Y+310)
            #, sticky=(tk.N, tk.E, tk.S, tk.W))
        #self.pb.grid(row=0, column=0, sticky=(N,E,S,W))
        self.button2 = tk.Button(
            self.root, 
            text='Next Detect: 40 '+str(self.cvv.xc_list_id +1)+'/'+str(len(self.cvv.xc_list)+1),font=("", 16),
            command=self.button2_clicked)
        self.button2.place(x=self.CANVAS_offsetX*2+self.CANVAS_X+60,
            y=self.CANVAS_offsetY+self.CANVAS2_Y+340)

        #  canvas1  main
        self.canvas = tk.Canvas(self.root, width=self.CANVAS_X, height=self.CANVAS_Y)
        self.canvas.create_rectangle(0, 0, self.CANVAS_X, self.CANVAS_Y, fill="#696969")
        self.canvas.place(x=10, y=100)

        #opencvからpil
        image_rgb = cv2.cvtColor(self.cvv.img, cv2.COLOR_BGR2RGB) # imreadはBGRなのでRGBに変換
        self.pil_img=Image.fromarray(image_rgb)

        self.canvas_img = ImageTk.PhotoImage(self.pil_img)
        self.canvas.create_image(self.CANVAS_X / 2, self.CANVAS_Y / 2, image=self.canvas_img)
        
        # canvas2  small画像表示
        self.canvas2 = tk.Canvas(self.root, width=self.CANVAS2_X, height=self.CANVAS2_Y)
        self.canvas2.create_rectangle(0, 0, self.CANVAS2_X, self.CANVAS2_Y, fill="#696969")
        self.canvas2.place(x=self.CANVAS_offsetX*2+self.CANVAS_X, y=self.CANVAS_offsetY)
        #opencvからpil
        image_rgb = cv2.cvtColor(self.cvv.img, cv2.COLOR_BGR2RGB) # imreadはBGRなのでRGBに変換
        self.pil_img=Image.fromarray(image_rgb)

        self.canvas_img2 = ImageTk.PhotoImage(self.pil_img)
        self.canvas2.create_image(self.CANVAS2_X / 2, self.CANVAS2_Y / 2, image=self.canvas_img2)

        # canvas3  ルーペ機能　拡大画像表示
        self.canvas3 = tk.Canvas(self.root, width=self.CANVAS2_X, height=self.CANVAS2_Y)
        self.canvas3.create_rectangle(0, 0, self.CANVAS2_X, self.CANVAS2_Y, fill="#696969")
        self.canvas3.place(x=self.CANVAS_offsetX*2+self.CANVAS_X, y=10+self.CANVAS2_Y+self.CANVAS_offsetY)
        #opencvからpil
        image_rgb = cv2.cvtColor(self.cvv.img, cv2.COLOR_BGR2RGB) # imreadはBGRなのでRGBに変換
        self.pil_img=Image.fromarray(image_rgb)
        self.canvas_img = ImageTk.PhotoImage(self.pil_img)
        self.canvas.create_image(self.CANVAS2_X / 2, self.CANVAS2_Y / 2, image=self.canvas_img)

        #  canvas4  Sub video
        self.canvas4 = tk.Canvas(self.root, width=self.CANVAS_X, height=self.CANVAS_Y)
        self.canvas4.create_rectangle(0, 0, self.CANVAS_X, self.CANVAS_Y, fill="#696969")
        self.canvas4.place(x=10, y=100+self.CANVAS_Y+10)

        #opencvからpil
        image_rgb = cv2.cvtColor(self.cvv.img, cv2.COLOR_BGR2RGB) # imreadはBGRなのでRGBに変換
        self.pil_img=Image.fromarray(image_rgb)

        self.canvas_img = ImageTk.PhotoImage(self.pil_img)
        self.canvas.create_image(self.CANVAS_X / 2, self.CANVAS_Y / 2, image=self.canvas_img)

class CV2video():
    def __init__(self, flist, dn):
        self.flist = flist
        self.target_dir = dn
        self.base_dir='C:/Users/root/Documents/Python Scripts/gym/tmp' #"./tmp"
        self.small_img_dir='C:/Users/root/Documents/Python Scripts/gym/tmp'
        self.del_dir="J:/tmp"
        self.idx = 0
        self.fn=''
        self.imgfn='C:/Users/root/Documents/Python Scripts/gym/tmp/20200501s.png'
        self.count_num=0
        self.frame_ID=0
        self.openFile()
        self.openCamera()
        self.xc = 0
        self.yc = 0
        self.xc_list = []
        self.yc_list = []
        self.detect_frame_list = []
        self.flg1_list = []
        self.xc_list_id = 0
        self.idx2 = 0
        self.fn2=""
        self.frame_ID2=0
        self.openCamera2()

    def openFile(self):
        if not os.path.isfile(self.imgfn):
            print('error(openFile):'+self.imgfn)
            return
        self.img = cv2.imread( self.imgfn )
        #BGR→RGB
        #self.img=self.img[:,:,::-1]

    def get_fn(self):
        if len(self.flist) > self.idx :
            #self.fn = self.flist[self.idx]
            #self.idx +=1
            return self.fn
        return './tmp/20190120_043836_156_00.avi'

    def set_idx(self,  i ):
        if i < 1 :
            i = 1
        if len(self.flist) > (i-1) :
            self.idx = (i-1)
            self.fn = self.flist[self.idx]
    
    def update_idx(self):
        if len(self.flist) == 0:
            print('file list empty (update_idx).')
            return            
        self.idx +=1
        if len(self.flist) > self.idx :
            self.fn = self.flist[self.idx]
        if len(self.flist) <= self.idx :
            self.idx = 0
            self.fn = self.flist[self.idx]
            print('End of file list. Return inx=0')
    def update_idx2(self):
        if len(self.flist) == 0:
            print('file list empty.')
            return            
        self.idx2 +=1
        #if len(self.flist) > self.idx2 :
            #self.fn = self.flist[self.idx2]
        #if len(self.flist) <= self.idx2 :
        #    self.idx2 = 0
        #    self.fn = self.flist[self.idx2]
        #    print('End of file list. Return inx=0')
    def update_xc_list_id(self):
        self.xc_list_id += 1
        if len(self.xc_list) <= self.xc_list_id :
            self.xc_list_id = 0
 
    def set_target_dir(self,dn):
        self.target_dir = dn

    def set_base_dir(self,bd):
        self.base_dir = bd

    def set_del_dir(self,dd):
        self.del_dir = dd

    def set_filelist(self,flist):
        self.flist = flist

    def remove_filelist(self,fn):
        self.flist.remove(fn)

    def openCamera(self):
        #fn="./tmp/20200322_224811_367_00.avi"
        if not os.path.isfile(self.get_fn()):
            print('error(openCamera):'+self.get_fn())
            return
        self.cap = cv2.VideoCapture(self.get_fn())
        if not self.cap.isOpened():
            print( self.fn +" opened fail.")
        self.frame_ID =0
        self.frame_ID2=0
        self.count_num=0

    def cameraFrame(self):
        self.frame_ID+=1
        try:
            self.ret,self.frame=self.cap.read()
        except AttributeError as e:
            print( 'AttributeError(cameraFrame a):  ', e.args )
            self.ret = False
    
        if not self.ret:
            #print("error(1) camaraFrame read()")
            try:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.ret,self.frame=self.cap.read()
            except AttributeError as e:
                print( 'AttributeError(cameraFrame b): ', e.args )
                self.ret = False
            if not self.ret:
                # 2度readエラー
                print("error(2) camaraFrame read()")
                try:
                    self.cap.release()
                except AttributeError as e:
                    print( 'AttributeError(cameraFrame c): ', e.args )               
                self.update_idx()
                self.openCamera()
                self.cameraFrame()
            self.frame_ID =0
            self.frame_ID2=0
            self.count_num=0
        #self.frame_flip = cv2.flip(self.frame, 1)
        #self.frame_flip = self.frame_flip[:, :, ::-1]

    def get_fn2(self, obslist, st):
        self.fn2 = './tmp/20190120_043836_156_00.avi'
        for f in obslist :
            if not (f.find(st) == -1) :
                self.fn2 = f
                break
        #print('fn2 ',obslist,st, self.fn2)
        return self.fn2
        #return './tmp/20190120_043836_156_00.avi'

    def openCamera2(self, obslist=[], st='_4.avi'):
        if not os.path.isfile(self.get_fn2(obslist,st)):
            print('error(openCamera2):'+self.get_fn2(obslist,st))
            return
        self.cap2 = cv2.VideoCapture(self.get_fn2(obslist,st))
        if not self.cap2.isOpened():
            print( self.fn2 +" opened fail.")
        self.frame_ID2=0

    def cameraFrame2(self):
        self.frame_ID2 +=1
        self.ret2,self.frame2=self.cap2.read()
        if not self.ret2:
            #print("error2(1) camaraFrame read()")
            self.cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.ret2,self.frame2=self.cap2.read()
            if not self.ret2:
                # 2度readエラー
                print("error2(2) camaraFrame read()")
                self.cap2.release()
                self.update_idx2()
                self.openCamera2()
                self.cameraFrame2()
            self.frame_ID2=0

class Main:
    def __init__(self, targetpath):
        self.gui=GUI( targetpath )
        self.gui.root.mainloop()


if __name__=="__main__":
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

        dir = dt.strftime("/%Y%m%d")
        TargetDrive   = "J:"
        BaseSoucePath = TargetDrive + "/MT"
        TargetPath1   = BaseSoucePath + dir
        TargetPath2   = BaseSoucePath + dir + "/Fish1"

        main_dir = TargetPath2   #'J:/MT/20200315/fish1/'
        t_dir = './tmp/'
        remake = True
        disp = False

        Main(TargetPath1)