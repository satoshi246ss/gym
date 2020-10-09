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
import sqlite3
import bmp2avi_lib
import copy
###import meteor1_eval

from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG

def setup_logger(log_folder, modname=__name__):
    logger = getLogger(modname)
    logger.setLevel(DEBUG)

    sh = StreamHandler()
    sh.setLevel(DEBUG)
    formatter = Formatter('%(asctime)s - %(name)s(%(lineno)d) - %(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = FileHandler(log_folder) #fh = file handler
    fh.setLevel(DEBUG)
    fh_formatter = Formatter('%(asctime)s - %(filename)s - %(name)s(%(lineno)d) - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    return logger
# 保存するファイル名を指定
#log_folder = '{0}.log'.format(datetime.date.today())
log_folder = 'tk_video.log'
# ログの初期設定を行う
logger = setup_logger(log_folder)

def make_sqlite():
    # 接続。なければDBを作成する。
    conn = sqlite3.connect('meteor.db')

    # カーソルを取得
    c = conn.cursor()
    
    # テーブルを作成
    #                       0     1                            2              3         4          5         6         7         8         9         10        11        12        13        14        15         16         17         18         19         20         21         22         23         24         25             26                    27      28      29               30                31                  32      33      34               35                36              37
    c.execute('CREATE TABLE obs  (id int not null primary key, time datetime, type int, value int, cam0 int, cam1 int, cam2 int, cam3 int, cam4 int, cam5 int, cam6 int, cam7 int, cam8 int, cam9 int, cam10 int, cam11 int, cam12 int, cam13 int, cam14 int, cam15 int, cam16 int, cam17 int, cam18 int, cam19 int, cam_train int, detect_time datetime, xd int, yd int, fish_az_d float, fish_alt_d float, lost_time datetime, xl int, yl int, fish_az_l float, fish_alt_l float, fish_mag float, fish_duration float)')
    # title varchar(1024), body text)')

    # コネクションをクローズ
    conn.close() 

class meteor_sqlite:
    def __init__(self):
        self.db_name = 'meteor.db'
        self.id    = 0
        self.time  = datetime.datetime(year=2019, month=12, day=30, hour=14, minute=0, second=29)
        self.type  = None  # 0:meteor  1:plane   2:other 
        self.value = None  # 0:poor    1:normal  2:good
        self.cam0 = 0      # 0:動画なし 1:あり
        self.cam1 = 0
        self.cam2 = None
        self.cam3 = 0
        self.cam4 = 0
        self.cam5 = None
        self.cam6 = None
        self.cam7 = None
        self.cam8 = 0
        self.cam9 = 0
        self.cam10 = 0
        self.cam11 = 0
        self.cam12 = None
        self.cam13 = 0
        self.cam14 = None
        self.cam15 = 0
        self.cam16 = 0
        self.cam17 = None
        self.cam18 = None
        self.cam19 = None
        self.cam_train = 0  #コマ数

        self.detect_time  = datetime.datetime.now()
        self.xd = 0  # cam0(fisheye)検出座標 xc
        self.yd = 0  # cam0(fisheye)検出座標 yc
        self.fish_az_d  = -99.0
        self.fish_alt_d = -99.0        

        self.lost_time  = datetime.datetime.now()
        self.xl = 0  # cam0(fisheye)検出座標 xc
        self.yl = 0  # cam0(fisheye)検出座標 yc
        self.fish_az_l  = -99.0
        self.fish_alt_l = -99.0        

        self.fish_mag   = -99.0 #magnitude
        self.fish_duration= None #duration[sec]

    def set_data(self, mtime, mtype, mvalue):
        self.time = mtime
        self.type = mtype
        self.value = mvalue

    def add_meteor_data(self):
        self.update_meteor_data()
        # 接続。なければDBを作成する。
        conn = sqlite3.connect(self.db_name)        
        # カーソルを取得
        c = conn.cursor()
        self.id = self.time.strftime("%Y%m%d%H%M%S")
        line = "INSERT INTO {0} VALUES ({1},'{2}',{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},{22},{23},{24},{25},'{26}',{27},{28},{29},{30},'{31}',{32},{33},{34},{35},{36})".format("obs", self.id, self.time, self.type,self.value,self.cam0,self.cam1, self.cam2, self.cam3, self.cam4, self.cam5, self.cam6, self.cam7, self.cam8, self.cam9, self.cam10, self.cam11, self.cam12, self.cam13, self.cam14, self.cam15, self.cam16, self.cam17, self.cam18, self.cam19, self.cam_train, self.detect_time, self.xd, self.yd, self.fish_az_d, self.fish_alt_d, self.lost_time, self.xl, self.yl, self.fish_az_l, self.fish_alt_l, self.fish_mag)
        #print(line)
        #md =  ({0} ,'{1}',{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},{22},{23},{24},'{25}',{26},{27},{28},{29},'{30}',{31},{32},{33},{34},{35})".format( self.id, self.time, self.type,self.value,self.cam0,self.cam1, self.cam2, self.cam3, self.cam4, self.cam5, self.cam6, self.cam7, self.cam8, self.cam9, self.cam10, self.cam11, self.cam12, self.cam13, self.cam14, self.cam15, self.cam16, self.cam17, self.cam18, self.cam19, self.cam_train, self.detect_time, self.xd, self.yd, self.fish_az_d, self.fish_alt_d, self.lost_time, self.xl, self.yl, self.fish_az_l, self.fish_alt_l, self.fish_mag)
        md =  ( self.id, self.time, self.type,self.value,self.cam0,self.cam1, self.cam2, self.cam3, self.cam4, self.cam5, self.cam6, self.cam7, self.cam8, self.cam9, self.cam10, self.cam11, self.cam12, self.cam13, self.cam14, self.cam15, self.cam16, self.cam17, self.cam18, self.cam19, self.cam_train, self.detect_time, self.xd, self.yd, self.fish_az_d, self.fish_alt_d, self.lost_time, self.xl, self.yl, self.fish_az_l, self.fish_alt_l, self.fish_mag, self.fish_duration)
        #print(md)
        # Insert実行
        #                        0     1                            2              3         4          5         6         7         8         9         10        11        12        13        14        15         16         17         18         19         20         21         22         23         24         25             26                    27      28      29               30                31                  32      33      34               35                36
        #c.execute('CREATE TABLE obs  (id int not null primary key, time datetime, type int, value int, cam0 int, cam1 int, cam2 int, cam3 int, cam4 int, cam5 int, cam6 int, cam7 int, cam8 int, cam9 int, cam10 int, cam11 int, cam12 int, cam13 int, cam14 int, cam15 int, cam16 int, cam17 int, cam18 int, cam19 int, cam_train int, detect_time datetime, xd int, yd int, fish_az_d float, fish_alt_d float, lost_time datetime, xl int, yl int, fish_az_l float, fish_alt_l float, fish_mag float)')  
        #c.execute("INSERT INTO articles VALUES (1,'今朝のおかず','魚を食べました','2020-02-01 00:00:00')")
        try:
            sid=( self.id, )
            c.execute('SELECT * FROM obs WHERE id=?',sid)
            if len(c.fetchall()):
                #return True  あり
           	    c.execute('DELETE FROM obs WHERE id=?',sid)
   	            c.execute('INSERT INTO obs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',md )
            else:
                #return False
   	            c.execute('INSERT INTO obs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',md )
        except sqlite3.Error as e:
	        print(e)
        
        # コミット
        conn.commit()
        # コネクションをクローズ
        conn.close() 

    def update_meteor_data(self):
        dt = self.lost_time - self.detect_time
        self.fish_duration = dt.total_seconds()

    def update_cam_status(self, obs_files):
        for f in obs_files:
            if f[-6:-4]  =='00':
                self.cam0 = 1
            elif f[-6:-4]=='_1':
                self.cam1 = 1
            elif f[-6:-4]=='_3':
                self.cam3 = 1
            elif f[-6:-4]=='_4':
                self.cam4 = 1
            elif f[-6:-4]=='_8':
                self.cam8 = 1
            elif f[-6:-4]=='_9':
                self.cam9 = 1
            elif f[-6:-4]=='10':
                self.cam10 = 1
            elif f[-6:-4]=='11':
                self.cam11 = 1
            elif f[-6:-4]=='13':
                self.cam13 = 1
            elif f[-6:-4]=='15':
                self.cam15 = 1
            elif f[-6:-4]=='16':
                self.cam16 = 1

    # WHERE 句の例 pos 3:value  2:type
    def where_time(self, dt, pos=3):
        try:
            conn = sqlite3.connect(self.db_name)        
            cur = conn.cursor()
            id = dt.strftime("%Y%m%d%H%M%S")
            cur.execute('SELECT * FROM obs WHERE id=?', (id,))
            if( cur.fetchone() is None ):
                id = (dt+ datetime.timedelta(seconds=1)).strftime("%Y%m%d%H%M%S")
                cur.execute('SELECT * FROM obs WHERE id=?', (id,))
            
            list1 = cur.fetchone()
            if( list1 is None ):
                ans = None
            else:
                #list1 = cur.fetchone()
                ans = list1[pos]
                print(id,pos,list1[pos])
            conn.close()
            return ans
        except sqlite3.Error as e:
            logger.exception(e)

def test_sql():     
    # 接続。なければDBを作成する。
    conn = sqlite3.connect('meteor.db')
    
    # カーソルを取得
    c = conn.cursor()
    # 1. カーソルをイテレータ (iterator) として扱う
    c.execute('select * from articles')
    for row in c:
        # rowオブジェクトでデータが取得できる。タプル型の結果が取得
        print(row)
    
    # 2. fetchallで結果リストを取得する
    c.execute('select * from articles')
    for row in c.fetchall():
        print(row)
    
    # 3. fetchoneで1件ずつ取得する
    c.execute('select * from articles')
    print(c.fetchone()) # 1レコード目が取得
    print(c.fetchone()) # 2レコード目が取得

    # コネクションをクローズ
    conn.close() 
   
def get_all_obs_files(fullfn):
    #fn = os.path.basename(fullfn)
    path = os.path.dirname(fullfn)
    #file_list = sorted(glob.glob(path+'/*'))
    file_list = sorted([p for p in glob.glob(path+'/**') if os.path.isfile(p)])
    return file_list

def get_all_obs_files_dir(path):
    #fn = os.path.basename(fullfn)
    #path = os.path.dirname(fullfn)
    #file_list = sorted(glob.glob(path+'/*'))
    file_list = sorted([p for p in glob.glob(path+'/**') if os.path.isfile(p)])
    return file_list

def get_same_obs_files(fullfn):
    if len(fullfn) == 0 :
        print('error fn empty. (get_same_obs_files): '+fullfn)
        return ''
        
    fn = os.path.basename(fullfn)
    dt0 = bmp2avi_lib.get_datetime(fn)
    #print('get_same_obs_files : ',fullfn,fn,dt0)

    # (_1　カメラ）データの時刻誤差を調査
    if fullfn[-7:] == '_00.avi':
        with open('time_error_data.txt','a') as fd:
            time_error = 120 #sec 　許容時間誤差
            time_error_dict = {}
            for f in get_all_obs_files(fullfn):
                if f.endswith('_1.avi'):
                    fn = os.path.basename(f)
                    dt1 = bmp2avi_lib.get_datetime(fn)
                    if dt1 >= dt0 :
                        td = dt1 -dt0
                        if td.seconds <=  time_error :
                            key = f[-6:-4]
                            time_error_dict.setdefault(key, []).append([td.seconds, f])
                            st = key+' '+str(td.seconds)+' '+f+'\n'
                            fd.write(st)
                    else :
                        td = dt0 -dt1
                        if td.seconds <=  time_error :
                            key = f[-6:-4]
                            time_error_dict.setdefault(key, []).append([td.seconds, f])
                            st = key+' '+str(td.seconds)+' '+f+'\n'
                            fd.write(st)
            print(time_error_dict)

    #path = os.path.dirname(fullfn)
    #file_list = sorted([p for p in glob.glob(path+'/**') if os.path.isfile(p)])
    time_error = 10 #sec 　許容時間誤差
    time_error_awr = 30 #sec 　train 撮影時間
    same_obs_files=[]
    for f in get_all_obs_files(fullfn):
        fn = os.path.basename(f)
        dt1 = bmp2avi_lib.get_datetime(fn)
        if fn.endswith('.ARW'):
            dt1 = dt1 - datetime.timedelta(seconds= time_error_awr )
        if dt1 >= dt0 :
            td = dt1 -dt0
            if fn.endswith('.ARW'):
                te = time_error + time_error_awr
            else:
                te = time_error
            
            if td.seconds <=  te :
                same_obs_files.append(f)
            #else:
                #break
        else :
            td = dt0 -dt1
            if td.seconds <=  time_error :
                same_obs_files.append(f)
        #print(f,dt0,dt1,td.seconds)
    return same_obs_files

class GUI:
    def __init__(self, targetpath, obsdate):
        self.obsdate = obsdate
        self.obs_path = targetpath
        self.meteor_data = meteor_sqlite()
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

        self.cvv=CV2video(avi_files, nnTargetDir, self.obsdate)
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
        if self.cvv.fn == '' :
            logger.info(targetpath+' : file not found.')
            return
        bmp2avi_lib.make_small_image(self.cvv.fn)
        self.cvv.imgfn = bmp2avi_lib.get_fn_small_image(self.cvv.fn)
        self.cvv.openFile()

        self.firstFrame()
        self.check_flg1_state()
        self.rename_imgdata()
        self.get_detect_loc()
        self.change_radio_button_bgcolor()
        self.change_radio_button3_bgcolor()
        self.label_idx_all.configure(text='/ '+str(len(self.cvv.flist)))
        ###self.get_nn_eval()

        self.afterMSec()

    # select file   in:obs_path
    def select_file(self):
        bmp2avi_lib.rename_00002_avi(self.obs_path)
        #print('選択ファイル名：'+str(self.obs_files))
        print(self.obs_path)
        avi_files = []
        same_files = []
        all_files = get_all_obs_files_dir(self.obs_path)
        ans = os.path.isdir( self.obs_path )
        print(ans, self.obs_path, all_files)
        if len(all_files) == 0 :
            print('Not obsfile exsit.')
            return
        #print('all_files:',all_files)
        for f in all_files :
            if f.endswith("00.avi"):
                avi_files.append(f) 
        for f in avi_files :
            same_files.extend( get_same_obs_files(f) ) # listにlistを追加
        if len(same_files) == 0 :
            non_00avi_files = all_files
        else:
            non_00avi_files = list( set(all_files) - set(same_files) )
        print('non_00avi_files:', non_00avi_files)
        for f in non_00avi_files :
            bmp2avi_lib.remake_00avi(f)

        avi_files = []
        all_files = get_all_obs_files_dir(self.obs_path)
        for f in all_files :
            if f.endswith("00.avi"):
                avi_files.append(f) 

        #print(avi_files)
        self.obsfn = avi_files[0]
        self.cvv.fn = avi_files[0]
        fish_dir =  bmp2avi_lib.serch_fish_dir( self.obsfn )
        bmp2avi_lib.split_log_file(fish_dir+'/log.txt')
        bmp2avi_lib.proc_logfile(avi_files[0])
        self.obs_files = get_same_obs_files( avi_files[0] )
        print('self.obs_path:', self.obs_files)
        self.cvv.set_filelist(avi_files)        
        self.cvv.set_base_dir( os.path.dirname(avi_files[0]) )
        self.cvv.set_del_dir(avi_files[0][:2]+'/tmp')
        self.cvv.set_idx(0)

    #流星以外を移動
    def move_del_files(self, flist):
        for f in flist :
            print('move:'+str(f)+'->'+str(self.cvv.del_dir))
            # tmpにファイルがあれば削除後、移動
            tf = self.cvv.del_dir +'/' + os.path.basename(f)
            if os.path.isfile(tf):
                os.remove(tf)
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

    def change_radio_button3_bgcolor(self):    
        color1 = 'RosyBrown1'
        color2 = 'gray90'
        if self.flg3.get()=='0':
            self.rb31['bg'] = color1
            self.rb32['bg'] = color2
            self.rb33['bg'] = color2
        if self.flg3.get()=='1':
            self.rb31['bg'] = color2
            self.rb32['bg'] = color1
            self.rb33['bg'] = color2
        if self.flg3.get()=='2':
            self.rb31['bg'] = color2
            self.rb32['bg'] = color2
            self.rb33['bg'] = color1

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

        #ラジオボタン3作成
        # value  0:poor　1:good  2:very good
        #
    def create_radio_button3(self) :
        #グループA用変数
        self.flg3 = tk.StringVar()
        self.flg3.set('0')

        #ラジオボタンを押した時の反応用関数
        def rb31_clicked():
            print(self.flg3.get())
            self.change_radio_button3_bgcolor()  
        
        step = 150 ; offset = 70
        #ラジオ1（グループA）
        self.rb31 = tk.Radiobutton(self.root,text='Poor(0)',value=0,variable=self.flg3,
        command=rb31_clicked, font=("",18))
        #rb1.grid(row=1,column=1)
        self.rb31.place(x=4*step+offset, y=8) 
        #ラジオ2（グループA）
        self.rb32 = tk.Radiobutton(self.root,text='Good(1)',value=1,variable=self.flg3,command=rb31_clicked, font=("",18))
        #rb2.grid(row=1,column=2)
        self.rb32.place(x=5*step+offset, y=8) 
    
        #ラジオ3（グループA）
        self.rb33 = tk.Radiobutton(self.root,text='Very good(2)',value=2,variable=self.flg3,command=rb31_clicked, font=("",18))
        #rb3.grid(row=1,column=3)
        self.rb33.place(x=6*step+offset, y=8) 

        #ラジオ4（グループA）
        #rb4 = tk.Radiobutton(self.root,text='Cloud(3)',value=3,variable=self.flg3,command=rb1_clicked)
        #rb4.grid(row=1,column=4)
        #rb4.place(x=4*step, y=8) 

        #ラジオ5（グループA）
        #rb5 = tk.Radiobutton(self.root,text='Ghost(4)',value=4,variable=self.flg3,command=rb1_clicked)
        #rb5.grid(row=1,column=5)
        #rb5.place(x=5*step, y=8) 
        return self.flg3

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

        self.meteor_data.cam0 = 0
        self.meteor_data.cam1 = 0
        self.meteor_data.cam3 = 0
        self.meteor_data.cam4 = 0
        self.meteor_data.cam8 = 0
        self.meteor_data.cam9 = 0
        self.meteor_data.cam10 = 0
        self.meteor_data.cam11 = 0
        self.meteor_data.cam13 = 0
        self.meteor_data.cam15 = 0
        self.meteor_data.cam16 = 0

    def update_radio_button2(self):
        for f in self.obs_files:
            if f[-6:-4]  =='00':
                self.change_radio_button2_bgcolor(0)
                self.meteor_data.cam0 = 1
            elif f[-6:-4]=='_1':
                self.change_radio_button2_bgcolor(1)
                self.meteor_data.cam1 = 1
            elif f[-6:-4]=='_3':
                self.change_radio_button2_bgcolor(3)
                self.meteor_data.cam3 = 1
            elif f[-6:-4]=='_4':
                self.change_radio_button2_bgcolor(4)
                self.meteor_data.cam4 = 1
            elif f[-6:-4]=='_8':
                self.change_radio_button2_bgcolor(8)
                self.meteor_data.cam8 = 1
            elif f[-6:-4]=='_9':
                self.change_radio_button2_bgcolor(9)
                self.meteor_data.cam9 = 1
            elif f[-6:-4]=='10':
                self.change_radio_button2_bgcolor(10)
                self.meteor_data.cam10 = 1
            elif f[-6:-4]=='11':
                self.change_radio_button2_bgcolor(11)
                self.meteor_data.cam11 = 1
            elif f[-6:-4]=='13':
                self.change_radio_button2_bgcolor(13)
                self.meteor_data.cam13 = 1
            elif f[-6:-4]=='15':
                self.change_radio_button2_bgcolor(15)
                self.meteor_data.cam15 = 1
            elif f[-6:-4]=='16':
                self.change_radio_button2_bgcolor(16)
                self.meteor_data.cam16 = 1

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
        self.rbframe.place(x=1010, y=120)

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
        # 
        if(key=="F4"):
            self.flg3.set('0')
        if(key=="F5"):
            self.flg3.set('1')
        if(key=="F6"):
            self.flg3.set('2')
        self.change_radio_button3_bgcolor() 

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
        detect_time=[]
        lost_time=[]
        flg1=[]
        j=0
        lockon_num     = 0
        lockon_num_pre = 0
        path_dir = self.cvv.base_dir
        basename = os.path.basename( self.cvv.fn )
        log_file_name = bmp2avi_lib.serch_logfile(self.cvv.fn )  #path_dir+'/'+basename.replace(".avi","t.txt")
        if log_file_name=='':
            bmp2avi_lib.proc_logfile(self.cvv.fn)
        #print('log_file_name:', log_file_name, basename, path_dir)
        if os.path.exists( log_file_name ) :
            with open( log_file_name, "r") as f:
                strlist = f.readlines()
                for line in strlist:
                    j = j+1
                    lockon_num_pre = lockon_num
                    try:
                        lockon_num = int( line[55:58] )
                    except ValueError as e:
                        print( "ValueError. (get_detect_loc) "+str(lockon_num) )
                        print(e.args)                        
                        continue

                    if lockon_num == 1 and lockon_num_pre == 0 :
                        xc.append(int(line.split( "](" )[1][0:3]))
                        yc.append(int(line.split( "](" )[1][4:7]))
                        detect_frame.append(j)
                        flg1.append(-1)

                        line_lost = line
                        hour = int( line[3:5])
                        mini = int( line[6:8])
                        sec  = int( line[9:11])
                        usec = int( line[12:15]+'000')
                        dettime = datetime.datetime(self.obsdate.year, self.obsdate.month, self.obsdate.day, hour, mini, sec, usec)
                        if( hour < 12 ):                        
                            dettime = dettime + datetime.timedelta(days=1)
                        detect_time.append(dettime)    #24 00:22:30.490  13  42   18[  0.5](105,353)[  0]
                        lost_time.append(dettime)
                        #          1         2         3         4         5         6         7         8         9         0         1         2         3         4
                        #0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890
                        #24 00:22:31.787  13  48   54[ -1.6](445,111)[  0]    0   1 ep=  0.1 dv=  2.3 th=184.9 dif=  0.3  xxc:444.640 110.968 fishAz: 40.38  33.22 MotorAz:  0.00   0.00 MTAz:  0.00   0.00 dis:  0.00 GML:0 TM:1 fo:0
                        self.meteor_data.time = dettime
                        self.meteor_data.detect_time = dettime
                        self.meteor_data.xd = float( line[101:108] )
                        self.meteor_data.yd = float( line[109:116] )
                        self.meteor_data.fish_az_d  = float( line[124:130] )
                        self.meteor_data.fish_alt_d = float( line[130:137] )
                        self.meteor_data.fish_mag = float( line[29:34] )

                    if lockon_num > lockon_num_pre : #update
                        line_lost = line
                        mag = float( line[29:34] )
                        if self.meteor_data.fish_mag > mag :
                            self.meteor_data.fish_mag = mag
                    if lockon_num == 0 and lockon_num_pre > 0 : #Lost
                        line = line_lost
                        hour = int( line[3:5])
                        mini = int( line[6:8])
                        sec  = int( line[9:11])
                        usec = int( line[12:15]+'000')
                        dettime = datetime.datetime(self.obsdate.year, self.obsdate.month, self.obsdate.day, hour, mini, sec, usec)
                        if( hour < 12 ):                        
                            dettime = dettime + datetime.timedelta(days=1)
                        lost_time.pop()
                        lost_time.append(dettime)    #24 00:22:30.490  13  42   18[  0.5](105,353)[  0]
                        #          1         2         3         4         5         6         7         8         9         0         1         2         3         4
                        #0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890
                        #24 00:22:31.787  13  48   54[ -1.6](445,111)[  0]    0   1 ep=  0.1 dv=  2.3 th=184.9 dif=  0.3  xxc:444.640 110.968 fishAz: 40.38  33.22 MotorAz:  0.00   0.00 MTAz:  0.00   0.00 dis:  0.00 GML:0 TM:1 fo:0
                        #self.meteor_data.time = dettime
                        self.meteor_data.lost_time = dettime
                        self.meteor_data.xl = float( line[101:108] )
                        self.meteor_data.yl = float( line[109:116] )
                        self.meteor_data.fish_az_l  = float( line[124:130] )
                        self.meteor_data.fish_alt_l = float( line[130:137] )
                    
                for i in range(0,len(xc)):
                    print(i)
                    print( "idx:"+str(self.cvv.idx) + " detectFrame:"+str(detect_frame[i])+ " (xc,yc)["+str(i)+"]=("+str(xc[i])+","+str(yc[i])+") "+detect_time[i].isoformat()+lost_time[i].isoformat())
            
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
            flg1_list_pre = copy.copy( self.cvv.flg1_list ) #deep copy 結果保持

            # meteor data 追加
            if flg1_list_pre.count(0) == 0 : # 流星なし
                self.meteor_data.type = int( flg1_list_pre[0] )
            else:
                self.meteor_data.type = 0  # meteor
            self.meteor_data.update_cam_status(self.obs_files)
            self.meteor_data.value = int(self.flg3.get())  # meteor value
            self.get_detect_loc()
            #self.meteor_data.set_data()            
            self.meteor_data.add_meteor_data()

            bmp2avi_lib.make_small_image(self.cvv.fn)
            self.rename_imgdata()
            # Meteor以外は、move file
            print(flg1_list_pre, self.cvv.flg1_list, self.cvv.flg1_list.count(0))
            if flg1_list_pre.count(0) == 0 :
                self.move_del_files(self.obs_files)
                print(self.obs_files)
                self.cvv.del_files_update_idx()
                self.EditBox.delete(0, tk.END)
                self.EditBox.insert(tk.END,self.cvv.idx+1)

            value = self.EditBox.get()
            self.cvv.set_idx( int(value) )
            self.label_idx_all.configure(text='/ '+str(len(self.cvv.flist)))
             
            print('cb02 '+str(self.cvv.idx)+' '+str(len(self.cvv.flist)))
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

            # meteor data 追加
            #self.get_detect_loc()
            #self.meteor_data.set_data()
            #self.meteor_data.add_meteor_data()

            self.cvv.imgfn = bmp2avi_lib.get_fn_small_image(self.cvv.fn, self.cvv.xc_list_id)
            #print(self.cvv.imgfn)
            self.cvv.openFile()
            self.button2.configure(text='Next Detect: '+str(self.cvv.detect_frame_list[self.cvv.xc_list_id])+' ' +str(self.cvv.xc_list_id +1)+'/'+str(len(self.cvv.xc_list)))
            self.check_flg1_state()
            #self.change_radio_button_bgcolor()
            self.update_button_bgcolor()
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
        self.update_button_bgcolor()

    def update_button_bgcolor(self):
        # DB からあれば、そのデータなければ'0'
        ans = self.meteor_data.where_time(bmp2avi_lib.get_datetime( os.path.basename(self.cvv.fn)),3)
        if ans is None :
            self.flg3.set('0')
        else:
            self.flg3.set( str(ans) )     
        self.change_radio_button3_bgcolor()     

        ans = self.meteor_data.where_time(bmp2avi_lib.get_datetime( os.path.basename(self.cvv.fn)),2)
        if ans is None :
            self.flg1.set('0')
        else:
            self.flg1.set( str(ans) )     
        self.change_radio_button_bgcolor()     

    def afterMSec(self):
        self.cvv.count_num+=1
        self.time_ms = self.cvv.count_num * self.time_ms_step
        self.label_count.configure(text=str(self.cvv.idx)+":"+  "{:3d}".format(self.cvv.frame_ID))
        self.pb.configure(value=self.cvv.frame_ID)
        #print(self.cvv.detect_frame_list,self.cvv.xc_list_id)
        #if len(self.cvv.detect_frame_list) > 0 :
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
        self.first_frame = tk.Frame(self.root, bd=2, relief="ridge", bg="gray50",
                                    width=self.ROOT_X, height=self.ROOT_Y)
        self.first_frame.grid(row=0, column=0)
        self.create_radio_button()
        self.create_radio_button3()
        self.create_button()
        self.create_radio_button_cam()

        self.label_idx = tk.Label(self.first_frame, text=str("index"),font=("", 16))
        self.label_idx.place(x=10,y=60)
        self.EditBox = tk.Entry(self.first_frame,width=6,font=("", 20))
        self.EditBox.insert(tk.END,"1")
        self.EditBox.place(x=80, y=60)
        self.label_idx_all = tk.Label(self.first_frame, text=str("/ 0"),font=("", 20))
        self.label_idx_all.place(x=160,y=60)

        self.label_nn = tk.Label(self.first_frame, text=str("NN"),font=("", 16))
        self.label_nn.place(x=600,y=60)
        self.label_date = tk.Label(self.first_frame, text=str("YYYY/MM/DD"),font=("", 14))
        self.label_date.place(x=650,y=10)
        self.label_time = tk.Label(self.first_frame, text=str("hh:mm:ss"),font=("", 14))
        self.label_time.place(x=780,y=10)

        self.label_count = tk.Label(self.first_frame, text=str(self.cvv.count_num),font=("", 40))
        self.label_count.place(x=220,y=50,width=280)
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
    def __init__(self, flist, dn, obsdate):
        self.obsdate = obsdate
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
        self.idx +=1
        if len(self.flist) == 0:
            self.idx = 0
            print('file list empty (update_idx).')
            return            
        if len(self.flist) == 1:
            self.idx = 0
            self.fn = self.flist[self.idx]
        elif len(self.flist) > self.idx :
            self.fn = self.flist[self.idx]
        elif len(self.flist) <= self.idx :
            self.idx = 0
            self.fn = self.flist[self.idx]
            print('End of file list. Return idx=0')

    def del_files_update_idx(self):
        self.idx -= 1
        if len(self.flist) == 0:
            self.idx = 0
            print('file list empty (update_idx).')
            return            
        if len(self.flist) == 1:
            self.idx = 0
            self.fn = self.flist[self.idx]
        elif len(self.flist) > self.idx :
            self.fn = self.flist[self.idx]
        elif len(self.flist) <= self.idx :
            self.idx = 0
            self.fn = self.flist[self.idx]
            print('End of file list. Return idx=0')
        
        if self.idx < 0 :
            self.idx = len(self.flist)-1 
            self.fn = self.flist[self.idx]
        print('del_files_update_idx():'+str(self.idx)+' '+str(len(self.flist)))

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
    def __init__(self, targetpath, obsdate):
        #make_sqlite()
        
        #mdata = meteor_sqlite()
        #mdata.time = datetime.datetime.now()
        #mdata.add_meteor_data()

        logger.info(targetpath)
        self.gui=GUI( targetpath, obsdate )
        self.gui.root.mainloop()


if __name__=="__main__":
    logger.debug(sys.argv)
    BaseSoucePath=''
    dtnow = datetime.datetime.now()
    drange=1 #実行日数（戻り日数）
    if len( sys.argv )   >= 6:
        yyyy=int(sys.argv[1])
        mm  =int(sys.argv[2])
        dd  =int(sys.argv[3])
        drange =int(sys.argv[4])
        BaseSoucePath = sys.argv[5] 
    elif len( sys.argv )  == 5:
        yyyy=int(sys.argv[1])
        mm  =int(sys.argv[2])
        dd  =int(sys.argv[3])
        drange =int(sys.argv[4])
    elif len( sys.argv )  == 4:
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
        #drange =7

    print( BaseSoucePath)
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
        if BaseSoucePath=='':
            TargetDrive   = "J:"
            BaseSoucePath = TargetDrive + "/MT"
        TargetPath1   = BaseSoucePath + dir
        TargetPath2   = BaseSoucePath + dir + "/Fish1"

        main_dir = TargetPath2   #'J:/MT/20200315/fish1/'
        t_dir = './tmp/'
        remake = True
        disp = False

        Main(TargetPath1, dt)