import tkinter
import tkinter.filedialog  as tkdialog
from tkinter import font
from PIL import Image, ImageTk
import os
import cv2
 
class AppForm(tkinter.Frame):
     
    def __init__(self, master=None):
        super().__init__(master, height=500, width=660)
        self.pack()
        self.create_widgets()
        self.menubar_create()
     
    def create_widgets(self):
        self.canvas = tkinter.Canvas(
                root,                   
                width = 640,            
                height = 480,           
                relief= tkinter.RIDGE,  
                bd=0                   
        )
        self.canvas.place(x=10, y=40)   
 
    def menubar_create(self):
         
        self.menubar = tkinter.Menu(root)
 
        filemenu = tkinter.Menu(self.menubar, tearoff=0)
        filemenu.add_command(label="Open", command=self.File_open)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=root.quit)
        self.menubar.add_cascade(label="File", menu=filemenu)
 
        editmenu = tkinter.Menu(self.menubar, tearoff=0)
        editmenu.add_command(label="Grayscale", command=self.Proc_grayscale)
        editmenu.add_command(label="Binarize", command=self.Proc_binarize)
        self.menubar.add_cascade(label="Processing", menu=editmenu)
 
        root.config(menu=self.menubar)
        root.config()
 
    def Disp_image(self, im_temp):
        self.img_temp = ImageTk.PhotoImage(Image.fromarray(im_temp))  
        self.canvas.create_image(               
            0,                                  
            0,                                  
            image = self.img_temp,              
            anchor = tkinter.NW                 
        )
 
    def File_open(self):
        fname = tkdialog.askopenfilename(filetypes=[("bmp files","*.bmp"),("png files","*.png")],initialdir=os.getcwd())
        image_bgr = cv2.imread(fname)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # imreadはBGRなのでRGBに変換
        self.im   = image_rgb # RGBからPILフォーマットへ変換
        #self.im   = Image.fromarray(image_rgb) # RGBからPILフォーマットへ変換
        self.Disp_image(self.im)
 
    def Proc_grayscale(self):
        self.im_gray = image_proc.Image_grayscale(self.im)
        self.Disp_image(self.im_gray)
     
    def Proc_binarize(self):
        self.ret, self.im_bin = image_proc.Image_binarize(self.im_gray)
        self.Disp_image(self.im_bin)
 
def Image_read(img_name):
    im = cv2.imread(img_name)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im
 
def Image_save(img_name, img_temp):
    if img_temp.ndim == '3':  # color image (gray image : 2)
        img_temp = cv2.cvtColor(img_temp, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_name, img_temp)
 
def Image_grayscale(img_temp):
    im_gray = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
    return im_gray
 
def Image_binarize(img_temp):
    ret, im_bin = cv2.threshold(img_temp, 60, 255, cv2.THRESH_BINARY)
    return ret, im_bin

    #ラジオボタン作成
def create_radio_button() :
        #グループA用変数
    flg1 = tkinter.StringVar()
    flg1.set('1')

    #ラジオボタンを押した時の反応用関数
    def rb1_clicked():
        print(flg1.get())

 
    #ラジオ1（グループA）
    rb1 = tkinter.Radiobutton(app,text='Meteor(0)',value=0,variable=flg1,command=rb1_clicked)
    rb1.grid(row=1,column=1)
 
    #ラジオ2（グループA）
    rb2 = tkinter.Radiobutton(app,text='Plane(1)',value=1,variable=flg1,command=rb1_clicked)
    rb2.grid(row=1,column=2)
    #rb2.state(['disabled'])
 
    #ラジオ3（グループA）
    rb3 = tkinter.Radiobutton(app,text='Ghost(2)',value=2,variable=flg1,command=rb1_clicked)
    rb3.grid(row=1,column=3)

    #ラジオ4（グループA）
    rb4 = tkinter.Radiobutton(app,text='Cloud(3)',value=3,variable=flg1,command=rb1_clicked)
    rb4.grid(row=1,column=4)

    #ラジオ5（グループA）
    rb5 = tkinter.Radiobutton(app,text='Other(4)',value=4,variable=flg1,command=rb1_clicked)
    rb5.grid(row=1,column=5)
    return flg1

def create_button() :    
    #ボタンを押した時の反応用関数
    def button1_clicked():
        print("B1")
        
    # 参照ボタン配置  
    button1 = tkinter.Button(root, text=u'OK', command=button1_clicked)  
    #button1.grid(row=0, column=1)  
    button1.place(x=2, y=0)  

if __name__ == '__main__':
    root = tkinter.Tk()
    root.title("https://blog.fc2.com/tag/OpenCV")
    #root.title("<a href="https://blog.fc2.com/tag/OpenCV" class="tagword">OpenCV</a> Developer")
    root.option_add('*font', ('MS Sans Serif', 16))
    app = AppForm(master=root)

    flg = create_radio_button()
    create_button()
    root.geometry("640x640+100+120")
    app.mainloop()

    print(flg.get())