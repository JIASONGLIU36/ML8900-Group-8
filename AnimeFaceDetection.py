# Author: Ratnaker, Jiasong Liu
# This is the front end of the AnimeFaceDetection system
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import Back_end


image_id = "Thisimage" # the image_id for the detection image
class Upload_window:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Uploader")
        self.root.geometry("600x400")
        
        upload_button = tk.Button(root, text="Upload Image", command=self.upload_and_process_image)
        upload_button.pack(pady=50)

    def upload_and_process_image(self):
        file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg;*.jpeg")])
        image_list = []
        if file_path:
            try:
                # a waiting window
                self.please_wait_window()
                # check if all file is generated in place, if not, give a message
                result = Back_end.DetectionProcess(file_path, image_id)
                # if the weight file does not exist or not in correct dir, this error is going to raised
                if result[2]['MaskRCNN'] == False:
                    raise Exception("Mask RCNN running failed, please check the weights location and name") 
                
                image_list.append(Image.open('./temp_img/'+image_id+'_faster.png'))
                image_list.append(Image.open('./temp_img/'+image_id+'_mask.png'))

                # Open the new window with the image
                # with a new object
                self.root.destroy()
                self.root = tk.Tk()
                self.app = Show_result_window(self.root, image_list, result[0], result[1])
                self.root.mainloop()

            except Exception as e:
                # Handle the situation where the user uploads a file that is not an image
                print(f"Error: {e}")
                messagebox.showerror("Error", e)
    
    def please_wait_window(self):
        wait_window = tk.Toplevel(self.root)
        wait_window.title("Please Wait")
        wait_window.resizable(False, False)
        Message = tk.Label(wait_window, text="Please wait until Image processing complete...", compound="top")
        Message.pack(pady=(20, 10))
        self.root.update()



class Show_result_window:

    def __init__(self, root, image, crop_img_num_mask, crop_img_num_faster):
        self.root = root
        # Create a new window
        root.title("Image Display")
        self.show_image_window(image, crop_img_num_mask, crop_img_num_faster)
        self.curr_subimg_mask = 0
        self.curr_subimg_faster = 0
        self.curr_feature = 'emotion'


    def selection_changed_for_maskfaces(self, *args):
        #print(f"Selection changed to: {self.crop_img_option.get()}")
        self.curr_subimg_mask = int(self.crop_img_option_1.get()[-1])
        print("Selected cropped Face for mask RCNN is:" + str(self.curr_subimg_mask))

    def selection_changed_for_fasterfaces(self, *args):
        #print(f"Selection changed to: {self.crop_img_option.get()}")
        self.curr_subimg_faster = int(self.crop_img_option_3.get()[-1])
        print("Selected cropped Face for faster RCNN is:" + str(self.curr_subimg_faster))

    def selection_changed_for_feature(self, *args):
        #print(f"Selection changed to: {self.crop_img_option.get()}")
        self.curr_feature = self.crop_img_option_2.get()
        print("Selected feature is:" + self.curr_feature)

    def show_image_window(self, image, crop_img_num_mask, crop_img_num_faster):

        # Create left and right panels
        left_panel = tk.Label(self.root, text="Mask R-CNN", compound="top")
        right_panel = tk.Label(self.root, text="Faster R-CNN", compound="top")

        left_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        right_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Add Output section
        self.output_label_mask = tk.Label(self.root, text="Output Section - Mask R-CNN")
        self.output_label_mask.grid(row=1, column=0, padx=10, pady=10)

        self.output_label_faster = tk.Label(self.root, text="Output Section - Faster R-CNN")
        self.output_label_faster.grid(row=1, column=1, padx=10, pady=10)


        Frame_for_model_buttons_1 = tk.Frame(self.root)
        Frame_for_model_buttons_1.grid(row=2, column=1) # put the button on the bottom left
        # reset button, go back to upload window
        reset_button = tk.Button(Frame_for_model_buttons_1, text='Reset Image', command=self.Reset_image)
        reset_button.pack(side=tk.LEFT)

        # Add model buttons
        model_buttons = ['Inception V4','Resnet 34','Resnet 50','Resnet 101','VGG 16','VGG 19']  # Add more buttons if necessary

        Frame_for_model_buttons_2 = tk.Frame(self.root)
        Frame_for_model_buttons_2.grid(row=2, column=0) # put the button on the bottom left

        if crop_img_num_faster > 0 or crop_img_num_mask > 0:
            for model in model_buttons:
                # need to have frame as master
                model_button = tk.Button(Frame_for_model_buttons_2, text=model, command=lambda m=model: self.show_output(m, crop_img_num_mask, crop_img_num_faster))
                model_button.pack(side=tk.LEFT)

            Frame_for_feature_dropdown = tk.Frame(self.root)
            Frame_for_feature_dropdown.grid(row=1, column=2)
            # show title of dropdown menu
            feature_label = tk.Label(Frame_for_feature_dropdown, text="Feature selection", compound="top")
            feature_label.pack(side=tk.TOP)
            # show dropdown menu for different features
            self.crop_img_option_2 = tk.StringVar(Frame_for_feature_dropdown)
            self.crop_img_option_2.set("emotion")
            # call back for other options
            self.crop_img_option_2.trace("w", self.selection_changed_for_feature)

            # Options for the dropdown menu
            #options = ['emotion', 'hairstyle', 'eyecolor']
            # we disable other features for this project, please check the report for reason
            options = ['emotion']

            # Create the dropdown menu
            dropdown_menu = tk.OptionMenu(Frame_for_feature_dropdown, self.crop_img_option_2, *options)
            dropdown_menu.pack(side=tk.TOP)

        # show dropdown menu if there are cropped faces
        Frame_for_cropface_dropdown = tk.Frame(self.root)
        Frame_for_cropface_dropdown.grid(row=0, column=2)

        # if there are no faces found, then don't show the options
        if crop_img_num_mask > 0:

            # show title of dropdown menu
            cropFace_label = tk.Label(Frame_for_cropface_dropdown, text="face selection MaskRCNN", compound="top")
            cropFace_label.pack(side=tk.TOP)

            self.crop_img_option_1 = tk.StringVar(Frame_for_cropface_dropdown)
            self.crop_img_option_1.set("crop_face_0")
            # call back for other options
            self.crop_img_option_1.trace("w", self.selection_changed_for_maskfaces)

            # Options for the dropdown menu
            options = ["crop_face_"+str(i) for i in range(crop_img_num_mask)]

            # Create the dropdown menu
            dropdown_menu = tk.OptionMenu(Frame_for_cropface_dropdown, self.crop_img_option_1, *options)
            dropdown_menu.pack(side=tk.TOP)

        if crop_img_num_faster > 0:

            # show title of dropdown menu
            cropFace_label = tk.Label(Frame_for_cropface_dropdown, text="face selection FasterRCNN", compound="top")
            cropFace_label.pack(side=tk.TOP)

            self.crop_img_option_3 = tk.StringVar(Frame_for_cropface_dropdown)
            self.crop_img_option_3.set("crop_face_0")
            # call back for other options
            self.crop_img_option_3.trace("w", self.selection_changed_for_fasterfaces)

            # Options for the dropdown menu
            options = ["crop_face_"+str(i) for i in range(crop_img_num_faster)]

            # Create the dropdown menu
            dropdown_menu = tk.OptionMenu(Frame_for_cropface_dropdown, self.crop_img_option_3, *options)
            dropdown_menu.pack(side=tk.TOP)

        # Configure rows to be uniform
        self.root.rowconfigure(2, weight=1)

        # Add the image to the left and right panels
        self.display_image(left_panel, image[1])
        self.display_image(right_panel, image[0])

    def Reset_image(self):
        # Open the window with upload images
        self.root.destroy()
        self.root = tk.Tk()
        self.app = Upload_window(self.root)
        self.root.mainloop()

    def display_image(self, panel, image):
        # Convert the PIL Image to a PhotoImage object for the panel
        panel.photo = ImageTk.PhotoImage(image)
        panel.config(image=panel.photo)

    def parse_label(self, input, feature):

        emd = {0:'negative',1:'positive', 2:'placeholder'}
        hsd = {0:'short hair',1:'long hair'}
        ecd = {0:'none',1:'red',2:'orange',3:'yellow',4:'green',5:'blue',6:'purple',7:'violet',8:'black'}
        # faster and mask
        if input == -1:
            output = "weight not found or other error"
        else:
            if feature == 'hairstyle':
                output = hsd[input]
            elif feature == 'emotion':
                output = emd[input]
            elif feature == 'eyecolor':
                output = ecd[input]

        return output

    # show the output for each cropped images
    def show_output(self, model, mask_crop, faster_crop):

        # initialize location of cropped faces
        if mask_crop == 0:
            self.curr_subimg_mask = -1
        if faster_crop == 0:
            self.curr_subimg_faster = -1

        #print(self.crop_img_option.get())
        window_for_waiting = tk.Tk()
        self.please_wait_window(window_for_waiting)
        if self.curr_subimg_mask != -1:
            output_label_mask = Back_end.step_2_models_Mask(image_id, self.curr_subimg_mask, self.curr_feature, model)
            Mask_result = self.parse_label(output_label_mask, self.crop_img_option_2.get())
            # generate output message
            output_mask = f"Mask R-CNN Output for Model {model} for "+ self.crop_img_option_1.get()+"\n"+self.crop_img_option_2.get()+": "+Mask_result
        else:
            output_mask = f"Mask R-CNN Output did not detect any faces"

        if self.curr_subimg_faster != -1:
            output_label_faster = Back_end.step_2_models_Faster(image_id, self.curr_subimg_faster, self.curr_feature, model)
            Faster_result = self.parse_label(output_label_faster, self.crop_img_option_2.get())
            # generate output message
            output_faster = f"Faster R-CNN Output for Model {model} for "+ self.crop_img_option_3.get()+"\n"+self.crop_img_option_2.get()+": "+Faster_result
        else:
            output_faster = f"Faster R-CNN Output did not detect any faces"
        
        window_for_waiting.destroy()

        # Placeholder for the actual output generation based on the selected model
        #output_mask = f"Mask R-CNN Output for Model {model} for "+ self.crop_img_option_1.get()+"\n"+self.crop_img_option_2.get()+": "+Mask_result
        #output_faster = f"Faster R-CNN Output for Model {model} for "+ self.crop_img_option_1.get()+"\n"+self.crop_img_option_2.get()+": "+Faster_result

        # Update the output labels in the image window
        self.output_label_mask.config(text=output_mask)
        self.output_label_faster.config(text=output_faster)

    def please_wait_window(self, thisroot):
        wait_window = thisroot
        wait_window.title("Please Wait")
        wait_window.resizable(False, False)
        Message = tk.Label(wait_window, text="Please wait until Image processing complete...", compound="top")
        Message.pack(pady=(20, 10))
        self.root.update()

if __name__ == "__main__":
    root = tk.Tk()
    app = Upload_window(root)
    root.mainloop()
