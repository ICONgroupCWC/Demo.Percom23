from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_size = (300,300)
image_path = "D:/dataset1/image"
label_path = "D:/dataset1/label"
images_save_dir = "D:/dataset1/augmented/image"
labels_save_dir = "D:/dataset1/augmented/label"
batch_size = 100

data_gen_args = dict(
                    featurewise_center=True,
                    featurewise_std_normalization=True,
                    rotation_range=10,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    zoom_range=0.2,
                    brightness_range=[0.2,1.2],
                    shear_range=0.05,
                    #zoom_range=zoom_range,
                    horizontal_flip=True,
                    fill_mode='constant',
                    cval=0)

image_datagen = ImageDataGenerator(**data_gen_args)

image_generator = image_datagen.flow_from_directory(
        image_path ,
        color_mode = "grayscale",
        target_size = img_size,
        batch_size = batch_size,
        save_to_dir = images_save_dir,
        save_prefix  = "image",
        seed = 2)

label_generator = image_datagen.flow_from_directory(
        label_path ,
        target_size = img_size,
        batch_size = batch_size,
        save_to_dir = labels_save_dir,
        save_prefix  = "label",
        seed = 2)


for i in range(0,batch_size):
  
  next(image_generator)[0].astype('uint8')
  next(label_generator)[0].astype('uint8')
  
