from tensorflow.keras.models import load_model
import cv2
import numpy as np
import warnings
import blur
warnings.simplefilter("ignore")
import os


def read_img(img_path):
    batch = []
    height = 512
    width = 512
    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, (height, width))
    img = img / 255.
    batch.append(img)
    batch = np.array(batch)
    return batch


def mask_blur(original_img, blur_img, predicted_img):
    # print("Shape: ", original_img.shape, blur_img.shape, predicted_img.shape)
    # cv2.imshow('img', predicted_img)
    # cv2.waitKeyEx(0)

    blue_channel_ori = original_img[:, :, 0]
    green_channel_ori = original_img[:, :, 1]
    red_channel_ori = original_img[:, :, 2]

    blue_channel_blr = blur_img[:, :, 0]
    green_channel_blr = blur_img[:, :, 1]
    red_channel_blr = blur_img[:, :, 2]

    blue_channel_pre = predicted_img[:, :, 0]
    green_channel_pre = predicted_img[:, :, 1]
    red_channel_pre = predicted_img[:, :, 2]

    new_b = []
    new_g = []


    mks_img_new = np.zeros([512, 512, 3])

    for i in range(3):
        if i == 0:
            img = blue_channel_blr
            msk = blue_channel_pre
            ori = blue_channel_ori
        if i == 1:
            img = green_channel_blr
            msk = green_channel_pre
            ori = green_channel_ori
        if i == 2:
            img = red_channel_blr
            msk = red_channel_pre
            ori = red_channel_ori

        if i == 0:
            new = new_b
        if i == 1:
            new = new_g
        if i == 2:
            new = new_r

        img = img.reshape(1, -1)[0]
        msk = msk.reshape(1, -1)[0]
        ori = ori.reshape(1, -1)[0]

        for k, m, o in zip(img, msk, ori):
            if int(m*255.) < 50:
                new.append(k)
            else:
                new.append(o)

        if i == 0:
            new_b = np.array(new_b).reshape(512, 512)
            mks_img_new[:, :, 0] = new_b
        if i == 1:
            new_g = np.array(new_g).reshape(512, 512)
            mks_img_new[:, :, 1] = new_g
        if i == 2:
            new_r = np.array(new_r).reshape(512, 512)
            mks_img_new[:, :, 2] = new_r

    return mks_img_new


def create_img(original_img, predicted_img, ch=1):
    blur_img = blur.make_blur(original_img[0])
    img = mask_blur(original_img[0], blur_img, predicted_img[0])

    if ch == 1:
        cv2.imshow("Img1", original_img[0])
        cv2.waitKeyEx(0)
        cv2.imshow("Img2", predicted_img[0])
        cv2.waitKeyEx(0)
        cv2.imshow("Img3", img)
        cv2.waitKeyEx(0)

    try:
        os.mkdir("results")
    except:
        pass
    cv2.imwrite("results/Original.jpg", original_img[0] * 255.)
    cv2.imwrite("results/PredictedMask.jpg", predicted_img[0] * 255.)
    cv2.imwrite("results/BlurImg.jpg", img * 255.)


if __name__ == '__main__':
    img_name = input("Enter Image Path or Image name: ")
    ch = input("Enter 1 to show and save images | Enter 2 for only save images: ")
    if ch == "1":
        ch = int(ch)
    elif ch == "2":
        ch = int(ch)
    else:
        print("Invalid Input, default is show and save images")
    img = read_img(img_path=img_name)
    model = load_model('Unet_black_background_81_epochs.h5')
    res = model.predict(img)
    create_img(img, res, ch)
    print("Done")
