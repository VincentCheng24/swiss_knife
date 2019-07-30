#!/usr/local/bin/python3

import cv2
import os


def img2video(img_dir, output='output.mp4', ext='png', show_frame_num=False):

    if show_frame_num:
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (30, 30)
        fontScale = 1
        fontColor = (255, 255, 0)
        lineType = 2

    images = []
    for f in os.listdir(img_dir):
        if f.endswith(ext):
            images.append(f)

    images.sort()

    # Determine the width and height from the first image
    image_path = os.path.join(img_dir, images[0])
    frame = cv2.imread(image_path)
    cv2.imshow('video', frame)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, 100.0, (width, height))

    for i, image in enumerate(images):

        image_path = os.path.join(img_dir, image)
        frame = cv2.imread(image_path)

        if show_frame_num:
            frame = cv2.putText(frame, 'F{:06d}'.format(i), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

        out.write(frame) # Write out frame to video

        cv2.imshow('video', frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
            break

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()

    print("The output video is {}".format(output))


if __name__ == '__main__':
    img_dir = ''

    img2video(img_dir=img_dir, output='output.mp4')
