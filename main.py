
#! /usr/bin/env python
'''
import os
import cv2
import argparse

from face_detection import select_face, select_all_faces
from face_swap import face_swap


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FaceSwapApp')
    parser.add_argument('--src', required=True, help='Path for source image')
    parser.add_argument('--dst', required=True, help='Path for target image')
    parser.add_argument('--out', required=True, help='Path for storing output images')
    parser.add_argument('--warp_2d', default=False, action='store_true', help='2d or 3d warp')
    parser.add_argument('--correct_color', default=False, action='store_true', help='Correct color')
    parser.add_argument('--no_debug_window', default=False, action='store_true', help='Don\'t show debug window')
    args = parser.parse_args()

    # Read images
    src_img = cv2.imread(args.src)
    dst_img = cv2.imread(args.dst)

    # Select src face
    src_points, src_shape, src_face = select_face(src_img)
    # Select dst face
    dst_faceBoxes = select_all_faces(dst_img)

    if dst_faceBoxes is None:
        print('Detect 0 Face !!!')
        exit(-1)

    output = dst_img
    for k, dst_face in dst_faceBoxes.items():
        output = face_swap(src_face, dst_face["face"], src_points,
                           dst_face["points"], dst_face["shape"],
                           output, args)

    dir_path = os.path.dirname(args.out)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    cv2.imwrite(args.out, output)

    ##For debug
    if not args.no_debug_window:
        cv2.imshow("From", dst_img)
        cv2.imshow("To", output)
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()
'''

import os

import numpy
import argparse
import streamlit as st
from PIL import Image, ImageEnhance
from face_detection import select_face, select_all_faces
from face_swap import face_swap
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FaceSwapApp')
    parser.add_argument('--correct_color',default=True, action = 'store_true',help = 'Correct color')
    parser.add_argument('--warp_2d',default=False, action='store_true',help = '2d or 3d warp')
    args = parser.parse_args()
    uploaded_scfile = st.file_uploader("Source File",type = ['jpg','png','jpeg'])
    uploaded_tgfile = st.file_uploader("Target File",type = ['jpg','png','jpeg'])
    
    if uploaded_scfile is not None and uploaded_tgfile is not None:
        source_image = Image.open(uploaded_scfile)
        target_image = Image.open(uploaded_tgfile)
        
        src_img = cv2.cvtColor(numpy.array(source_image), cv2.IMREAD_COLOR)
        dst_img = cv2.cvtColor(numpy.array(target_image), cv2.IMREAD_COLOR)
    
        src_points, src_shape, src_face = select_face(src_img)
        
        dst_faceBoxes = select_all_faces(dst_img)
        
        if dst_faceBoxes is None:
            print('Detext 0 Face !!')
            exit(-1)
        
        output = dst_img
        
        for k, dst_face in dst_faceBoxes.items():
            output = face_swap(src_face, dst_face["face"],src_points,\
                dst_face["points"],dst_face["shape"],output,args)
            
            
            st.markdown('<p style = "text-align: left;">Result</p>',unsafe_allow_html=True)
            
            st.image(output, width=500)
            
                