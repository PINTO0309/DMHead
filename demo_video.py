
import cv2
import time
import argparse
import onnxruntime
import numpy as np
from math import cos, sin


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=100):
    # Referenced from HopeNet https://github.com/natanielruiz/deep-head-pose
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2
    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy
    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy
    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy
    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)
    return img


def main(args):
    # YOLOv4-Head
    yolov4_head = onnxruntime.InferenceSession(
        path_or_bytes=f'yolov4_headdetection_480x640_post.onnx',
        providers=[
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
    )
    yolov4_head_input_name = yolov4_head.get_inputs()[0].name
    yolov4_head_H = yolov4_head.get_inputs()[0].shape[2]
    yolov4_head_W = yolov4_head.get_inputs()[0].shape[3]

    # DMHead
    dmhead_input_name = None
    dmhead = onnxruntime.InferenceSession(
        path_or_bytes=f'dmhead_Nx3x224x224.onnx',
        providers=[
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
    )
    dmhead_input_name = dmhead.get_inputs()[0].name
    dmhead_H = dmhead.get_inputs()[0].shape[2]
    dmhead_W = dmhead.get_inputs()[0].shape[3]

    cap_width = int(args.height_width.split('x')[1])
    cap_height = int(args.height_width.split('x')[0])
    if args.device.isdecimal():
        cap = cv2.VideoCapture(int(args.device))
    else:
        cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    WINDOWS_NAME = 'Demo'
    cv2.namedWindow(WINDOWS_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOWS_NAME, cap_width, cap_height)

    while True:
        start = time.time()

        ret, frame = cap.read()
        if not ret:
            continue

        # ============================================================= YOLOv4
        # Resize
        resized_frame = cv2.resize(frame, (yolov4_head_W, yolov4_head_H))
        # BGR to RGB
        rgb = resized_frame[..., ::-1]
        # HWC -> CHW
        chw = rgb.transpose(2, 0, 1)
        # normalize to [0, 1] interval
        chw = np.asarray(chw / 255., dtype=np.float32)
        # hwc --> nhwc
        nchw = chw[np.newaxis, ...]
        # Inference YOLOv4
        heads = yolov4_head.run(
            None,
            input_feed = {yolov4_head_input_name: nchw}
        )[0]

        canvas = resized_frame.copy()
        # ============================================================= DMHead
        croped_resized_frame = None
        scores = heads[:,4]
        keep_idxs = scores > 0.6
        heads = heads[keep_idxs, :]

        if len(heads) > 0:
            dmhead_inputs = []
            heads[:, 0] = heads[:, 0] * cap_width
            heads[:, 1] = heads[:, 1] * cap_height
            heads[:, 2] = heads[:, 2] * cap_width
            heads[:, 3] = heads[:, 3] * cap_height

            for head in heads:
                x_min = int(head[0])
                y_min = int(head[1])
                x_max = int(head[2])
                y_max = int(head[3])

                # enlarge the bbox to include more background margin
                y_min = max(0, y_min - abs(y_min - y_max) / 10)
                y_max = min(resized_frame.shape[0], y_max + abs(y_min - y_max) / 10)
                x_min = max(0, x_min - abs(x_min - x_max) / 5)
                x_max = min(resized_frame.shape[1], x_max + abs(x_min - x_max) / 5)
                x_max = min(x_max, resized_frame.shape[1])
                croped_frame = resized_frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                # h,w -> 224,224
                croped_resized_frame = cv2.resize(croped_frame, (dmhead_W, dmhead_H))
                # bgr --> rgb
                rgb = croped_resized_frame[..., ::-1]
                # hwc --> chw
                chw = rgb.transpose(2, 0, 1)
                dmhead_inputs.append(chw)
            # chw --> nchw
            nchw = np.asarray(dmhead_inputs, dtype=np.float32)

            yaw = 0.0
            pitch = 0.0
            roll = 0.0
            # Inference DMHead
            outputs = dmhead.run(
                None,
                input_feed = {dmhead_input_name: nchw}
            )[0]

            for yaw, roll, pitch in outputs:
                yaw, pitch, roll = np.squeeze([yaw, pitch, roll])
                print(f'yaw: {yaw}, pitch: {pitch}, roll: {roll}')

                # BBox draw
                deg_norm = 1.0 - abs(yaw / 180)
                blue = int(255 * deg_norm)
                cv2.rectangle(
                    canvas,
                    (int(x_min), int(y_min)),
                    (int(x_max), int(y_max)),
                    color=(blue, 0, 255-blue),
                    thickness=2
                )

                if abs(yaw) != 0.0:
                    # Draw
                    draw_axis(
                        canvas,
                        yaw,
                        pitch,
                        roll,
                        tdx=(x_min+x_max)/2,
                        tdy=(y_min+y_max)/2,
                        size=abs(x_max-x_min)//2
                    )
                    cv2.putText(
                        canvas,
                        f'yaw: {np.round(yaw)}',
                        (int(x_min), int(y_min)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (100, 255, 0),
                        1
                    )
                    cv2.putText(
                        canvas,
                        f'pitch: {np.round(pitch)}',
                        (int(x_min), int(y_min) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (100, 255, 0),
                        1
                    )
                    cv2.putText(
                        canvas,
                        f'roll: {np.round(roll)}',
                        (int(x_min), int(y_min)-30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (100, 255, 0),
                        1
                    )

            time_txt = f'{(time.time()-start)*1000:.2f} ms'
            cv2.putText(
                canvas,
                time_txt,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                time_txt,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

        cv2.imshow(WINDOWS_NAME, canvas)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default='0',
        help='Path of the mp4 file or device number of the USB camera. Default: 0',
    )
    parser.add_argument(
        "--height_width",
        type=str,
        default='480x640',
        help='{H}x{W}. Default: 480x640',
    )
    args = parser.parse_args()
    main(args)
