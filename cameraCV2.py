import cv2

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def get_csi_stream(req_width, req_height, req_framerate):
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0, capture_width=req_width, capture_height=req_height, framerate=req_framerate), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        return video_capture

def end_stream(video):
    video.release()
