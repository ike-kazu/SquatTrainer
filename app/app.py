import numpy as np
from flask import Flask, render_template, Response
from camera import Camera
import cv2

app = Flask(__name__)

isDone  =False

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/stream")
def stream():
    return render_template("stream.html")

def gen_video(camera, isVideo):
    """Video streaming generator function."""
    while True:
        success, image = camera.get_frame()
        if not success:
           break
        
        ret, frame = cv2.imencode(".jpg", image)
        """ if(isDone):
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame.tobytes() + b"\r\n"
            )
            #return 'OK!'
        else:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame.tobytes() + b"\r\n"
            )
            #return 'No!' """
        """ if(isVideo):
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame.tobytes() + b"\r\n"
            )
        else:
            yield (isDone) """

        yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame.tobytes() + b"\r\n"
        )

            

@app.route("/video_feed")
def video_feed():
    cam = Camera()
    return Response(
        gen_video(cam, True), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

""" @app.route("/isDone")
def isDone():
    cam = Camera()
    return Response(
        gen_video(cam, False)
    ) """


@app.route("/kill_camera")
def kill_camera():
    cam = Camera()
    del cam
    return render_template("camera_killed.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
