import cv2

cap = cv2.VideoCapture('sperm-ex.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(3)) # float `width`
frame_height = int(cap.get(4))
out = cv2.VideoWriter('data_25s.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, frameSize=(frame_width, frame_height))

counter = 0
print('FPS = ', fps)

while 10:
    ret, frame = cap.read()
    counter += 1

    if not ret:
        break

    out.write(frame)

    if counter > fps * 3:
        break
