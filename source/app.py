from sperm_tracking_ui import main_ui

if __name__ == '__main__':
    UI = main_ui()
    UI.launch(share=True)

# if __name__ == '__main__':
#     from sperm_tracking_ui import processor
#     video_url = 'data/sperm-ex.mp4'
#     video_url = 'data/sperm-ex (online-video-cutter.com).mp4'
#     print(processor(video_url))