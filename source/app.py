from sperm_tracking_ui import main_ui

import gradio as gr

if __name__ == '__main__':
    UI = main_ui()
    UI.launch(share=True)

# if __name__ == '__main__':
    # from sperm_tracking_ui import processor
    # video_url = 'data/data_25s.avi'
    # print(processor(video_url))