import os
from secvcm.utils.video_reader import YUVReader
from secvcm.utils.video_writer import PNGWriter

def convert_one_seq_to_png(src_path, width, height, dst_path):
    src_reader = YUVReader(src_path, width, height, src_format='420')
    png_writer = PNGWriter(dst_path, width, height)
    rgb = src_reader.read_one_frame(dst_format='rgb')
    processed_frame = 0
    while not src_reader.eof:
        png_writer.write_one_frame(rgb=rgb, src_format='rgb')
        processed_frame += 1
        rgb = src_reader.read_one_frame(dst_format='rgb')
    print(f"Processed {processed_frame} frames from {src_path}")

def main():
    src_dir = "/path/to/source/yuv"
    width = 1920
    height = 1080
    dst_dir = "/path/to/output/png"

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for filename in os.listdir(src_dir):
        if filename.endswith('.yuv'):
            src_path = os.path.join(src_dir, filename)
            file_base_name = os.path.splitext(filename)[0]
            file_dst_dir = os.path.join(dst_dir, file_base_name)
            
            if not os.path.exists(file_dst_dir):
                os.makedirs(file_dst_dir)
                
            convert_one_seq_to_png(src_path, width, height, file_dst_dir)

if __name__ == "__main__":
    main()
