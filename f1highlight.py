import os
import subprocess
import yt_dlp
import cv2
import pytesseract
import tqdm
import re
from fractions import Fraction
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import time
import datetime

url = "https://www.youtube.com/watch?v=7ynDOY1PR74"

playlist_url = "https://www.youtube.com/watch?v=7ynDOY1PR74&list=PLfoNZDHitwjUv0pjTwlV1vzaE0r7UDVDR"


def download_video(url, output_file):
    ydl_opts = {'outtmpl': '%(id)s.%(ext)s',
                'format': 'bestvideo[height<=?1080]+bestaudio/best',
                }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    video_id = url.split('=')[-1]
    os.rename(f"{video_id}.webm", output_file)


def crop_video(input_file, output_file, x, y, width, height):
    command = f"ffmpeg -i {input_file} -filter_complex 'crop={width}:{height}:{x}:{y}' {output_file}"
    subprocess.call(command, shell=True)


def extract_text_from_video(video_path, output_file, skip_frames=30):
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Calculate the number of frames to process
    total_processed_frames = total_frames // skip_frames

    text_list = []

    with tqdm.tqdm(total=total_processed_frames, desc="Extracting text") as pbar:
        frame_count = 0
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            if frame_count % skip_frames == 0:
                # Convert frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Perform thresholding to enhance text contrast
                _, thresh = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # Perform text detection using pytesseract
                text = pytesseract.image_to_string(thresh)

                text_list.append(text)

                pbar.update(1)

    # Save the text list to a text file
    with open(output_file, 'w', encoding='utf-8') as f:
        for text_string in text_list:
            f.write(text_string + '\n')

    cap.release()
    cv2.destroyAllWindows()


def filter_lines(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            try:
                if line and re.match('^[0-9/]+$', line) and all(x and int(float(x)) <= 80 for x in line.split('/')):
                    f_out.write(line)
            except:
                pass


def sort_lines(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        lines = f_in.readlines()
        sorted_lines = sorted(lines)
        f_out.writelines(sorted_lines)


def textfile_to_nparray(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
        return np.array([float(line.strip()) for line in lines])


def decimalize_text_file(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            fraction = Fraction(line.strip())
            decimal = float(fraction)
            if 0 <= decimal <= 1:
                f_out.write(f'{decimal}\n')


def plot_gaussian_kde(input_file):
    data = textfile_to_nparray(input_file)

    # Kernel density estimation (KDE) using a Gaussian kernel
    kde = stats.gaussian_kde(data)

    # Generate a range of x values for the KDE plot
    x_grid = np.linspace(min(data), max(data), 100)

    #  valuate the KDE at the x_grid points
    density = kde(x_grid)

    # Plot the histogram and KDE
    plt.hist(data, bins=50, density=True, label='Histogram')
    plt.plot(x_grid, density, label='Kernel Density Estimation')
    plt.xlabel('Race Progress (%)')
    plt.ylabel('Occurancy in Race')
    plt.legend()
    plt.title('How often is every lap featured in a race')
    plt.show()


def download_playlist(playlist_url, video_func):
    ydl_opts = {}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Extract info without downloading
        result = ydl.extract_info(playlist_url, download=False)

        # Loop over each video in the playlist
        for video in result['entries']:
            url = video['webpage_url']
            download_video(url, "video.mp4")

            # Call the provided function with the downloaded video
            video_func("video.mp4")

            # Remove the downloaded video
            os.remove("video.mp4")


def append_text(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'a') as f_out:
        for line in f_in:
            f_out.write(line)


def multiply_lines(input_file, output_file, multiplier):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            f_out.write(str(float(line.strip()) * float(multiplier)) + '\n')


def process_video(video_file):
    print(f"Processing: {video_file}")
    # Process the video
    crop_video(video_file, "highlight_cropped.mp4", 185, 109, 90, 30)
    extract_text_from_video("highlight_cropped.mp4", "text.txt", 30)
    filter_lines("text.txt", "filtered.txt")
    decimalize_text_file("filtered.txt", "sorted_decimal.txt")
    append_text("sorted_decimal.txt", "total.txt")
    os.remove("highlight_cropped.mp4")
    os.remove("text.txt")
    os.remove("filtered.txt")
    os.remove("sorted_decimal.txt")


start_time = time.time()

download_playlist(playlist_url, process_video)

sort_lines("total.txt", "sorted.txt")
multiply_lines("sorted.txt", "total.txt", 100)
os.remove("sorted.txt")
plot_gaussian_kde("total.txt")

print("--- %s seconds ---" % (time.time() - start_time))
