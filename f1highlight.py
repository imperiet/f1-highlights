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

url = "https://www.youtube.com/watch?v=7ynDOY1PR74"


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


def save_still_frame(frameIndex):
    vidcap = cv2.VideoCapture('highlight_cropped.mp4')
    success, image = vidcap.read()
    count = 0
    while success:
        if count == frameIndex:
            cv2.imwrite("singleframe.png", image)     # save frame as PNG file
            break
        success, image = vidcap.read()
        count += 1


def extract_text_from_video(video_path, output_file, skip_frames=30):
    """
    Extracts text from every nth frame of a video and saves it to a text file, displaying a loading bar.

    Args:
        video_path (str): Path to the video file.
        output_file (str): Path to the output text file.
        skip_frames (int): Number of frames to skip between processing.

    Returns:
        None
    """

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
            if re.match('^[0-9/]+$', line) and all(int(x) <= 80 for x in line.split('/')):
                f_out.write(line)


def sort_lines(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        lines = f_in.readlines()
        sorted_lines = sorted(lines)
        f_out.writelines(sorted_lines)


def fraction_to_float(string):
    num, denom = map(int, string.split('/'))
    return num / denom


def textfile_to_nparray(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
        return np.array([x for x in (fraction_to_float(line.strip()) for line in lines) if 0 <= x <= 1])


def plot_gaussian_kde(data):
    data = textfile_to_nparray("sorted.txt")

    # Kernel density estimation (KDE) using a Gaussian kernel
    kde = stats.gaussian_kde(data)

    # Generate a range of x values for the KDE plot
    x_grid = np.linspace(min(data), max(data), 100)

    #  valuate the KDE at the x_grid points
    density = kde(x_grid)

    # Plot the histogram and KDE
    plt.hist(data, bins=20, density=True, label='Histogram')
    plt.plot(x_grid, density, label='KDE')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Kernel Density Estimation')
    plt.show()


download_video(url, "highlight.mp4")
crop_video("highlight.mp4", "highlight_cropped.mp4", 185, 109, 90, 30)
extract_text_from_video("highlight_cropped.mp4", "text.txt", 30)
filter_lines("text.txt", "filtered.txt")
sort_lines("filtered.txt", "sorted.txt")
plot_gaussian_kde("sorted.txt")
