import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from src.utils.commons import load_json_data
from src.entity.config_entity import ImageExtractionSplitConfig
from concurrent.futures import ThreadPoolExecutor, as_completed

class ImageExtractionSplit:
    def __init__(self, config, resume: bool = True):
        self.config = config
        self.resume = resume
        if not self.resume:
            if self.config.image_destination_dir.exists():
                import shutil
                shutil.rmtree(self.config.image_destination_dir)


    def resize_with_padding(self, image):
        # Get the original dimensions
        original_height, original_width = image.shape[:2]

        # Calculate the scaling factor while maintaining aspect ratio
        scale = min(self.config.IMAGE_SIZE / original_width, self.config.IMAGE_SIZE / original_height)

        # Compute the new size while maintaining the aspect ratio
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # Resize the image
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Create a black canvas of the target size
        canvas = np.zeros((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, 3), dtype=np.uint8)

        # Calculate the top-left corner position to center the image on the canvas
        x_offset = (self.config.IMAGE_SIZE - new_width) // 2
        y_offset = (self.config.IMAGE_SIZE - new_height) // 2

        # Place the resized image onto the black canvas
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

        return canvas

    def frames_extraction(self, video_path):
        '''
        This function will extract the required frames from a video after resizing and normalizing them.
        Args:
            video_path: The path of the video in the disk, whose frames are to be extracted.
        Returns:
            frames_list: A list containing the resized.
        '''
        # Declare a list to store video frames.
        frames_list = []

        # Read the Video File using the VideoCapture object.
        video_reader = cv2.VideoCapture(video_path)

        # Get the total number of frames in the video.
        video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate the interval after which frames will be added to the list.
        skip_frames_window = video_frames_count / self.config.FRAMES_PER_VIDEO

        # Iterate through the Video Frames.
        for frame_counter in range(self.config.FRAMES_PER_VIDEO):

            # Set the current frame position of the video.
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, np.floor(frame_counter * skip_frames_window))

            # Reading the frame from the video.
            success, frame = video_reader.read()

            # Check if Video frame is not successfully read then break the loop
            if not success:
                break

            # Resize the Frame to fixed height and width.
            resized_frame = self.resize_with_padding(frame)

            # # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
            # normalized_frame = resized_frame / 255.0

            # Append the normalized frame into the frames list
            frames_list.append(resized_frame)

        # Release the VideoCapture object.
        video_reader.release()

        # Return the frames list.
        return frames_list

    def process_video(self, video_path):
        '''
        This function processes a single video by extracting frames and saving them to the destination directory.
        '''
        # get file name without extension
        file_name = video_path.stem

        # get the destination directory for the images
        dest_dir = self.config.image_destination_dir / self.video_label_dict.get(file_name, 'train') / file_name

        # Skip processing if frames already exist
        if (dest_dir / f"{self.config.FRAMES_PER_VIDEO-1:0>3}.{self.config.image_format}").exists():
            return
        
        dest_dir.mkdir(parents=True, exist_ok=True)

        frames = self.frames_extraction(video_path)
        for i, frame in enumerate(frames):
            cv2.imwrite(dest_dir / f"{i:0>3}.{self.config.image_format}", frame.astype('uint8'))

    def set_label(self):
        self.video_label_dict = {}
        for json_file_path, label in self.config.caption_details:
            json_data = load_json_data(json_file_path)
            for item in json_data:
                self.video_label_dict[item['video_id']] = label

    def is_imageAlreadyExtracted(self, video_path):
        dest_dir = self.config.image_destination_dir / self.video_label_dict.get(video_path.stem, 'train') / video_path.stem
        return (dest_dir / f"{self.config.FRAMES_PER_VIDEO-1:0>3}.{self.config.image_format}").exists()

    def run(self):
        self.set_label()
        video_paths = sorted(self.config.video_source_dir.glob("*"))
        tasks = []
        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            for video_path in video_paths:
                if self.is_imageAlreadyExtracted(video_path):
                    continue
                tasks.append(
                    executor.submit(self.process_video, video_path)
                )

            # Display progress with tqdm
            for future in tqdm(as_completed(tasks), total=len(tasks), desc="Processing videos"):
                # Wait for task completion
                future.result()


if __name__ == "__main__":
    from src import logger
    from src.config.configuration import ConfigurationManager

    # Image Extraction
    STAGE_NAME = "Image Extraction"
    logger.info(f">>> stage {STAGE_NAME} started")
    config = ConfigurationManager().get_image_extraction_split_config()
    imageExtraction = ImageExtractionSplit(config)
    imageExtraction.run()
    logger.info(f">>> stage {STAGE_NAME} completed and save it to: {config.image_destination_dir}")
