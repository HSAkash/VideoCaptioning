import torch
import gc
from concurrent.futures import wait, FIRST_COMPLETED
from pathlib import Path
from tqdm import tqdm
import torchvision.io as io
from torchvision.utils import save_image
import kornia.augmentation as K
from src.utils.interrupt_check import DelayedInterrupt
from src.entity.config_entity import AugmentationConfig
from concurrent.futures import ThreadPoolExecutor, as_completed


class Augmentation:
    def __init__(self, config: AugmentationConfig):
        self.config = config

        self.aug = torch.nn.Sequential(
            K.RandomResizedCrop(
                (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
                scale=self.config.resize_crop_scale,
                same_on_batch=True
            ),
            K.RandomHorizontalFlip(
                p=self.config.horizontal_flip_p,
                same_on_batch=True
            ),
            K.ColorJitter(
                self.config.color_jitter[0], # brightness
                self.config.color_jitter[1], # contrast
                self.config.color_jitter[2], # saturation
                self.config.color_jitter[3], # hue
                same_on_batch=True
            ),
            K.RandomGaussianBlur(
                (3, 3),
                self.config.gaussian_blur_sigma,
                p=1.0,
                same_on_batch=True
            ),
        )

    def augment_clip(self, video_folder, dest_dir):
        # Collect all image file names
        image_files = sorted(video_folder.glob('*'))

        # Read each image and stack into [T,C,H,W]
        frames = []
        for image_path in image_files:
            img = io.read_image(image_path).float() / 255.0  # [C,H,W], [0,1]
            frames.append(img)
        frames = torch.stack(frames, dim=0)  # [T,C,H,W]

        # Apply augmentation (same for all frames)
        aug_frames = self.aug(frames)
        
        # Save augmented frames
        with DelayedInterrupt():
            dest_dir.mkdir(parents=True, exist_ok=True)
            for i, fname in enumerate(image_files):
                save_path = dest_dir / fname.name
                save_image(aug_frames[i], save_path)
        del frames
        del aug_frames
        torch.cuda.empty_cache()  # if using GPU
        gc.collect()

    def run(self):
        video_folders = []
        for source_root_folder, destination_root_folder in self.config.source_destination_dirs:
            video_paths = sorted(source_root_folder.glob("*"))
            ignore_folders = sorted(source_root_folder.glob(f"*___[0-{self.config.N-1}][0-f{self.config.N-1}]"))
            for video_path in video_paths:
                if video_path in ignore_folders:
                    continue
                video_folders.append((video_path, destination_root_folder/video_path.stem))

        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            futures = set()
            video_iter = iter(video_folders)
            total_tasks = len(video_folders) * self.config.N
            with tqdm(total=total_tasks, desc="Video Augmentation") as pbar:
                while True:
                    # Fill up to max_workers
                    while len(futures) < self.config.MAX_WORKERS:
                        try:
                            video_path, destination_video_path = next(video_iter)
                        except StopIteration:
                            break
                        for i in range(self.config.N):
                            dest_dir =  Path(f"{destination_video_path}___{i:0>2}")
                            if (dest_dir / f"{self.config.FRAMES_PER_VIDEO-1:0>3}.{self.config.image_format}").exists():
                                pbar.update(1)
                                continue
                            f = executor.submit(self.augment_clip, video_path, dest_dir)
                            futures.add(f)

                    if not futures:
                        break

                    # Wait for one to finish
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    for f in done:
                        f.result()  # release memory
                        pbar.update(1)


if __name__ == "__main__":
    from src import logger
    from src.config.configuration import ConfigurationManager

    # Augmentation
    STAGE_NAME = "Augmentation"
    logger.info(f">>> stage {STAGE_NAME} started")
    config = ConfigurationManager().get_augmentation_config()
    augmentation = Augmentation(config)
    augmentation.run()
    logger.info(f">>> stage {STAGE_NAME} completed.")
