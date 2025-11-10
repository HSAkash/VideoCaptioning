import click
from src import logger
from src.pipeline.stage_01_download_dataset import DownloadDatasetPipeline
from src.pipeline.stage_02_unzip_dataset import UnzipDatasetPipeline
from src.pipeline.stage_03_ImageExtraction import ImageExtractionPipeline

@click.command()
@click.option(
    '--download', 
    is_flag=True, 
    default=False, 
    help='Download dataset.'
)
@click.option(
    '--unzip', 
    is_flag=True, 
    default=False, 
    help='Unzip the download zip data.'
)
@click.option(
    '--im_ex', 
    is_flag=True, 
    default=False, 
    help='Video to image extraction.'
)
@click.option(
    '--imex_resume', 
    is_flag=True, 
    default=False, 
    help='Not Resume the image extraction process. It will delete the previous images & start from the beginning'
)

def main(
    download: bool,
    unzip: bool,
    im_ex:bool,
    imex_resume: bool
):
    # Download dataset
    if download:
        STAGE_NAME = "Download Dataset"
        logger.info(f">>> stage {STAGE_NAME} started")
        pipeline = DownloadDatasetPipeline()
        pipeline.run()
        logger.info(f">>> stage {STAGE_NAME} completed.")

    # Unzip dataset
    if unzip:
        STAGE_NAME = "Unzip dataset"
        logger.info(f">>> stage {STAGE_NAME} started")
        pipeline = UnzipDatasetPipeline()
        pipeline.run()
        logger.info(f">>> stage {STAGE_NAME} completed.")
        logger.info(f">>> stage {STAGE_NAME} completed.")

    # Image Extraction
    if im_ex:
        STAGE_NAME = "Image Extraction"
        logger.info(f">>> stage {STAGE_NAME} started")
        pipeline = ImageExtractionPipeline()
        pipeline.run(resume = not imex_resume)
        logger.info(f">>> stage {STAGE_NAME} completed.")

if __name__ == '__main__':
    main()