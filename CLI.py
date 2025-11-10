import click
from src import logger
from src.pipeline.stage_01_download_dataset import DownloadDatasetPipeline
from src.pipeline.stage_02_unzip_dataset import UnzipDatasetPipeline

@click.command()
@click.option(
    '--download', 
    is_flag=True, 
    default=False, 
    help='Skiping the download dataset.'
)
@click.option(
    '--unzip', 
    is_flag=True, 
    default=False, 
    help='Skiping the unzip part.'
)
def main(
    download: bool,
    unzip: bool
):
    # Download dataset
    if not download:
        STAGE_NAME = "Download Dataset"
        logger.info(f">>> stage {STAGE_NAME} started")
        Pipeline = DownloadDatasetPipeline()
        Pipeline.run()
        logger.info(f">>> stage {STAGE_NAME} completed.")

    # Unzip dataset
    if not unzip:
        STAGE_NAME = "Unzip dataset"
        logger.info(f">>> stage {STAGE_NAME} started")
        Pipeline = UnzipDatasetPipeline()
        Pipeline.run()
        logger.info(f">>> stage {STAGE_NAME} completed.")

if __name__ == '__main__':
    main()