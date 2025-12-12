import click
from pathlib import Path
from src import logger
from src.pipeline.stage_01_download_dataset import DownloadDatasetPipeline
from src.pipeline.stage_02_unzip_dataset import UnzipDatasetPipeline
from src.pipeline.stage_03_ImageExtraction import ImageExtractionPipeline
from src.pipeline.stage_04_augmentation import AugmentationPipeline
from src.pipeline.stage_05_videoEncoding import VideoEncodingPipeline
from src.pipeline.stage_06_generateDatasetLabel import GenerateDatasetLabelPipeline
from src.pipeline.stage_07_training import TrainingPipeline
from src.pipeline.stage_08_generateCaption import GenerateCaptionPipeline
from src.pipeline.stage_09_refineGeneratedCaption import RefineGeneratedCaptionPipeline
from src.pipeline.stage_10_evaluation import EvaluationPipeline

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
@click.option(
    '--augmentation', 
    is_flag=True, 
    default=False, 
    help='Image Augmentation'
)
@click.option(
    '--video_encoding', 
    is_flag=True, 
    default=False, 
    help='Video feature extraction'
)
@click.option(
    '--generate_label', 
    is_flag=True, 
    default=False, 
    help='Generate Dataset label'
)
@click.option(
    '--training', 
    is_flag=True, 
    default=False, 
    help='training'
)
@click.option(
    '--train_csv',
    type=click.Path(exists=True, dir_okay=True, file_okay=True),
    default=None,
    help='Training csv file Path(optional)'
)
@click.option(
    '--val_csv',
    type=click.Path(exists=True, dir_okay=True, file_okay=True),
    default=None,
    help='Validation csv file Path(optional)'
)
@click.option(
    '--epochs',
    type=int,
    default=0,
    help='epochs'
)
@click.option(
    '--generate', 
    is_flag=True, 
    default=False, 
    help='Generate caption'
)
@click.option(
    '--refine', 
    is_flag=True, 
    default=False, 
    help='Generated text, rearrange the sentence.'
)
@click.option(
    '--evaluation', 
    is_flag=True, 
    default=False, 
    help='Evaluate the model'
)
@click.option(
    '--caption', 
    type=click.STRING,
    default=None, 
    help='Single caption.'
)
@click.option(
    '--folder', 
    is_flag=True, 
    default=False, 
    help='Giver path is folder or single file. If true that mean multiple file folder else single file or single folder'
)
@click.option(
    '--source',
    type=click.Path(exists=True, dir_okay=True, file_okay=True),
    default=None,
    help='Path of the root folder where all the feature or image file in there/ single file <video, feature path, image folder> / in evaluation json path'
)
@click.option(
    '--destination',
    type=click.Path(exists=False, dir_okay=True, file_okay=True),
    default=None,
    help='Where to save: <directory / file>'
)
@click.option(
    '--model_path',
    type=click.Path(exists=True, dir_okay=True),
    default=None,
    help='saved model path'
)
@click.option(
    '--process_type',
    type=click.STRING,
    default='cloud', 
    help='Will be use local deepseek or API of deepseek; valude: cloud/local'
)
@click.option(
    '--reference_column',
    type=click.STRING,
    default=None, 
    help='Ground Truth caption columns'
)
@click.option(
    '--generated_column',
    type=click.STRING,
    default=None, 
    help='Which columns we will compare with ground Truth columns (caption columns)'
)
def main(
    download: bool,
    unzip: bool,
    im_ex:bool,
    imex_resume: bool,
    augmentation: bool,
    video_encoding: bool,
    generate_label: bool,
    training: bool,
    train_csv: Path,
    val_csv: Path,
    epochs: int,
    generate: bool,
    refine: bool,
    evaluation: bool,
    caption: str,
    folder: bool,
    source: Path,
    destination: Path,
    model_path: Path,
    process_type: str,
    reference_column: str,
    generated_column: str
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

    # Augmentation
    if augmentation:
        STAGE_NAME = "Augmentation"
        logger.info(f">>> stage {STAGE_NAME} started")
        pipeline = AugmentationPipeline()
        pipeline.run()
        logger.info(f">>> stage {STAGE_NAME} completed.")

    # Video Encoding
    if video_encoding:
        STAGE_NAME = "Video Encoding"
        logger.info(f">>> stage {STAGE_NAME} started")
        pipeline = VideoEncodingPipeline()
        pipeline.run()
        logger.info(f">>> stage {STAGE_NAME} completed.")

    # Generate Dataset Label
    if generate_label:
        STAGE_NAME = "Generate Dataset Label"
        logger.info(f">>> stage {STAGE_NAME} started")
        pipeline = GenerateDatasetLabelPipeline()
        pipeline.run()
        logger.info(f">>> stage {STAGE_NAME} completed.")
    
    # Training
    if training:
        STAGE_NAME = "Training"
        logger.info(f">>> stage {STAGE_NAME} started")
        pipeline = TrainingPipeline()
        pipeline.run(train_csv, val_csv, epochs)
        logger.info(f">>> stage {STAGE_NAME} completed.")

    # Generating Caption
    if generate:
        STAGE_NAME = "Generating Caption"
        logger.info(f">>> stage {STAGE_NAME} started")
        pipeline = GenerateCaptionPipeline()
        if folder:
            pipeline.run(folder_path=source, model_path=model_path, generated_text_save_dir_path=destination)
        # Single Caption generate
        else:
            pipeline.run(folder_path=source, model_path=model_path, generated_text_save_dir_path=destination, is_signle_path=True)
        logger.info(f">>> stage {STAGE_NAME} completed.")

    # Refine Sentence / Caption
    if refine:
        STAGE_NAME = "Refine the sentecne Caption"
        logger.info(f">>> stage {STAGE_NAME} started")
        pipeline = RefineGeneratedCaptionPipeline()
        pipeline.run(
            source=source,
            destination=destination,
            process_type=process_type,
            caption = caption
        )
        logger.info(f">>> stage {STAGE_NAME} completed.")

    # Evaluation
    if evaluation:
        STAGE_NAME = "Evaluation"
        logger.info(f">>> stage {STAGE_NAME} started")
        pipeline = EvaluationPipeline()
        pipeline.run(
            model_path = model_path,
            reference_json_path = train_csv or val_csv,
            reference_column =reference_column,
            generated_json_path = source,
            generated_column =generated_column,
            save_path = destination
        )
        logger.info(f">>> stage {STAGE_NAME} completed.")

if __name__ == '__main__':
    main()