"""
Utility script to download flood detection datasets from various sources
"""

import os
import subprocess
import argparse
from pathlib import Path
import urllib.request
import zipfile
import tarfile
import shutil


def download_file(url: str, output_path: str):
    """
    Download file from URL with progress bar
    
    Args:
        url: Download URL
        output_path: Output file path
    """
    print(f"Downloading from {url}...")
    
    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\rProgress: {percent}%", end='')
    
    urllib.request.urlretrieve(url, output_path, reporthook)
    print("\n✅ Download complete")


def extract_archive(archive_path: str, output_dir: str):
    """
    Extract zip or tar archive
    
    Args:
        archive_path: Path to archive file
        output_dir: Output directory
    """
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting {archive_path.name}...")
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(output_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
    
    print(f"✅ Extracted to {output_dir}")


def download_floodnet(output_dir: str = "data/raw/floodnet"):
    """
    Download FloodNet dataset from GitHub
    
    Args:
        output_dir: Output directory
    """
    print("=" * 60)
    print("Downloading FloodNet Dataset")
    print("=" * 60)
    
    repo_url = "https://github.com/BinaLab/FloodNet.git"
    output_path = Path(output_dir)
    
    if output_path.exists():
        print(f"Directory {output_dir} already exists. Skipping download.")
        return
    
    print(f"Cloning repository from {repo_url}...")
    
    try:
        subprocess.run(['git', 'clone', repo_url, str(output_path)], check=True)
        print(f"✅ FloodNet dataset downloaded to {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error cloning repository: {e}")
        print("Make sure git is installed: brew install git (macOS) or apt install git (Linux)")


def download_kaggle_dataset(dataset_name: str, output_dir: str):
    """
    Download dataset from Kaggle using Kaggle API
    
    Args:
        dataset_name: Kaggle dataset name (e.g., 'ratthachat/ai4floods')
        output_dir: Output directory
    """
    print("=" * 60)
    print(f"Downloading Kaggle Dataset: {dataset_name}")
    print("=" * 60)
    
    # Check if kaggle is installed
    try:
        import kaggle
    except ImportError:
        print("❌ Kaggle API not installed")
        print("Install with: pip install kaggle")
        print("\nSetup instructions:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Move kaggle.json to ~/.kaggle/")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download dataset
        print(f"Downloading to {output_dir}...")
        subprocess.run([
            'kaggle', 'datasets', 'download', 
            '-d', dataset_name, 
            '-p', str(output_path),
            '--unzip'
        ], check=True)
        
        print(f"✅ Dataset downloaded to {output_dir}")
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\nMake sure:")
        print("1. Kaggle API is properly configured")
        print("2. You have accepted the dataset's terms on Kaggle website")


def download_sample_cctv_videos(output_dir: str = "data/raw/videos"):
    """
    Download sample CCTV videos for testing
    
    Args:
        output_dir: Output directory
    """
    print("=" * 60)
    print("Sample CCTV Video Sources")
    print("=" * 60)
    
    print("\nYou can find sample flood/waterlogging videos from:")
    print("\n1. Pexels (Free stock videos)")
    print("   URL: https://www.pexels.com/search/videos/flood/")
    print("   License: Free to use")
    
    print("\n2. Pixabay (Free stock videos)")
    print("   URL: https://pixabay.com/videos/search/flood/")
    print("   License: Free to use")
    
    print("\n3. YouTube (with proper attribution)")
    print("   Search: 'urban flooding', 'waterlogged roads', 'street flooding'")
    print("   Use: youtube-dl or yt-dlp to download")
    
    print("\n4. Government/Municipal CCTV Archives")
    print("   Check local municipality websites for public CCTV footage")
    
    print(f"\n📁 Save videos to: {output_dir}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Download flood detection datasets"
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['floodnet', 'ai4floods', 'all', 'videos'],
        default='all',
        help='Dataset to download'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)
    
    if args.dataset in ['floodnet', 'all']:
        download_floodnet(str(output_base / 'floodnet'))
    
    if args.dataset in ['ai4floods', 'all']:
        download_kaggle_dataset('ratthachat/ai4floods', str(output_base / 'ai4floods'))
    
    if args.dataset in ['videos', 'all']:
        download_sample_cctv_videos(str(output_base / 'videos'))
    
    print("\n" + "=" * 60)
    print("Dataset Download Complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Verify downloaded data in: {output_base}")
    print(f"2. Preprocess data:")
    print(f"   python utils/prepare_dataset.py --input {output_base}/floodnet --output data/processed")
    print(f"3. Split into train/val/test:")
    print(f"   python utils/prepare_dataset.py --input data/processed --output data --split")


if __name__ == "__main__":
    main()
