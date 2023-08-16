import argparse
import os
from datasets import load_dataset
import glob
import shutil 
import logging

logger = logging.getLogger(__name__)

def download_fleurs(download_dir,locale):  
    if not os.path.exists(download_dir):
        os.mkdir(download_dir)
    cache_dir = "cache/"
    try: 
        logger.log(logging.INFO, f"start downloading {locale}")
        fleurs_asr = load_dataset("google/xtreme_s", f"fleurs.{locale}", cache_dir=cache_dir )
        save_dir= os.path.join(cache_dir,"downloads","extracted")
        # tmp_file = glob.glob(f"{save_dir}/*")
        tmp_file = [name for name in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, name))]
        dest = shutil.move(os.path.join(save_dir,tmp_file[0],locale),download_dir) 
        logger.log(logging.INFO, f"Complete downloading {locale}")
    except:
        raise RuntimeError(f"Could not download locale: {locale}")
    finally:
        shutil.rmtree(cache_dir)

        
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='download commonvoice 13.')
    parser.add_argument('data_path', type=str,
                        help='folder to save the downloaded data')
    parser.add_argument('--locales', nargs='+',
                        help='languages to downlaod')

    args = parser.parse_args()
    for language in args.locales:
        download_fleurs(args.data_path,language)
