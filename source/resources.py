import os
from pathlib import Path
import tarfile
import time
import tempfile
import zipfile
import tqdm
import urllib3
import gzip, shutil
from enum import Enum

REPO_DIR = Path(__file__).resolve().parent.parent
RESOURCES_DIR = REPO_DIR / 'resources'
EXP_DIR = REPO_DIR / 'experiments'
DATASETS_DIR = RESOURCES_DIR / 'datasets'
PROCESSED_DATA_DIR = RESOURCES_DIR / "processed_data"
DUMPS_DIR = RESOURCES_DIR / "dumps"
DUMPS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = RESOURCES_DIR / "cache"

class Phase:
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'
    
class Dataset:
    LexMTurk = 'lex.mturk'
    NNSeval = 'NNSeval'
    BenchLS = 'BenchLS'
    SemEval2012 = 'semeval2012'
    TSAR_EN = 'tsar_en'
    TSAR_EN_TESTSET = 'tsar_en_test'
    ALEXSIS = 'ALEXSIS_v1.0'
    
class Language:
    EN = 'en'
    ES = 'es'
    PT = 'pt'
    FR = 'fr'
    DE = 'de'

class Train_Type:
    DIFF = 0 # Train and Test different data
    WHOLE = 1  # Train and Test with the same data


def get_dataset_filepath(dataset):
    return DATASETS_DIR / f'{dataset}.tsv'


def get_temp_filepath(create=False):
    temp_filepath = Path(tempfile.mkstemp()[1])
    if not create:
        temp_filepath.unlink()
    return temp_filepath

def get_experiment_dir(create_dir=False):
    dir_name = f'{int(time.time() * 1000000)}'
    path = EXP_DIR / f'exp_{dir_name}'
    if create_dir : path.mkdir(parents=True, exist_ok=True)
    return path


def get_tuning_log_dir():
    log_dir = EXP_DIR / 'tuning_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

def get_last_experiment_dir():
    return sorted(list(EXP_DIR.glob('exp_*')), reverse=True)[0]

    
def download_fasttext_embedding(lang):
    
    dest_dir = Path(tempfile.gettempdir())
    filename = f'cc.{lang}.300.vec'
    filepath = dest_dir / filename
    if filepath.exists(): return filepath

    url = f'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/{filename}.gz'
    downlaod_filepath = download_url(url, dest_dir)
    print("Extracting: ", downlaod_filepath.name)
    with gzip.open(downlaod_filepath, 'rb') as f:
            with open(filepath, 'wb') as f_out:
                shutil.copyfileobj(f, f_out)
    downlaod_filepath.unlink()
    return filepath 

def download_report_hook(t):
  last_b = [0]
  def inner(b=1, bsize=1, tsize=None):
    if tsize is not None:
        t.total = tsize
    t.update((b - last_b[0]) * bsize)
    last_b[0] = b
  return inner


def download_url(url, output_path):
    name = url.split('/')[-1]
    file_path = Path(output_path) / name
    if not file_path.exists():
        with tqdm(unit='B', unit_scale=True, leave=True, miniters=1, desc=name) as t: 
            urllib3.request.urlretrieve(url, filename=file_path, reporthook=download_report_hook(t), data=None)
    return file_path


def unzip(file_path, dest_dir=None):
    file_path = str(file_path)
    if dest_dir is None:
        dest_dir = os.path.dirname(file_path)
    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(dest_dir)
    elif file_path.endswith("tar.gz") or file_path.endswith("tgz"):
        tar = tarfile.open(file_path, "r:gz")
        tar.extractall(dest_dir)
        tar.close()
    elif file_path.endswith(".gz"):
        tofile = file_path.replace('.gz', '')
        with open(file_path, 'rb') as inf, open(tofile, 'wb') as tof:
            decom_str = gzip.decompress(inf.read())
            tof.write(decom_str)
    elif file_path.endswith("tar"):
        tar = tarfile.open(file_path, "r:")
        tar.extractall(dest_dir)
        tar.close()


if __name__ == '__main__':
    print(get_temp_filepath())
