import gc
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # fix path

import argparse
# from pytorch_lightning import seed_everything
from transformers import set_seed
from source import wandb_config
import pytorch_lightning as pl
from source.helper import log_params, log_stdout
from source.model import FineTunerModel
from source.resources import get_experiment_dir, Dataset, get_last_experiment_dir, EXP_DIR
from source.preprocessor import Preprocessor
import logging
import wandb, os
from pytorch_lightning.loggers import WandbLogger

logger = logging.getLogger(__name__)

class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))

def train(train_args):
    args = argparse.Namespace(**train_args)
    # seed_everything(args.seed)
    set_seed(args.seed)

    print(train_args)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        filename="checkpoint-{epoch}",
        monitor="val_loss",
        verbose=True,
        mode="min",
        save_top_k=3
    )

    wandb_logger = WandbLogger(wandb_config.WANDB_PROJECT_NAME)
    
    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        # early_stop_callback=False,
        precision=16 if args.fp_16 else 32,
        amp_level=args.opt_level,
        amp_backend='apex',
        # gradient_clip_val=args.max_grad_norm,
        callbacks=[LoggingCallback(), checkpoint_callback],
        num_sanity_val_steps=args.nb_sanity_val_steps,  # skip sanity check to save time for debugging purpose
        # progress_bar_refresh_rate=1,
        logger=wandb_logger,
        
    )

    print("Initialize model")
    model = FineTunerModel(**train_args)
    trainer = pl.Trainer(**train_params)
    print(" Training model")
    trainer.fit(model)

    # del model # clean up memory
    print("training finished")


def train_on(dataset, args_dict):
    
    args_dict['output_dir'] = get_experiment_dir(create_dir=True)
    log_params(args_dict["output_dir"] / "params.json", args_dict)
    
    os.environ['WANDB_API_KEY'] = wandb_config.WANDB_API_KEY
    os.environ['WANDB_MODE'] = wandb_config.WANDB_MODE
    wandb.init(project=wandb_config.WANDB_PROJECT_NAME, name=f"{args_dict['output_dir'].stem}_Train", job_type='Train', config=args_dict)
    

    preprocessor = Preprocessor(args_dict['features_kwargs'], args_dict['lang'])
    preprocessor.preprocess(dataset)
    
    args_dict["dataset"] = dataset
    with log_stdout(args_dict['output_dir'] / "logs.txt"):
        train(args_dict)
    
    gc.collect() # clean up memory
    wandb.finish()
    


