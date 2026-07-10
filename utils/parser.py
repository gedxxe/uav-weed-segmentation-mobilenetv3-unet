import argparse

from utils.model_registry import (
    ARCHITECTURE_CHOICES,
    BASELINE_ARCHITECTURES,
    ENCODER_CHOICES,
    PROPOSED_ENCODERS,
    PROPOSED_ARCHITECTURES,
    RESNET_ENCODERS,
)


def validate_train_args(parser, args):
    if args.architecture in BASELINE_ARCHITECTURES and args.encoder_name not in RESNET_ENCODERS:
        parser.error(
            f"architecture '{args.architecture}' requires one of these ResNet encoders: "
            f"{', '.join(RESNET_ENCODERS)}"
        )
    if args.architecture in PROPOSED_ARCHITECTURES and args.encoder_name not in PROPOSED_ENCODERS:
        parser.error(
            f"architecture '{args.architecture}' requires encoder_name='mobilenetv3_large'. "
            "Use the ResNet encoders only with baseline architectures."
        )
    if args.loss == "ce_dice_aux_foreground" and args.architecture != "unet_mobilenetv3_aux":
        parser.error(
            "--loss ce_dice_aux_foreground requires architecture 'unet_mobilenetv3_aux'."
        )


def create_train_parser():
    my_parser = argparse.ArgumentParser(
        description='Trains deep learning models for Weed/Crop Segmentation using kFold Cross validation and saves the results in a optuna database')

    my_parser.add_argument('--db_name',
                        metavar='db_name',
                        type=str,
                        help='Name of the optuna database', default="")

    my_parser.add_argument('--study_name',
                        metavar='study_name',
                        type=str,
                        help='Name of the optuna study', default="")

    my_parser.add_argument('architecture',
                        type=str,
                        choices=ARCHITECTURE_CHOICES,
                        help='String of an architecture, implemented: fcn8s, fcn16s, fcn32s, unet, dlplus, unet_mobilenetv3_base, unet_mobilenetv3_ppm, unet_mobilenetv3, unet_mobilenetv3_aux')

    my_parser.add_argument('encoder_name',
                        type=str,
                        choices=ENCODER_CHOICES,
                        help='String of an encoder. Baselines use resnet18/resnet34/resnet50/resnet101; proposed model uses mobilenetv3_large.')
    
    my_parser.add_argument('--run_prefix',
                        type=str,
                        help='Prefix for the optuna database', default="db")
    
    my_parser.add_argument('--save_checkpoint',
                        help='Bool, if True will save checkpoints of each epoch.',
                        action='store_true')

    my_parser.add_argument('--pretrained',
                        help='Bool, if True will use a pretrained feature extractor (on ImageNet)',
                        action=argparse.BooleanOptionalAction, default=True)
    
    my_parser.add_argument('--b_bilinear',
                        help='Bool, if True will use bilinear interpolation, if False will use transposed convolutions',
                        action=argparse.BooleanOptionalAction, default=True)

    my_parser.add_argument('--replace_stride_with_dilation',
                        help='Bool, if True will replace strides with dilated convolutions',
                        action='store_true')

    my_parser.add_argument('--b_clean_study',
                        help='Bool, if True will delete all Trials and start a fresh study',
                        action='store_true')

    my_parser.add_argument('--lr_max',
                        type=float,
                        help='Maximal learning rate to sample from', default=1e-2)

    my_parser.add_argument('--lr_min',
                        type=float,
                        help='Minimal learning rate to sample from', default=1e-4)

    my_parser.add_argument('--n_folds',
                        type=int,
                        help='Number of folds for kFold Cross validation', default=4)

    my_parser.add_argument('--batch_size',
                        type=int,
                        help='Number patches per batch', default=8)

    my_parser.add_argument('--n_trials',
                        type=int,
                        help='Number of trials per optuna study', default=1)

    my_parser.add_argument('--max_epochs',
                        type=int,
                        help='Maximum epochs per fold/trial', default=100)

    my_parser.add_argument('--early_stop_patience',
                        type=int,
                        help='Stop a fold after this many epochs without validation improvement', default=10)

    my_parser.add_argument('--lr_scheduler_patience',
                        type=int,
                        help='ReduceLROnPlateau patience in epochs', default=5)

    my_parser.add_argument('--loss',
                        type=str,
                        choices=["dice", "ce", "ce_dice", "ce_dice_aux_foreground"],
                        help='Training loss. dice preserves the original workflow; ce_dice is better for class confusion/imbalance.', default="dice")

    my_parser.add_argument('--class_weights',
                        type=str,
                        choices=["none", "auto"],
                        help='Class weighting for CE-based losses. auto estimates weights from the training masks in each fold.', default="none")

    my_parser.add_argument('--class_weight_max',
                        type=float,
                        help='Maximum clipping value for automatically estimated class weights.', default=5.0)

    my_parser.add_argument('--class_weight_strategy',
                        type=str,
                        choices=["inverse_frequency", "sqrt_inverse"],
                        help='Class-weight formula used when --class_weights auto is enabled.', default="inverse_frequency")

    my_parser.add_argument('--ce_weight',
                        type=float,
                        help='Cross-entropy term weight when --loss ce_dice is used.', default=1.0)

    my_parser.add_argument('--dice_weight',
                        type=float,
                        help='Dice term weight when --loss ce_dice is used.', default=1.0)

    my_parser.add_argument('--foreground_aux_weight',
                        type=float,
                        help='Auxiliary BG-vs-vegetation CE term weight when --loss ce_dice_aux_foreground is used.', default=0.3)

    my_parser.add_argument('--validation_loss',
                        type=str,
                        choices=["dice", "same", "macro_f1", "weed_f1", "foreground_macro_f1"],
                        help='Checkpoint selection objective. dice preserves the original workflow; same uses the configured training loss; macro_f1/weed_f1/foreground_macro_f1 minimize 1-F1 on validation predictions.', default="dice")

    my_parser.add_argument('--num_workers',
                        type=int,
                        help='PyTorch DataLoader workers. Use 0 if Windows multiprocessing causes issues.', default=2)

    my_parser.add_argument('--device',
                        type=str,
                        choices=["auto", "cuda", "cpu"],
                        help='Training device. auto uses CUDA when available.', default="auto")

    my_parser.add_argument('--amp',
                        help='Use CUDA automatic mixed precision. Ignored on CPU.',
                        action=argparse.BooleanOptionalAction, default=True)

    my_parser.add_argument('--pin_memory',
                        help='Use DataLoader pinned memory. Defaults to on for CUDA and off for CPU.',
                        action=argparse.BooleanOptionalAction, default=None)

    my_parser.add_argument('--seed',
                        type=int,
                        help='Seed of the experiment', default=42)

    my_parser.add_argument('--root_path',
                        type=str,
                        help='Path to root of the project. "data" needs to be subpath of this', default=".")

    args = my_parser.parse_args()
    validate_train_args(my_parser, args)
    return args

def create_test_parser():
    my_parser = argparse.ArgumentParser(
        description='Predicts segmentation masks on complete UAV captures')

    my_parser.add_argument('model',
                        metavar='model',
                        type=str,
                        help='Path to retrained model.pt file that is used to generate predictions.')

    my_parser.add_argument('subset',
                        metavar='subset',
                        type=str,
                        help='String of the subset folder in /data to predict images on.')

    my_parser.add_argument('--batch_size',
                        type=int,
                        help='Number patches per batch', default=20)

    my_parser.add_argument('--num_workers',
                        type=int,
                        help='PyTorch DataLoader workers. Prediction defaults to 0 for Windows-safe execution.', default=0)

    my_parser.add_argument('--device',
                        type=str,
                        choices=["auto", "cuda", "cpu"],
                        help='Prediction device. auto uses CUDA when available.', default="auto")

    my_parser.add_argument('--amp',
                        help='Use CUDA automatic mixed precision. Ignored on CPU.',
                        action=argparse.BooleanOptionalAction, default=True)

    my_parser.add_argument('--pin_memory',
                        help='Use DataLoader pinned memory. Defaults to on for CUDA and off for CPU.',
                        action=argparse.BooleanOptionalAction, default=None)

    my_parser.add_argument('--seed',
                        type=int,
                        help='Seed of the experiment', default=42)

    my_parser.add_argument('--root_path',
                        type=str,
                        help='Path to root of the project. "data" needs to be subpath of this', default=".")

    my_parser.add_argument('--model_name',
                        type=str,
                        help='Human-readable model name used in prediction reports. Defaults to the checkpoint filename.', default=None)

    my_parser.add_argument('--output_dir',
                        type=str,
                        help='Directory used to save prediction PNG files. Defaults to results/predictions/<subset>.', default=None)

    my_parser.add_argument('--report',
                        help='Generate a metrics and qualitative prediction report after saving predictions.',
                        action=argparse.BooleanOptionalAction, default=True)

    my_parser.add_argument('--report_dir',
                        type=str,
                        help='Directory used to save the generated report. Defaults to results/reports/<subset>/<model_name>.', default=None)

    my_parser.add_argument('--crop_size',
                        type=int,
                        help='Crop size in pixels for qualitative report examples.', default=400)

    my_parser.add_argument('--max_examples',
                        type=int,
                        help='Maximum number of representative qualitative examples in the report grid.', default=6)
    
    args = my_parser.parse_args()
    return args

def compare_studies_parser():
    my_parser = argparse.ArgumentParser(
        description='Compares different study databases in /results')

    my_parser.add_argument('--root_path',
                        type=str,
                        help='Path to root of the project. "data" needs to be subpath of this', default=".")
    
    args = my_parser.parse_args()
    return args


def generate_patches_parser():
    my_parser = argparse.ArgumentParser(
        description='Generates patches from UAV captures and their masks')

    my_parser.add_argument('--root_path',
                        type=str,
                        help='Path to root of the project. "data" needs to be subpath of this', default=".")
    
    args = my_parser.parse_args()
    return args


def create_compare_parser():
    my_parser = argparse.ArgumentParser(
        description='Compares Ground Truth with Predictions')

    my_parser.add_argument('--bbch',
                        type=str,
                        help='BBCH stage as string', default=None)
    
    my_parser.add_argument('subset',
                        metavar='subset',
                        type=str,
                        help='String of the subset folder in /data to predict images on.')

    args = my_parser.parse_args()
    return args
