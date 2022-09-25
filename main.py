import argparse
import ast
import glob
import os
import shutil

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import albumentations
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pretrainedmodels
import sklearn
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.nn import functional as F
from tqdm import tqdm

INPUT_FOLDER = "kaggle/input/"
TRAINING_FOLDS_CSV = "train_folds.csv"
TESTING_CSV = "test.csv"
PICKLE_FOLDER = 'image_pickles/'

CONSONANT_DIACRITIC = "consonant_diacritic"
VOWEL_DIACRITIC = "vowel_diacritic"
GRAPHEME_ROOT = "grapheme_root"
KFOLD = "kfold"
IMAGE_ID = "image_id"
COLUMNS = [GRAPHEME_ROOT, VOWEL_DIACRITIC, CONSONANT_DIACRITIC, IMAGE_ID, KFOLD]

RESNET = 'resnet34'
EFFNET = 'efficientnet-b3'
BASE_MODELS = [RESNET, EFFNET]
MODEL_MEAN = ast.literal_eval("(0.485, 0.456, 0.406)")
MODEL_STD = ast.literal_eval("(0.229, 0.224, 0.225)")

CUDA_VISIBLE_DEVICES = 1
IMG_HEIGHT = 137
IMG_WIDTH = 236
EPOCHS = 100
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 32


# def clean_directory():
#     print("cleaning input directory... \n")
#     files_to_delete = ['train_folds.csv', 'image_pickles']
#     for filename in tqdm(os.listdir(INPUT_FOLDER)):
#         if filename in files_to_delete:
#             file_path = os.path.join(INPUT_FOLDER, filename)
#             try:
#                 if os.path.isfile(file_path) or os.path.islink(file_path):
#                     print('deleting file %s' % file_path)
#                     os.unlink(file_path)
#                 elif os.path.isdir(file_path):
#                     print('deleting folder %s' % file_path)
#                     shutil.rmtree(file_path)
#             except Exception as e:
#                 print('Failed to delete %s. Reason: %s' % (file_path, e))
#     print("finished cleaning input directory!! \n")

def delete_if_exists(file_path):
    if os.path.exists(file_path):
        if os.path.isfile(file_path) or os.path.islink(file_path):
            print('deleting file %s' % file_path)
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            print('deleting folder %s' % file_path)
            shutil.rmtree(file_path)


def add_fold():
    print('reading train data ...')
    df = pd.read_csv(INPUT_FOLDER + "train.csv")
    print(df.head())
    print('total values: %s' % df.shape[0])
    df.loc[:, KFOLD] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    X = df.image_id.values
    y = df[[GRAPHEME_ROOT, VOWEL_DIACRITIC, CONSONANT_DIACRITIC]].values
    mskf = MultilabelStratifiedKFold(n_splits=5)

    print('assigning folds ...')
    for fold, (trn_, val_) in tqdm(enumerate(mskf.split(X, y))):
        df.loc[val_, KFOLD] = fold
    print('total values in each fold: \n%s' % df.kfold.value_counts())
    output_path = "%s%s" % (INPUT_FOLDER, TRAINING_FOLDS_CSV)
    delete_if_exists(output_path)
    print('writing folded data to %s...' % output_path)
    df.to_csv(output_path, index=False)
    print("finished writing!\n")


def pkl_imgs(type='train'):
    print("creating image pickles from parquet files...\n")
    if type == 'train':
        files = glob.glob("%strain_*.parquet" % INPUT_FOLDER)
    elif type == 'test':
        files = glob.glob("%stest_*.parquet" % INPUT_FOLDER)
    else:
        raise Exception(f"Invalid type {type}")
    if not os.path.exists(INPUT_FOLDER + PICKLE_FOLDER):
        os.mkdir(INPUT_FOLDER + PICKLE_FOLDER)
    for f in files:
        print("reading file %s" % f)
        df = pd.read_parquet(f)
        image_ids = df.image_id.values
        df = df.drop(IMAGE_ID, axis=1)
        image_array = df.values
        for j, image_id in tqdm(enumerate(image_ids), total=len(image_ids)):
            output_path = os.path.dirname(
                os.path.realpath(__file__)) + "/" + INPUT_FOLDER + f"{PICKLE_FOLDER}{image_id}.pkl"
            delete_if_exists(output_path)
            joblib.dump(image_array[j, :], output_path)
    print('Finished creating image pickles!!\n')


def preprocess_input_data():
    add_fold()
    pkl_imgs()


class ResNet34(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        if pretrained:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained=None)

        # based on input images
        self.l0 = nn.Linear(512, 168)
        self.l1 = nn.Linear(512, 11)
        self.l2 = nn.Linear(512, 7)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)

        return l0, l1, l2


class EfficientNetB3(nn.Module):
    def __init__(self, pretrained):
        super(EfficientNetB3, self).__init__()

        self.effNet = EfficientNet.from_name('efficientnet-b3')

        # based on input images
        self.fc_root = nn.Linear(in_features=512, out_features=168)
        self.fc_vowel = nn.Linear(in_features=512, out_features=11)
        self.fc_consonant = nn.Linear(in_features=512, out_features=7)

    def forward(self, X):
        output = self.effNet(X)
        output_root = self.fc_root(output)
        output_vowel = self.fc_vowel(output)
        output_consonant = self.fc_consonant(output)

        return output_root, output_vowel, output_consonant


class TrainDataset:
    def __init__(self, folds, img_height, img_width, mean, std):
        df = pd.read_csv(INPUT_FOLDER + TRAINING_FOLDS_CSV)
        df = df[COLUMNS]

        df = df[df.kfold.isin(folds)].reset_index(drop=True)
        self.image_ids = df.image_id.values
        self.grapheme_root = df.grapheme_root.values
        self.vowel_diacritic = df.vowel_diacritic.values
        self.consonant_diacritic = df.consonant_diacritic.values
        self.img_height = img_height
        self.img_width = img_width
        self.mean = mean
        self.std = std
        self.is_valid_train = len(folds) == 1

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        image_path = f"{INPUT_FOLDER}{PICKLE_FOLDER}{self.image_ids[item]}.pkl"
        image = joblib.load(image_path)
        image = process_image_arr(image, self.img_height, self.img_width, self.mean, self.std, type="train",
                                  is_valid_train=self.is_valid_train)
        return {
            "image": torch.tensor(image, dtype=torch.float),
            GRAPHEME_ROOT: torch.tensor(self.grapheme_root[item], dtype=torch.long),
            VOWEL_DIACRITIC: torch.tensor(self.vowel_diacritic[item], dtype=torch.long),
            CONSONANT_DIACRITIC: torch.tensor(self.consonant_diacritic[item], dtype=torch.long),
        }


def process_image_arr(image, img_height, img_width, mean, std, type="test", is_valid_train=False):
    if is_valid_train or type == 'test':
        aug = albumentations.Compose(
            [
                albumentations.Resize(img_height, img_width, always_apply=True),
                albumentations.Normalize(mean, std, always_apply=True),
            ]
        )
    else:
        aug = albumentations.Compose(
            [
                albumentations.Resize(img_height, img_width, always_apply=True),
                albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=5, p=0.9),
                albumentations.Normalize(mean, std, always_apply=True),
            ]
        )

    image = image.reshape(137, 236).astype(float)
    image = Image.fromarray(image).convert("RGB")
    image = aug(image=np.array(image))["image"]
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    return image


class TestDataset:
    def __init__(self, df, img_height, img_width, mean, std):
        self.image_ids = df.image_id.values
        self.img_arr = df.iloc[:, 1:].values
        self.img_height = img_height
        self.img_width = img_width
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        image = self.img_arr[item, :]
        img_id = self.image_ids[item]
        image = process_image_arr(image, self.img_height, self.img_width, self.mean, self.std, type="test",
                                  is_valid_train=False)

        return {
            "image": torch.tensor(image, dtype=torch.float),
            "image_id": img_id
        }


# source: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def macro_recall(pred_y, y):
    pred_labels = predicted_labels(pred_y)
    y = y.cpu().numpy()
    recall_consonant, recall_grapheme, recall_vowel = calc_recalls(pred_labels, y)
    final_score = calc_recall_score_from_recalls(recall_consonant, recall_grapheme, recall_vowel)
    print(
        f'recall: grapheme {recall_grapheme}, vowel {recall_vowel}, '
        f'consonant {recall_consonant}, 'f'total {final_score}, y {y.shape}')

    return final_score


def predicted_labels(pred_y):
    n_grapheme = 168
    n_vowel = 11
    n_consonant = 7
    pred_y = torch.split(pred_y, [n_grapheme, n_vowel, n_consonant], dim=1)
    pred_labels = [torch.argmax(py, dim=1).cpu().numpy() for py in pred_y]
    return pred_labels


def calc_recall_score_from_recalls(recall_consonant, recall_grapheme, recall_vowel):
    scores = [recall_grapheme, recall_vowel, recall_consonant]
    final_score = np.average(scores, weights=[2, 1, 1])
    return final_score


def calc_recalls(pred_labels, y):
    recall_grapheme = sklearn.metrics.recall_score(pred_labels[0], y[:, 0], average='macro')
    recall_vowel = sklearn.metrics.recall_score(pred_labels[1], y[:, 1], average='macro')
    recall_consonant = sklearn.metrics.recall_score(pred_labels[2], y[:, 2], average='macro')
    return recall_consonant, recall_grapheme, recall_vowel


def calc_loss(outputs, targets):
    out1, out2, out3 = outputs
    target1, target2, target3 = targets
    loss1, loss2, loss3 = calc_entropy_losses(out1, out2, out3, target1, target2, target3)
    return (loss1 + loss2 + loss3) / 3


def calc_entropy_losses(out1, out2, out3, target1, target2, target3):
    loss1 = nn.CrossEntropyLoss()(out1, target1)
    loss2 = nn.CrossEntropyLoss()(out2, target2)
    loss3 = nn.CrossEntropyLoss()(out3, target3)
    return loss1, loss2, loss3


def train(dataset, device, data_loader, model, optimizer):
    model.train()
    final_loss = 0
    counter = 0
    final_outputs = []
    final_targets = []
    for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset) / data_loader.batch_size)):
        counter = counter + 1
        consonant_diacritic, grapheme_root, image, vowel_diacritic = extract(d, device)

        optimizer.zero_grad()
        outputs = model(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        loss = calc_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        final_loss = process_output(final_loss, final_outputs, final_targets, loss, outputs, targets)

    final_outputs = torch.cat(final_outputs)
    final_targets = torch.cat(final_targets)
    macro_recall_score = macro_recall(final_outputs, final_targets)

    return final_loss / counter, macro_recall_score


def process_output(final_loss, final_outputs, final_targets, loss, outputs, targets):
    final_loss += loss
    o1, o2, o3 = outputs
    t1, t2, t3 = targets
    final_outputs.append(torch.cat((o1, o2, o3), dim=1))
    final_targets.append(torch.stack((t1, t2, t3), dim=1))
    return final_loss


def evaluate(dataset, device, data_loader, model):
    with torch.no_grad():
        model.eval()
        final_loss = 0
        counter = 0
        final_outputs = []
        final_targets = []
        for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset) / data_loader.batch_size)):
            counter = counter + 1
            consonant_diacritic, grapheme_root, image, vowel_diacritic = extract(d, device)
            outputs = model(image)
            targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
            loss = calc_loss(outputs, targets)
            final_loss = process_output(final_loss, final_outputs, final_targets, loss, outputs, targets)

        final_outputs = torch.cat(final_outputs)
        final_targets = torch.cat(final_targets)

        macro_recall_score = macro_recall(final_outputs, final_targets)

    return final_loss / counter, macro_recall_score


def extract(d, device):
    image = d["image"]
    grapheme_root = d[GRAPHEME_ROOT]
    vowel_diacritic = d[VOWEL_DIACRITIC]
    consonant_diacritic = d[CONSONANT_DIACRITIC]

    image = image.to(device, dtype=torch.float)
    grapheme_root = grapheme_root.to(device, dtype=torch.long)
    vowel_diacritic = vowel_diacritic.to(device, dtype=torch.long)
    consonant_diacritic = consonant_diacritic.to(device, dtype=torch.long)

    return consonant_diacritic, grapheme_root, image, vowel_diacritic


def should_stop_training_early(val_score, model, early_stopping):
    return early_stopping(val_score, model).early_stop


def save_plots(train_scores, valid_data_scores, train_losses, valid_data_losses, plot_prefix):
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_scores, color='green', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_data_scores, color='blue', linestyle='-',
        label='validation accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{plot_prefix}_accuracy.png')

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_losses, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_data_losses, color='red', linestyle='-',
        label='validation loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{plot_prefix}_loss.png')


# plotting collected data
def plots(data_collector, base_model):
    for i in data_collector:
        print(f"plotting for valid fold:{i}")
        i_data = i['data']
        epochs = []
        train_losses = []
        train_scores = []
        valid_data_scores = []
        valid_data_losses = []
        for entry in i_data:
            epochs.append(entry['epoch'])
            train_losses.append(entry['train_loss'])
            train_scores.append(entry['train_score'])
            valid_data_losses.append(entry['valid_data_loss'])
            valid_data_scores.append(entry['valid_data_score'])

        save_plots(train_scores, valid_data_scores, train_losses, valid_data_losses, f'{base_model}_valid_fold_{i}')


def train_main(base_model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    model_switch = {RESNET: ResNet34, EFFNET: EfficientNetB3}
    model = model_switch[base_model](pretrained=True)
    model.to(device)

    folds = {0, 1, 2, 3, 4}
    top_level_data_collector = []
    for valid_fold in range(5):
        VALID_FOLDS = [valid_fold]
        TRAIN_FOLDS = list(folds - {valid_fold})

        train_dataset, train_loader, valid_dataset, valid_loader = get_train_data_loaders(TRAIN_FOLDS, VALID_FOLDS)

        early_stopping, optimizer, scheduler = setup_model(model)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        top_score_so_far = -1

        print("FOLD : ", VALID_FOLDS[0])
        data_collector = []
        for epoch in range(EPOCHS):
            train_loss, train_score = train(train_dataset, device, train_loader, model, optimizer)
            valid_data_loss, valid_data_score = evaluate(valid_dataset, device, valid_loader, model)
            if scheduler:
                scheduler.step(valid_data_score)

            # check and update top_score if valid_data_score is better
            if valid_data_score > top_score_so_far:
                top_score_so_far = valid_data_score
                torch.save(model.state_dict(), f"trained_models/{base_model}_fold{VALID_FOLDS[0]}.pth")

            print(f'train_loss: {train_loss:.5f} ' +
                  f'train_score: {train_score:.5f} ' +
                  f'valid_data_loss: {valid_data_loss:.5f} ' +
                  f'valid_data_score: {valid_data_score:.5f}')

            data_collector.append({"epoch": epoch, "train_loss": train_loss,
                                   "train_score": train_score, "valid_data_loss": valid_data_loss,
                                   "valid_data_score": valid_data_score})

            if should_stop_training_early(valid_data_score, model, early_stopping):
                print("Early stopping")
                break
        top_level_data_collector.append({"valid_fold": valid_fold, "data": data_collector})
    plots(top_level_data_collector, base_model)


def setup_model(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.3, verbose=True)
    early_stopping = EarlyStopping(patience=5, verbose=True)
    return early_stopping, optimizer, None


def get_train_data_loaders(TRAIN_FOLDS, VALID_FOLDS):
    train_dataset = TrainDataset(
        folds=TRAIN_FOLDS, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, mean=MODEL_MEAN, std=MODEL_STD
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=2
    )
    valid_dataset = TrainDataset(
        folds=VALID_FOLDS, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, mean=MODEL_MEAN, std=MODEL_STD
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=2
    )
    return train_dataset, train_loader, valid_dataset, valid_loader


def test(base_model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    model_switch = {RESNET: ResNet34, EFFNET: EfficientNetB3}
    model = model_switch[base_model](pretrained=False)
    final_g_pred = []
    final_v_pred = []
    final_c_pred = []
    final_img_ids = []
    for fold in range(5):
        model_path = f"trained_models/{base_model}_fold{fold}.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model: {model_path}")
        model.to(device)
        model.eval()

        c_pred, g_pred, img_ids_list, v_pred = predict_using_model(device, model)

        final_g_pred.append(g_pred)
        final_v_pred.append(v_pred)
        final_c_pred.append(c_pred)
        if fold == 0:
            final_img_ids.extend(img_ids_list)

    create_submission_csv(final_c_pred, final_g_pred, final_img_ids, final_v_pred)


def predict_using_model(device, model):
    g_pred, v_pred, c_pred = [], [], []
    img_ids_list = []
    for file_idx in range(4):
        parquet_path = "%stest_image_data_%s.parquet" % (INPUT_FOLDER, file_idx)
        df = pd.read_parquet(parquet_path)
        print(f"Loaded file: {parquet_path}")

        dataset = TestDataset(df=df,
                              img_height=IMG_HEIGHT,
                              img_width=IMG_WIDTH,
                              mean=MODEL_MEAN,
                              std=MODEL_STD)

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=TEST_BATCH_SIZE,
            shuffle=False,
            num_workers=2
        )

        for bi, d in enumerate(data_loader):
            image = d["image"]
            img_id = d["image_id"]
            image = image.to(device, dtype=torch.float)

            grapheme_pred, vowel_pred, consonant_pred = model(image)

            for ii, imid in enumerate(img_id):
                g_pred.append(grapheme_pred[ii].cpu().detach().numpy())
                v_pred.append(vowel_pred[ii].cpu().detach().numpy())
                c_pred.append(consonant_pred[ii].cpu().detach().numpy())
                img_ids_list.append(imid)
    return c_pred, g_pred, img_ids_list, v_pred


def create_submission_csv(final_c_pred, final_g_pred, final_img_ids, final_v_pred):
    final_g = np.argmax(np.mean(np.array(final_g_pred), axis=0), axis=1)
    final_v = np.argmax(np.mean(np.array(final_v_pred), axis=0), axis=1)
    final_c = np.argmax(np.mean(np.array(final_c_pred), axis=0), axis=1)
    predictions = []
    for ii, imid in enumerate(final_img_ids):
        predictions.append((f"{imid}_grapheme_root", final_g[ii]))
        predictions.append((f"{imid}_vowel_diacritic", final_v[ii]))
        predictions.append((f"{imid}_consonant_diacritic", final_c[ii]))
    submission = pd.DataFrame(predictions, columns=["row_id", "target"])
    delete_if_exists("submission.csv")
    submission.to_csv("submission.csv", index=False)


def load_as_npa(file):
    print(f'loading file {file}')
    df = pd.read_parquet(file)
    print(df.head(10))
    return df.iloc[:, 0], df.iloc[:, 1:].values.reshape(-1, IMG_HEIGHT, IMG_WIDTH)


def image_from_char(char):
    image = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT))
    draw = ImageDraw.Draw(image)
    myfont = ImageFont.truetype(f'{INPUT_FOLDER}font.ttf', 120)
    w, h = draw.textsize(char, font=myfont)
    draw.text(((IMG_WIDTH - w) / 2, (IMG_HEIGHT - h) / 2), char, font=myfont)

    return image


def sample_train_images():
    image_ids0, images0 = load_as_npa(f'{INPUT_FOLDER}train_image_data_0.parquet')
    f, ax = plt.subplots(5, 5, figsize=(16, 8))
    ax = ax.flatten()

    for i in range(25):
        ax[i].imshow(images0[i], cmap='Greys')


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Operation to perform')
    my_parser.add_argument('-op',
                           type=str,
                           help="Operation to perform", required=True,
                           choices=['preprocess', 'train', 'test', "sample_images"])
    my_parser.add_argument('-m',
                           type=str,
                           help="base model to use", required=False, choices=[RESNET, EFFNET])
    args = my_parser.parse_args()
    if args.op == 'preprocess':
        preprocess_input_data()
    elif args.op == 'train':
        train_main(args.m)
    elif args.op == 'test':
        test(args.m)
    else:
        sample_train_images()
    print("DONE!!!")
