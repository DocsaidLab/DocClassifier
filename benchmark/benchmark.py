from itertools import combinations

import capybara as cb
import numpy as np
import prettytable
from sklearn.metrics import roc_curve

from docclassifier import DocClassifier

DIR = cb.get_curdir(__file__)

LABEL_MAPPING = {
    'Doc_A': 0,
    'Doc_B': 1,
    'Doc_C': 2,
    'Doc_D': 3,
    'Doc_E': 4,
    'Doc_F': 5,
}


def calc_combinations(norm_embeddings: np.ndarray, labels: np.ndarray):
    combinations_pair = np.array(
        list(combinations(range(norm_embeddings.shape[0]), 2)))
    base_inds, tgt_inds = combinations_pair[:, 0], combinations_pair[:, 1]
    combinations_scores = np.sum(
        norm_embeddings[base_inds] * norm_embeddings[tgt_inds], axis=-1)
    combinations_scores = (combinations_scores + 1) / 2
    combinations_labels = np.where(
        labels[base_inds] == labels[tgt_inds], 1, 0)
    return combinations_scores, combinations_labels


def export_tpr_fpr_table(comb_scores, comb_labels, index):

    fprs, tprs, ths = roc_curve(comb_labels, comb_scores)

    fpr_row = ["FPR", 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1]
    tpr_row = ["TPR", ]
    th_row = ["Threshold"]

    for x_label in fpr_row[1:]:
        min_index = np.argmin(np.abs(fprs - x_label))
        tpr_tmp = tprs[min_index].round(3)
        th_tmp = ths[min_index].round(5)
        tpr_row.append(tpr_tmp)
        th_row.append(th_tmp)

    tpr_fpr_table = prettytable.PrettyTable(header=False)
    tpr_fpr_table.add_row(fpr_row)
    tpr_fpr_table.add_row(tpr_row)
    tpr_fpr_table.add_row(th_row)

    tpr_fpr_table_txt_fpath = str(DIR / f"tpr_fpr_table_{index}.txt")
    with open(tpr_fpr_table_txt_fpath, 'w') as f:
        f.write(tpr_fpr_table.get_string())


def main():

    model = DocClassifier(model_cfg='20240326')
    ds = cb.load_json(DIR / 'real_cache.json')

    # Checking feature precision in different decimal places
    for i in range(1, 8):

        print(f"Round: {i}\n\n")

        comb_scores, comb_labels = [], []
        for batch in cb.Tqdm(cb.make_batch(ds, batch_size=1024), total=1 + len(ds) // 1024):
            feats, labels = [], []
            for data in batch:
                lb = data[0]
                img_path = DIR / 'docs_benchmark_dataset' / \
                    lb / cb.Path(data[1]).name
                img = cb.imread(img_path)
                feat = model.classifier.extract_feature(img)
                feat = feat.round(i)

                feats.append(feat)
                labels.append(LABEL_MAPPING[lb])

            feats = np.stack(feats)
            labels = np.array(labels)

            combinations_scores, combinations_labels = calc_combinations(
                feats, labels)

            comb_scores.extend(combinations_scores)
            comb_labels.extend(combinations_labels)

        comb_scores = np.stack(comb_scores, axis=0)
        comb_labels = np.stack(comb_labels, axis=0)
        export_tpr_fpr_table(comb_scores, comb_labels, i)


if __name__ == '__main__':
    main()
