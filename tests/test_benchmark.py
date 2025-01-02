import capybara as cb

from docclassifier import DocClassifier

DIR = cb.get_curdir(__file__)

classifier = DocClassifier(
    register_root='/home/shayne/workspace/DocClassifier/data/private/RegisterCard',
    model_cfg='20240326'
)

fs = cb.get_files(DIR.parent / 'benchmark' /
                  'jpn_driver_color_benchmark_dataset', suffix=['.jpg'])
for f in fs:
    img = cb.imread(f)
    print(f' 定義為：{f.parent.name}，預測為：{classifier(img)[0]}')
