import capybara as cb
from docaligner import DocAligner

from docclassifier import DocClassifier

IPADDR = '192.168.0.179'  # Change this to your IP camera address

aligner = DocAligner()
classifier = DocClassifier()

# If you feel model performance is not good, you should find a better quality
# and update the registered images in `register` folder.


def _run(img):
    doc_info = aligner(img)
    if doc_info.has_doc_polygon:
        img = cb.draw_polygon(img, doc_info.doc_polygon, (0, 255, 0), 3)

    max_sim, max_score = classifier(doc_info.doc_flat_img)

    img = cb.draw_text(
        img, f'{max_sim} {max_score:.2f}',
        (30, 30), (0, 255, 0), 36
    )

    return img


demo = cb.WebDemo(IPADDR, pipelines=[_run])
demo.run()
