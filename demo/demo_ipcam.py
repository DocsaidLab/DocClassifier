import docsaidkit as D
from docaligner import DocAligner
from docclassifier import DocClassifier

IPADDR = '192.168.0.179'  # Change this to your IP camera address

aligner = DocAligner()
classifier = DocClassifier()


def _run(img):
    doc_info = aligner(img)
    if doc_info.has_doc_polygon:
        img = D.draw_polygon(img, doc_info.doc_polygon, (0, 255, 0), 3)

    max_sim, max_score = classifier(doc_info.doc_flat_img)

    img = D.draw_text(
        img, f'{max_sim} {max_score:.2f}',
        (30, 30), (0, 255, 0), 36
    )

    return img


demo = D.WebDemo(IPADDR, pipelines=[_run])
demo.run()
