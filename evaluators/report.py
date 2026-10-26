import base64
import html
from io import BytesIO
from pathlib import Path

def export_badcase_html(
    bad_cases,
    dataset,
    output_path,
    annotations=None
):
    annotations = annotations or {}
    output_path = Path(output_path)

    # 建立 sample_id 到错误记录的映射
    case_map = {}
    for case_type, records in bad_cases.items():
        for record in records:
            case_map[str(record["sample_id"])] = (case_type, record)

    cards = []

    for i in range(len(dataset)):
        sample = dataset[i]
        key = str(sample.sample_id)

        if key not in case_map:
            continue

        case_type, record = case_map[key]

        annotation = annotations.get(key, {})

        labels = annotation.get("labels", [])
        note = annotation.get("note", "")

        labels_html = ", ".join(
            html.escape(label) for label in labels
        ) or "Unannotated"

        # PIL 图片转 base64
        buffer = BytesIO()
        sample.image.convert("RGB").save(buffer, format="JPEG")
        image_data = base64.b64encode(buffer.getvalue()).decode()

        cards.append(f"""
        <article>
            <h2>{html.escape(case_type)} · #{html.escape(key)}</h2>

            <img src="data:image/jpeg;base64,{image_data}">

            <p><b>Question:</b> {html.escape(sample.question)}</p>
            <p><b>Ground Truth:</b> {html.escape(record["ground_truth"])}</p>
            <p><b>Prediction:</b> {html.escape(str(record["prediction"]))}</p>

            <p><b>Error Labels:</b> {labels_html}</p>
            <p><b>Analysis:</b> {html.escape(note)}</p>

            <pre>{html.escape(record["raw_answer"])}</pre>
        </article>
        """)

    page = """
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial; max-width: 900px; margin: 40px auto; }
            article { border: 1px solid #ddd; padding: 20px; margin-bottom: 24px; }
            img { max-width: 450px; display: block; }
            pre { white-space: pre-wrap; overflow-wrap: anywhere; }
        </style>
    </head>
    <body>
        <h1>MultiEval — Bad Case Report</h1>
    """ + "\n".join(cards) + "</body></html>"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")