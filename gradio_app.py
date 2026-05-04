import os
import sys
import time
from typing import List, Dict, Any

try:
    import torch
    from torchvision import models, transforms
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    import networkx as nx
    from ultralytics import YOLO
    import gradio as gr
except Exception as e:
    raise ImportError(f"Missing dependency: {e}")

ROOT = os.path.abspath(os.path.dirname(__file__))


def get_train_classes() -> List[str]:
    train_dir = os.path.join(ROOT, "data", "train")
    if os.path.isdir(train_dir):
        classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        if classes:
            return classes
    fallback = ["bartels_exterior", "bergami_exterior", "dodds_exterior", "john_and_leona_exterior", "kaplan_exterior", "maxy_exterior"]
    return fallback


_YOLO = None
_RESNET = None
_RESNET_CLASSES = get_train_classes()


def parse_location(class_name: str):
    parts = class_name.split("_")
    building_map = {
        "dodds": "Robert B. Dodds Hall",
        "kaplan": "Kaplan Hall",
        "library": "Marvin K. Peterson Library",
        "bergami": "Bergami Learning Center",
    }
    building = building_map.get(parts[0], parts[0].title())
    floor, room = None, None
    if len(parts) > 1 and parts[1].startswith("floor"):
        floor = "Floor " + parts[1].replace("floor", "")
        if len(parts) > 2:
            room = " ".join(parts[2:]).replace("_", " ").title()
    elif len(parts) > 1:
        room = " ".join(parts[1:]).replace("_", " ").title()
    return building, floor, room


def build_default_graph(classes: List[str]) -> nx.Graph:
    G = nx.Graph()
    for c in classes:
        G.add_node(c)
    for a, b in zip(classes, classes[1:]):
        G.add_edge(a, b, weight=60)
    if len(classes) > 3:
        G.add_edge(classes[0], classes[min(2, len(classes)-1)], weight=90)
    return G


# default navigation graph (editable by user)
G = build_default_graph(_RESNET_CLASSES)


def get_graph_text() -> str:
    lines = []
    for a, b, d in G.edges(data=True):
        w = d.get("weight", 60)
        lines.append(f"{a},{b},{w}")
    return "\n".join(lines)


def update_graph_from_text(text: str):
    """Parse lines of 'node1,node2,weight' and rebuild G. Returns updated text and status."""
    global G
    newG = nx.Graph()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for ln in lines:
        parts = [p.strip() for p in ln.split(",") if p.strip()]
        if len(parts) >= 2:
            a, b = parts[0], parts[1]
            try:
                w = float(parts[2]) if len(parts) >= 3 else 60.0
            except Exception:
                w = 60.0
            newG.add_node(a)
            newG.add_node(b)
            newG.add_edge(a, b, weight=w)
    if newG.number_of_nodes() == 0:
        return get_graph_text(), "Graph not changed: no valid edges found.", gr.Dropdown.update(choices=list(G.nodes()), value=None)
    G = newG
    status = f"Saved graph — {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    return get_graph_text(), status, gr.Dropdown.update(choices=list(G.nodes()), value=(list(G.nodes())[0] if G.nodes() else None))


def load_yolo(model_path: str = None):
    global _YOLO
    if _YOLO is not None:
        return _YOLO
    model_path = model_path or os.path.join(ROOT, "yolov8s.pt")
    if not os.path.exists(model_path):
        _YOLO = None
        return None
    _YOLO = YOLO(model_path)
    return _YOLO


def load_resnet(weights_path: str = None, device: str = "cpu"):
    global _RESNET
    if _RESNET is not None:
        return _RESNET
    num_classes = max(1, len(_RESNET_CLASSES))
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    if weights_path is None:
        candidate = os.path.join(ROOT, "resnet50_campus_improved.pth")
        if os.path.exists(candidate):
            weights_path = candidate
    if weights_path and os.path.exists(weights_path):
        sd = torch.load(weights_path, map_location=device)
        try:
            model.load_state_dict(sd)
        except Exception:
            model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    _RESNET = model
    return _RESNET


def preprocess_resnet(pil_img: Image.Image, device: str = "cpu"):
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf(pil_img).unsqueeze(0).to(device)


def run_inference(pil_img: Image.Image, use_yolo=True, use_resnet=True, conf_thresh: float = 0.25, device: str = "cpu") -> (Image.Image, List[Dict[str, Any]]):
    yolo = load_yolo()
    resnet = None
    if use_resnet:
        resnet = load_resnet(device=device)

    if pil_img is None:
        return None, []
    img = pil_img.convert("RGB")
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    detections: List[Dict[str, Any]] = []

    if use_yolo and yolo is not None:
        results = yolo(np.array(img))
        if len(results) > 0:
            r = results[0]
            boxes = getattr(r, "boxes", None)
            if boxes is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls_idxs = boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), conf, cid in zip(xyxy, confs, cls_idxs):
                    if conf < conf_thresh:
                        continue
                    label_yolo = str(cid)
                    box = [int(x1), int(y1), int(x2), int(y2)]
                    draw.rectangle(box, outline="red", width=3)
                    draw.text((box[0] + 3, box[1] + 3), f"{label_yolo} {conf:.2f}", fill="yellow")
                    resnet_label = ""
                    resnet_conf = 0.0
                    if resnet is not None:
                        crop = img.crop(box).convert("RGB")
                        try:
                            inp = preprocess_resnet(crop, device=device)
                            with torch.no_grad():
                                out = resnet(inp)
                                probs = torch.nn.functional.softmax(out, dim=1)[0].cpu().numpy()
                                idx = int(probs.argmax())
                                resnet_label = _RESNET_CLASSES[idx] if idx < len(_RESNET_CLASSES) else str(idx)
                                resnet_conf = float(probs.max())
                        except Exception:
                            resnet_label = "error"
                    detections.append({"yolo_class": label_yolo, "yolo_conf": float(conf), "resnet_class": resnet_label, "resnet_conf": float(resnet_conf), "box": box})
    else:
        if use_resnet and resnet is not None:
            try:
                inp = preprocess_resnet(img, device=device)
                with torch.no_grad():
                    out = resnet(inp)
                    probs = torch.nn.functional.softmax(out, dim=1)[0].cpu().numpy()
                    idx = int(probs.argmax())
                    detections.append({"resnet_class": _RESNET_CLASSES[idx] if idx < len(_RESNET_CLASSES) else str(idx), "resnet_conf": float(probs.max())})
            except Exception:
                detections.append({"resnet_class": "error", "resnet_conf": 0.0})

    return annotated, detections


def classify_full_image(pil_img: Image.Image, device: str = "cpu") -> (str, float):
    """Return predicted class and confidence for the full image using ResNet."""
    resnet = load_resnet(device=device)
    if pil_img is None or resnet is None:
        return None, 0.0
    img = pil_img.convert("RGB")
    try:
        inp = preprocess_resnet(img, device=device)
        with torch.no_grad():
            out = resnet(inp)
            probs = torch.nn.functional.softmax(out, dim=1)[0].cpu()
            idx = int(probs.argmax())
            location = _RESNET_CLASSES[idx] if idx < len(_RESNET_CLASSES) else str(idx)
            confidence = float(probs.max())
            return location, confidence
    except Exception:
        return None, 0.0


def navigate_from_image(pil_img: Image.Image, destination: str, conf_thresh: float = 0.25, class_thresh: float = 0.8, device: str = "cpu") -> (Image.Image, str, List[Dict[str, Any]]):
    """Runs full navigation: resnet classification (location), yolo detection, then compute shortest path to destination.
    Returns (annotated_image, navigation_markdown, detections_list)."""
    if pil_img is None:
        return None, "No image provided", []

    # run resnet-only classification on full image
    img = pil_img.convert("RGB")
    location, confidence = classify_full_image(pil_img, device=device)

    # run YOLO for signs and get annotated image
    annotated, detections = run_inference(pil_img, use_yolo=True, use_resnet=False, conf_thresh=conf_thresh, device=device)

    # build navigation instructions
    nav_md_lines = []
    if location is None:
        nav_md_lines.append("**Predicted location:** Unknown (classification failed)")
    else:
        nav_md_lines.append(f"**Predicted location:** {location}  — {confidence:.0%}")
    nav_md_lines.append("")

    if confidence < class_thresh:
        nav_md_lines.append(f"Classification confidence below threshold ({class_thresh:.0%}). Navigation not attempted.")
        return annotated, "\n".join(nav_md_lines), detections
    nav_md_lines.append("")

    if location not in G:
        nav_md_lines.append("Predicted location not present in navigation graph.")
        nav_text = "\n".join(nav_md_lines)
        return annotated, nav_text, detections

    if destination not in G:
        nav_md_lines.append("Destination not present in navigation graph.")
        nav_text = "\n".join(nav_md_lines)
        return annotated, nav_text, detections

    try:
        path = nx.shortest_path(G, location, destination, weight="weight")
        secs = nx.shortest_path_length(G, location, destination, weight="weight")
        nav_md_lines.append(f"**Walking route** — {secs//60}m {secs%60}s")
        for i, node in enumerate(path, 1):
            b2, f2, r2 = parse_location(node)
            label = b2 + (f" · {f2}" if f2 else "") + (f" · {r2}" if r2 else "")
            mark = "► Start" if i == 1 else ("✓ Destination" if i == len(path) else f"Step {i}")
            nav_md_lines.append(f"- {mark}: {label}")
    except Exception as e:
        nav_md_lines.append(f"Navigation error: {e}")

    nav_text = "\n".join(nav_md_lines)
    return annotated, nav_text, detections


def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## Campus Navigation — Detection & Classification Demo")
        with gr.Row():
            inp = gr.Image(type="pil", label="Upload Image")
            with gr.Column():
                yolo_cb = gr.Checkbox(label="Run YOLO detection", value=True)
                resnet_cb = gr.Checkbox(label="Run ResNet classification", value=True)
                conf = gr.Slider(0.0, 1.0, value=0.25, label="YOLO confidence threshold")
                class_conf = gr.Slider(0.0, 1.0, value=0.8, label="Classification min confidence")
                device = gr.Dropdown(choices=["cpu", "cuda"], value="cpu", label="Device")
                run_btn = gr.Button("Run Detection / Classification")
                dest_dd = gr.Dropdown(choices=_RESNET_CLASSES, value=(_RESNET_CLASSES[-1] if _RESNET_CLASSES else None), label="Destination (for navigation)")
                nav_btn = gr.Button("Navigate")
                # Graph editor
                graph_txt = gr.Textbox(value=get_graph_text(), lines=8, label="Navigation graph (one edge per line: node1,node2,weight)")
                save_graph = gr.Button("Save Graph")
        with gr.Column(scale=2):
            out_img = gr.Image(label="Annotated image")
            pred_md = gr.Markdown("", label="Prediction")
            out_nav = gr.Markdown(label="Navigation")
        with gr.Column(scale=1):
            out_json = gr.JSON(label="Detections / Classifications")
            out_graph_status = gr.Markdown(label="Graph status")

        def infer(image, run_yolo, run_resnet, conf_thresh, class_conf_thresh, device_choice):
            if image is None:
                return None, {}, "No image provided", ""
            annotated, dets = run_inference(image, use_yolo=run_yolo, use_resnet=run_resnet, conf_thresh=conf_thresh, device=device_choice)
            # full-image classification
            loc, confv = classify_full_image(image, device=device_choice)
            if loc is None:
                nav_text = "Classification failed"
                pred_text = "**Prediction:** Unknown"
            else:
                b, f, r = parse_location(loc)
                pretty = b + (f" · {f}" if f else "") + (f" · {r}" if r else "")
                pred_text = f"### {pretty}  \n**Class:** {loc} — {confv:.0%}"
                if confv >= class_conf_thresh:
                    nav_text = f"**Predicted location:** {loc} ({pretty}) — {confv:.0%}"
                else:
                    nav_text = f"Predicted location: {loc} — confidence {confv:.0%} (below threshold {class_conf_thresh:.0%})"
            return annotated, dets, nav_text, pred_text

        def navigate_ui(image, destination, conf_thresh, class_conf_thresh, device_choice):
            if image is None:
                return None, "No image provided", {}, ""
            annotated, nav_text, dets = navigate_from_image(image, destination, conf_thresh=conf_thresh, class_thresh=class_conf_thresh, device=device_choice)
            # also compute a prediction display
            loc, confv = classify_full_image(image, device=device_choice)
            if loc:
                b, f, r = parse_location(loc)
                pretty = b + (f" · {f}" if f else "") + (f" · {r}" if r else "")
                pred_text = f"### {pretty}  \n**Class:** {loc} — {confv:.0%}"
            else:
                pred_text = ""
            return annotated, nav_text, dets, pred_text

        run_btn.click(infer, inputs=[inp, yolo_cb, resnet_cb, conf, class_conf, device], outputs=[out_img, out_json, out_nav, pred_md])
        # allow running navigation separately
        nav_btn.click(navigate_ui, inputs=[inp, dest_dd, conf, class_conf, device], outputs=[out_img, out_nav, out_json, pred_md])
        # auto-run infer when image is uploaded
        inp.upload(infer, inputs=[inp, yolo_cb, resnet_cb, conf, class_conf, device], outputs=[out_img, out_json, out_nav, pred_md])
        save_graph.click(update_graph_from_text, inputs=[graph_txt], outputs=[graph_txt, out_graph_status, dest_dd])
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", share=False)
