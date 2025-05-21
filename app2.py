import streamlit as st
import yaml
import os
import shutil
import tempfile
import zipfile
import io
import json
from collections import defaultdict

# For Data Augmentation option
import cv2
import numpy as np
import albumentations as A

st.set_page_config(page_title="Multi-Tool Dataset Utility", layout="wide")

# ========= Helper Functions =========
def get_unique_filename(directory, filename):
    """Generate a unique filename in a directory."""
    base, ext = os.path.splitext(filename)
    candidate = filename
    counter = 1
    while os.path.exists(os.path.join(directory, candidate)):
        candidate = f"{base}_{counter}{ext}"
        counter += 1
    return candidate

# ========= Option 1: Label Counter =========
def label_counter(yaml_file, label_files):
    output_lines = []
    try:
        yaml_content = yaml_file.read().decode("utf-8")
        data = yaml.safe_load(yaml_content)
        class_names = data["names"]
        num_classes = len(class_names)
    except Exception as e:
        st.error("Error reading YAML file. Please check that it is valid.")
        return

    class_counts = defaultdict(int)
    for uploaded_file in label_files:
        try:
            content = uploaded_file.read().decode("utf-8")
            lines = content.splitlines()
        except Exception as e:
            st.error(f"Error reading file {uploaded_file.name}.")
            continue

        for line in lines:
            parts = line.strip().split()
            if parts:
                try:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                except:
                    continue

    output_lines.append("📊 **Class Label Count Summary:**")
    for class_id, count in sorted(class_counts.items()):
        if class_id < num_classes:
            class_name = class_names[class_id]
        else:
            class_name = f"Unknown Class {class_id}"
        output_lines.append(f"- **{class_name} ({class_id})**: {count} labels")

    unused_classes = [name for i, name in enumerate(class_names) if i not in class_counts]
    if unused_classes:
        output_lines.append("\n⚠️ **Warning:** The following classes have zero labels in your dataset:")
        for name in unused_classes:
            output_lines.append(f"❌ {name}")

    output_lines.append("\n✅ Analysis Complete!")
    st.markdown("\n".join(output_lines))

# ========= Option 2: Extract Specific Labels =========
def extract_specific_labels(image_files, label_files, target_count, target_class_id):
    allowed_ext = [".jpg", ".jpeg", ".png"]
    images_dict = {img.name: img for img in image_files}
    labels_dict = {lab.name: lab for lab in label_files}

    with tempfile.TemporaryDirectory() as tmpdirname:
        output_images_dir = os.path.join(tmpdirname, "images")
        output_labels_dir = os.path.join(tmpdirname, "labels")
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)

        extracted_count = 0
        for label_filename, label_file in labels_dict.items():
            if not label_filename.endswith(".txt"):
                continue
            try:
                content = label_file.read().decode("utf-8")
            except Exception as e:
                st.error(f"Error reading {label_filename}")
                continue
            lines = content.splitlines()

            target_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 1:
                    continue
                try:
                    cls = int(parts[0])
                except:
                    continue
                if cls == target_class_id:
                    target_lines.append(line)

            label_file.seek(0)
            if target_lines and extracted_count < target_count:
                base_filename = os.path.splitext(label_filename)[0]
                image_found = None
                for ext in allowed_ext:
                    candidate = base_filename + ext
                    if candidate in images_dict:
                        image_found = images_dict[candidate]
                        break
                if not image_found:
                    st.warning(f"No image found for {label_filename}. Skipping.")
                    continue

                unique_image_name = get_unique_filename(output_images_dir, image_found.name)
                image_path = os.path.join(output_images_dir, unique_image_name)
                with open(image_path, "wb") as f:
                    f.write(image_found.read())
                image_found.seek(0)

                unique_label_name = get_unique_filename(output_labels_dir, label_filename)
                label_out_path = os.path.join(output_labels_dir, unique_label_name)
                with open(label_out_path, "w") as f:
                    f.write("\n".join(target_lines) + "\n")

                extracted_count += 1
                st.write(f"Extracted **{base_filename}** with {len(target_lines)} target label(s). Total extracted: {extracted_count}")

            if extracted_count >= target_count:
                break

        st.success("✅ Extraction complete!")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for foldername, subfolders, filenames in os.walk(tmpdirname):
                for filename in filenames:
                    file_path = os.path.join(foldername, filename)
                    arcname = os.path.relpath(file_path, tmpdirname)
                    zipf.write(file_path, arcname)
        zip_buffer.seek(0)
        zip_data = zip_buffer.getvalue()
        st.session_state['extraction_zip'] = zip_data

        st.download_button(
            label="Download Extracted Dataset (ZIP)",
            data=zip_data,
            file_name="extracted_dataset.zip",
            mime="application/zip"
        )

        st.markdown("### Save ZIP to a Specific Folder")
        folder_path = st.text_input("Enter the local folder path to save the ZIP file (optional):", key="folder_path")
        if st.button("Save ZIP to Folder"):
            if folder_path:
                if not os.path.exists(folder_path):
                    try:
                        os.makedirs(folder_path)
                    except Exception as e:
                        st.error(f"Failed to create folder: {e}")
                        return
                zip_file_path = os.path.join(folder_path, "extracted_dataset.zip")
                try:
                    with open(zip_file_path, "wb") as f:
                        f.write(zip_data)
                    st.success(f"ZIP file saved to {zip_file_path}")
                except Exception as e:
                    st.error(f"Failed to save ZIP file: {e}")
            else:
                st.error("Please provide a folder path.")

# ========= Option 3: Update Class Labels =========
def update_class_labels(label_files, old_mapping_str, new_mapping_str):
    try:
        old_class_mapping = json.loads(old_mapping_str)
        new_class_mapping = json.loads(new_mapping_str)
    except Exception as e:
        st.error(f"Error parsing mapping JSON: {e}")
        return

    old_inverted = {v: k for k, v in old_class_mapping.items()}
    with tempfile.TemporaryDirectory() as tmpdirname:
        for uploaded_file in label_files:
            if not uploaded_file.name.endswith(".txt"):
                continue
            try:
                content = uploaded_file.read().decode("utf-8")
            except Exception as e:
                st.error(f"Error reading {uploaded_file.name}")
                continue
            lines = content.splitlines()
            updated_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    old_class_id = int(parts[0])
                except:
                    continue
                class_name = old_inverted.get(old_class_id)
                if class_name and class_name in new_class_mapping:
                    new_class_id = new_class_mapping[class_name]
                    parts[0] = str(new_class_id)
                    updated_lines.append(" ".join(parts))
            out_filename = get_unique_filename(tmpdirname, uploaded_file.name)
            out_path = os.path.join(tmpdirname, out_filename)
            with open(out_path, "w") as f:
                f.write("\n".join(updated_lines) + "\n")
            st.write(f"Updated **{uploaded_file.name}**")
        
        st.success("✅ All label files updated successfully!")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for foldername, subfolders, filenames in os.walk(tmpdirname):
                for filename in filenames:
                    file_path = os.path.join(foldername, filename)
                    arcname = os.path.relpath(file_path, tmpdirname)
                    zipf.write(file_path, arcname)
        zip_buffer.seek(0)
        st.download_button(
            label="Download Updated Label Files (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="updated_labels.zip",
            mime="application/zip"
        )

# ========= Option 4: Data Augmentation =========
def data_augmentation(image_files, label_files, aug_types, aug_count):
    # Build a dictionary for label files keyed by the base filename (without extension)
    labels_dict = {os.path.splitext(f.name)[0]: f for f in label_files}

    with tempfile.TemporaryDirectory() as tmpdirname:
        output_images_dir = os.path.join(tmpdirname, "aug_images")
        output_labels_dir = os.path.join(tmpdirname, "aug_labels")
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)

        # This function returns an Albumentations transformation based on aug_type.
        # For "Random Crop", it computes a crop size (80% of original) using the input image.
        def get_transform(aug_type, img):
            if aug_type == "Horizontal Flip":
                return A.Compose([A.HorizontalFlip(p=1.0)],
                                 bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
            elif aug_type == "Vertical Flip":
                return A.Compose([A.VerticalFlip(p=1.0)],
                                 bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
            elif aug_type == "Random Brightness Contrast":
                return A.Compose([A.RandomBrightnessContrast(p=1.0)],
                                 bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
            elif aug_type == "Rotate 90":
                return A.Compose([A.Rotate(limit=(90, 90), p=1.0)],
                                 bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
            elif aug_type == "Gaussian Blur":
                return A.Compose([A.GaussianBlur(blur_limit=(3,7), p=1.0)],
                                 bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
            elif aug_type == "ShiftScaleRotate":
                return A.Compose([A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=1.0)],
                                 bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
            elif aug_type == "CLAHE":
                return A.Compose([A.CLAHE(p=1.0)],
                                 bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
            elif aug_type == "Hue Saturation Value":
                return A.Compose([A.HueSaturationValue(p=1.0)],
                                 bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
            elif aug_type == "Random Gamma":
                return A.Compose([A.RandomGamma(p=1.0)],
                                 bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
            elif aug_type == "Elastic Transform":
                return A.Compose([A.ElasticTransform(p=1.0)],
                                 bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
            elif aug_type == "Grid Distortion":
                return A.Compose([A.GridDistortion(p=1.0)],
                                 bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
            elif aug_type == "Optical Distortion":
                return A.Compose([A.OpticalDistortion(p=1.0)],
                                 bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
            elif aug_type == "Coarse Dropout":
                return A.Compose([A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=1.0)],
                                 bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
            elif aug_type == "Cutout":
                return A.Compose([A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, p=1.0)],
                                 bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
            elif aug_type == "Random Crop":
                h, w = img.shape[:2]
                new_h = int(h * 0.8)
                new_w = int(w * 0.8)
                return A.Compose([A.RandomSizedBBoxSafeCrop(height=new_h, width=new_w, p=1.0)],
                                 bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
            else:
                return None

        # Process each image file
        for image_file in image_files:
            base_name = os.path.splitext(image_file.name)[0]
            if base_name not in labels_dict:
                st.warning(f"Label file for {image_file.name} not found. Skipping.")
                continue

            file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_file.seek(0)

            label_file = labels_dict[base_name]
            try:
                label_content = label_file.read().decode("utf-8")
            except Exception as e:
                st.error(f"Error reading label file {label_file.name}")
                continue
            lines = label_content.splitlines()
            bboxes = []
            category_ids = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    cls = int(parts[0])
                    bbox = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                    bboxes.append(bbox)
                    category_ids.append(cls)
                except:
                    continue

            # For each selected augmentation type, generate aug_count copies
            for aug_type in aug_types:
                transform = get_transform(aug_type, img)
                if transform is None:
                    st.error(f"Augmentation {aug_type} not recognized.")
                    continue
                for i in range(aug_count):
                    augmented = transform(image=img, bboxes=bboxes, category_ids=category_ids)
                    aug_image = augmented["image"]
                    aug_bboxes = augmented["bboxes"]
                    aug_category_ids = augmented["category_ids"]

                    new_image_name = f"{base_name}_{aug_type.replace(' ', '_')}_aug_{i+1}.jpg"
                    new_label_name = f"{base_name}_{aug_type.replace(' ', '_')}_aug_{i+1}.txt"
                    image_path = os.path.join(output_images_dir, new_image_name)
                    label_path = os.path.join(output_labels_dir, new_label_name)

                    cv2.imwrite(image_path, aug_image)
                    with open(label_path, "w") as f:
                        for cid, bbox in zip(aug_category_ids, aug_bboxes):
                            f.write(f"{cid} {' '.join(str(x) for x in bbox)}\n")
                    st.write(f"Augmented: {new_image_name}")

        st.success("✅ Data Augmentation complete!")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for foldername, subfolders, filenames in os.walk(tmpdirname):
                for filename in filenames:
                    file_path = os.path.join(foldername, filename)
                    arcname = os.path.relpath(file_path, tmpdirname)
                    zipf.write(file_path, arcname)
        zip_buffer.seek(0)
        zip_data = zip_buffer.getvalue()
        st.session_state['aug_zip'] = zip_data

        st.download_button(
            label="Download Augmented Dataset (ZIP)",
            data=zip_data,
            file_name="augmented_dataset.zip",
            mime="application/zip"
        )

        st.markdown("### Save Augmented ZIP to a Specific Folder")
        folder_path = st.text_input("Enter the local folder path to save the ZIP file (optional):", key="aug_folder_path")
        if st.button("Save Augmented ZIP to Folder"):
            if folder_path:
                if not os.path.exists(folder_path):
                    try:
                        os.makedirs(folder_path)
                    except Exception as e:
                        st.error(f"Failed to create folder: {e}")
                        return
                zip_file_path = os.path.join(folder_path, "augmented_dataset.zip")
                try:
                    with open(zip_file_path, "wb") as f:
                        f.write(zip_data)
                    st.success(f"ZIP file saved to {zip_file_path}")
                except Exception as e:
                    st.error(f"Failed to save ZIP file: {e}")
            else:
                st.error("Please provide a folder path.")

# ========= Main UI =========
st.title("Multi-Tool Dataset Utility")
st.markdown("Select an option below to perform a specific dataset operation.")

option = st.sidebar.radio(
    "Choose an Operation",
    ("Label Counter", "Extract Specific Labels", "Update Class Labels", "Data Augmentation")
)

if option == "Label Counter":
    st.header("Label Counter")
    st.markdown("Upload your **data.yaml** file and one or more label **.txt** files.")
    yaml_file = st.file_uploader("Upload YAML File", type=["yaml", "yml"])
    label_files = st.file_uploader("Upload Label Files (.txt)", type=["txt"], accept_multiple_files=True)
    if st.button("Run Label Counter"):
        if not yaml_file:
            st.error("Please upload a YAML file.")
        elif not label_files:
            st.error("Please upload at least one label file.")
        else:
            label_counter(yaml_file, label_files)

elif option == "Extract Specific Labels":
    st.header("Extract Specific Labels")
    st.markdown("""
    Upload your dataset files.  
    - **Label Files:** Upload one or more label (.txt) files.  
    - **Image Files:** Upload corresponding image files (.jpg, .jpeg, .png).  
    Set the **Target Count** (max number of images/labels to extract) and the **Target Class ID**.
    """)
    label_files = st.file_uploader("Upload Label Files (.txt)", type=["txt"], accept_multiple_files=True, key="extract_labels")
    image_files = st.file_uploader("Upload Image Files", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="extract_images")
    target_count = st.number_input("Target Count", min_value=1, value=140, step=1)
    target_class_id = st.number_input("Target Class ID", min_value=0, value=0, step=1)
    if st.button("Run Extraction"):
        if not label_files:
            st.error("Please upload label files.")
        elif not image_files:
            st.error("Please upload image files.")
        else:
            extract_specific_labels(image_files, label_files, target_count, target_class_id)

elif option == "Update Class Labels":
    st.header("Update Class Labels")
    st.markdown("""
    Upload your label (.txt) files that need to be updated.  
    Then provide the **Old Class Mapping** and the **New Class Mapping** as JSON key/value pairs.
    """)
    label_files = st.file_uploader("Upload Label Files (.txt)", type=["txt"], accept_multiple_files=True, key="update_labels")
    st.markdown("#### Old Class Mapping (JSON)")
    default_old = json.dumps({
        "BHL1": 0,
        "BHR1": 1,
        "RH_10": 2,
        "RH_11": 3,
        "RH_12": 4,
        "RH_2": 5,
        "RH_3": 6,
        "RH_4": 7,
        "RH_5": 8,
        "RH_8": 9,
        "RH_9": 10,
        "Rh_6": 11,
        "unknown": 12
    }, indent=4)
    old_mapping_str = st.text_area("Enter Old Class Mapping (JSON)", value=default_old, height=250)
    st.markdown("#### New Class Mapping (JSON)")
    default_new = json.dumps({
        "BHL1": 9,
        "BHR1": 10,
        "RH_10": 11,
        "RH_11": 12,
        "RH_12": 13,
        "RH_2": 14,
        "RH_3": 15,
        "RH_4": 16,
        "RH_5": 17,
        "RH_8": 18,
        "RH_9": 19,
        "Rh_6": 20,
        "unknown": 21
    }, indent=4)
    new_mapping_str = st.text_area("Enter New Class Mapping (JSON)", value=default_new, height=250)
    if st.button("Update Labels"):
        if not label_files:
            st.error("Please upload at least one label file.")
        else:
            update_class_labels(label_files, old_mapping_str, new_mapping_str)

elif option == "Data Augmentation":
    st.header("Data Augmentation")
    st.markdown("""
    Upload your dataset for augmentation:  
    - **Image Files:** Upload your image files (e.g. .jpg, .png).  
    - **Label Files:** Upload corresponding label files (.txt in YOLO format).  
    Then select one or more augmentations from the dropdown and specify how many augmented copies to generate per image.
    """)
    image_files = st.file_uploader("Upload Image Files", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="aug_images")
    label_files = st.file_uploader("Upload Label Files", type=["txt"], accept_multiple_files=True, key="aug_labels")
    aug_types = st.multiselect(
        "Select Augmentation Types",
        options=[
            "Horizontal Flip",
            "Vertical Flip",
            "Random Brightness Contrast",
            "Rotate 90",
            "Gaussian Blur",
            "ShiftScaleRotate",
            "CLAHE",
            "Hue Saturation Value",
            "Random Gamma",
            "Elastic Transform",
            "Grid Distortion",
            "Optical Distortion",
            "Coarse Dropout",
            "Cutout",
            "Random Crop"
        ],
        default=["Horizontal Flip", "Random Brightness Contrast"]
    )
    aug_count = st.number_input("Number of Augmented Copies per Image", min_value=1, value=1, step=1)
    if st.button("Run Data Augmentation"):
        if not image_files:
            st.error("Please upload image files.")
        elif not label_files:
            st.error("Please upload label files.")
        elif not aug_types:
            st.error("Please select at least one augmentation type.")
        else:
            data_augmentation(image_files, label_files, aug_types, aug_count)
