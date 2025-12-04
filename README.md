# 🔍 Face Search AI

This repository contains tools to perform **AI-based facial recognition** on large datasets of images.  
It allows you to find all photos of a specific person (e.g., yourself) within a gallery of thousands of images.

The project uses **DeepFace** with the **Facenet512** model and **RetinaFace** detector for high accuracy.

---

## 📂 Repository Structure

### **manual_search.py**
A script designed for **Google Colab**.  
It connects to Google Drive, copies images to the local runtime for speed, and performs the face search.

### **app.py**
A script designed for **Hugging Face Spaces**.  
It creates a **Gradio web interface** where users can upload a selfie and instantly find matches from a pre-uploaded dataset.

---

## 🚀 Part 1: Running Manually (Google Colab)

This script is ideal for processing your dataset for the first time and generating the `.pkl` index files.

### **Prerequisites**
- A Google Drive folder containing your images  
- A reference photo of yourself  

### **How to Use**
1. Open **Google Colab**  
2. Paste the contents of `manual_search_colab.py` into a cell  
3. Update the **CONFIGURATION** section at the top:

```python
DRIVE_FOLDER_PATH = "/content/drive/MyDrive/part2"  # Your images
REFERENCE_IMG_PATH = "/content/saad.jpg"            # Your selfie
```
Run the script.

> **Note:**  
> The first run will be slow because it has to detect faces in all images.

### **Result**
It will generate a `.pkl` file (e.g., `ds_model_facenet512...pkl`) inside your image folder.  
**Do not delete this file** — it is required for the deployment step.

---

# 🌐 Part 2: Deployment (Hugging Face Spaces)

This script creates a public (or private) web app to search your gallery instantly.

## **Prerequisites**
- **Dataset ZIPs:** Zip your image folders  
- Ensure the `.pkl` files generated in Part 1 are included inside the zips  
  - Example: `part1.zip`, `part2.zip`

## **How to Deploy**
1. Create a new **Space** on Hugging Face using the **Gradio SDK**  
2. Upload:
   - `app.py`
   - `requirements.txt`  
3. Upload your ZIP files (`part1.zip`, `part2.zip`) to the Space's **Files** tab  

The app will unzip the images on startup and use the `.pkl` files to enable instant searching.

---

# 📦 Requirements (requirements.txt)
   - deepface
   - tf-keras
   - opencv-python-headless
   - pandas
   - matplotlib
---

# 🛠️ Technology Stack

- **DeepFace** — facial recognition pipeline  
- **Facenet512** — high-accuracy face recognition model  
- **RetinaFace** — state-of-the-art face detection (works even at difficult angles)  
- **Gradio** — web interface  
- **OpenCV** — image processing    

