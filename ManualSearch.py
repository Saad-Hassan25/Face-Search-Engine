import os
import shutil
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from deepface import DeepFace

# Only import drive if running on Colab
try:
    from google.colab import drive
    IS_COLAB = True
except ImportError:
    IS_COLAB = False
    print("Not running on Google Colab. Drive mount will be skipped.")

# --- CONFIGURATION ---
# Path to your folder on Google Drive containing the images
DRIVE_FOLDER_PATH = "/content/"
# Path to your reference image (the photo of YOU)
REFERENCE_IMG_PATH = "/content/"
# Where to save the results list (CSV)
OUTPUT_CSV = "/content/"
# Temporary local path for faster processing
LOCAL_DB_PATH = "/content/temp_db"
# ---------------------

def main():
    # 1. Mount Google Drive
    if IS_COLAB:
        drive.mount('/content/drive')

    # 2. Copy images to local runtime (Speed Optimization)
    print("Step 1: Copying images to local Colab runtime for speed (It may take some initial time...")
    
    if not os.path.exists(LOCAL_DB_PATH):
        os.makedirs(LOCAL_DB_PATH)
        # Copy folder contents. 
        if os.path.exists(DRIVE_FOLDER_PATH):
            print(f"   Copying from {DRIVE_FOLDER_PATH}...")
            os.system(f'cp -r "{DRIVE_FOLDER_PATH}/." "{LOCAL_DB_PATH}"')
            print("Copy complete. Ready to search.")
        else:
            print(f"Error: Drive folder not found at {DRIVE_FOLDER_PATH}")
            return
    else:
        print("Images already copied. Skipping copy step.")

    # 3. Run Search
    print(f" Step 2: Starting search using Facenet512...")
    print("   Note: This creates/reads a database index (.pkl) file.")

    try:
        results = DeepFace.find(
            img_path=REFERENCE_IMG_PATH,
            db_path=LOCAL_DB_PATH,
            model_name='Facenet512',
            distance_metric='cosine',
            detector_backend='retinaface', # 'retinaface' is accurate but slow. Use 'opencv' for speed.
            enforce_detection=False,
            align=True
        )
    except Exception as e:
        print(f"❌ Error during search: {e}")
        results = []

    # 4. Display Results
    if results and len(results) > 0 and not results[0].empty:
        df = results[0]
        print(f"Success! Found {len(df)} images matching you.")

        # Save results
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"List of found images saved to: {OUTPUT_CSV}")

        # Display the top 12 matches
        print("\n--- Preview of First 12 Matches ---")
        
        # Determine how many images to show (min of 5 or total found)
        num_to_show = min(5, len(df))
        
        for i, row in df.head(num_to_show).iterrows():
            img_path = row['identity']

            # Load and display image
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.figure(figsize=(3, 3))
                    plt.imshow(img_rgb)
                    plt.axis('off')
                    plt.title(f"Match {i+1}\nConf: {1-row['distance']:.2f}") # Approx confidence
                    plt.show()
                else:
                    print(f"Could not read image: {img_path}")
            except Exception as viz_err:
                print(f"Error displaying image: {viz_err}")
    else:
        print("No matches found. Try a clearer reference photo.")

if __name__ == "__main__":
    main()
