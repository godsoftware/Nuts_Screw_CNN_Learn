import cv2
import numpy as np
import os
import shutil

# Dosya pathlerini ayarla
path_nut = r"C:/python/InternshipTask/nutscrew1/Images/nuts"
path_screw = r"C:/python/InternshipTask/nutscrew1/Images/screws"
path_output = r"C:/python/InternshipTask/nutscrew1"

# Kırpılmış resimler için geçici klasörler
temp_nut_path = os.path.join(path_output, "temp_nut")
temp_screw_path = os.path.join(path_output, "temp_screw")

# Nihai kırpılmış resimler için klasörler
output_nut_path = os.path.join(path_output, "cropnut")
output_screw_path = os.path.join(path_output, "cropscrew")

# Klasörleri boşalt ve oluştur
def clear_and_create_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

clear_and_create_folder(temp_nut_path)
clear_and_create_folder(temp_screw_path)
clear_and_create_folder(output_nut_path)
clear_and_create_folder(output_screw_path)

# Görüntü kırpma fonksiyonu
def crop_images(input_path, temp_path, prefix):
    image_files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
    
    for index, image_file in enumerate(image_files):
        img_path = os.path.join(input_path, image_file)
        img = cv2.imread(img_path)
        
        if img is None:
            continue

        height, width, _ = img.shape
        
        # Kenarlardan kırpma oranı
        crop_percent = 0.1  # %10 oranında kenarlardan kırp
        
        # Kırpma boyutlarını hesapla
        x_crop = int(width * crop_percent)
        y_crop = int(height * crop_percent)
        
        # Resmi kırp
        cropped_img = img[y_crop:height-y_crop, x_crop:width-x_crop]
        
        crop_file_name = f"{prefix}{index+1}.jpg"
        crop_file_path = os.path.join(temp_path, crop_file_name)
        
        cv2.imwrite(crop_file_path, cropped_img)

# Görüntü işleme ve kırpma fonksiyonu
def process_images(input_path, temp_path, output_path, prefix):
    image_files = [f for f in os.listdir(temp_path) if os.path.isfile(os.path.join(temp_path, f))]
    
    for index, image_file in enumerate(image_files):
        temp_img_path = os.path.join(temp_path, image_file)
        original_img_path = os.path.join(input_path, image_file)
        img = cv2.imread(temp_img_path)
        original_img = cv2.imread(original_img_path)
        
        if img is None or original_img is None:
            continue

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Alt ve üst sınırları belirle (Yeşil renk aralığı için ayarlama)
        lower = np.array([37, 151, 89])
        upper = np.array([45, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)
        _, thres = cv2.threshold(mask, 75, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((5, 5), np.uint8)
        
        # Konturları bul
        contours, _ = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        crop_index = 1
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 800 < area < 50000:  # Alanı belirli bir aralıkta olan konturları seç
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(original_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cropped_img = img[y:y+h, x:x+w]
                
                crop_file_name = f"{prefix}{index+1}_{crop_index}.jpg"
                crop_file_path = os.path.join(output_path, crop_file_name)
                
                cv2.imwrite(crop_file_path, cropped_img)
                crop_index += 1
        
        # Orijinal resimdeki konturları göster
        cv2.namedWindow(f'Original {prefix}{index+1}', cv2.WINDOW_NORMAL)
        cv2.imshow(f'Original {prefix}{index+1}', original_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Nuts resimlerini kırp ve geçici klasöre kaydet
crop_images(path_nut, temp_nut_path, "n")

# Screws resimlerini kırp ve geçici klasöre kaydet
crop_images(path_screw, temp_screw_path, "s")

# Nuts geçici klasöründeki resimleri işleyip nihai klasöre kaydet
process_images(path_nut, temp_nut_path, output_nut_path, "n")

# Screws geçici klasöründeki resimleri işleyip nihai klasöre kaydet
process_images(path_screw, temp_screw_path, output_screw_path, "s")

print("Görüntü kırpma ve işleme işlemleri tamamlandı.")
