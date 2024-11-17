# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# Define folder you will work with
PHOTOS_DIR_PATH = "C:\\Users\\Michal\\Pictures\\TOPPHOTOS\\"
TOP_SECRET_DIR_PATH = "C:\\Users\\Michal\\Pictures\\WRONG TURN\\TOP SECRET !"

# +
import cv2
import matplotlib.pyplot as plt
# library handful for path operations
import os

# Use the folder path instead of copy & paste full path
image = cv2.imread(os.path.join(PHOTOS_DIR_PATH, "MAJKL a KEVIN.png"))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Zobrazení obrázku v notebooku
# It is good to have here figure creation (otherwise, run as a script will rewrite the output)
plt.figure()
plt.imshow(image_rgb)
plt.axis('off')
plt.show()

# +
# it is not necessary to import library in each cell, once per file is sufficient
cap = cv2.VideoCapture(os.path.join(TOP_SECRET_DIR_PATH, "Záznam obrazovky 2024-08-18 230910.mp4"))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Počet snímků ve videu: {frame_count}")

# You want to use the same "cap" in the next cell - so do not release it here
# -

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS videa: {fps}")

# This approach to color format is wrong, because OpenCV VideoCapture converts video automatically into BGR.
# If you want the raw format, you must set CAP_PROP_CONVERT_RGB property to False.
# Please do.
ret, frame = cap.read()
if ret:
    channels = frame.shape[2] if len(frame.shape) > 2 else 1
    print(f"Počet barevných kanálů: {channels}")
    
    if channels == 3:
        print("Barevný formát: BGR")
    elif channels == 1:
        print("Barevný formát: Grayscale")
else:
    print("Nepodařilo se načíst video.")

# +
ret, frame = cap.read()

if ret:
    frame_size = frame.nbytes
    print(f"Velikost snímku v paměti: {frame_size} bajtů")
else:
    print("Nepodařilo se načíst snímek.")

# +
# Here is a little bit tricky part:
# Because you have already read one frame, calling cap.read() reads the next one.
# If you would like to seek in the video use cap.set(CAP_PROP_POS_FRAMES, target_frame_number)

ret, frame = cap.read()

if ret:
    blue_channel = frame[:, :, 0]
    green_channel = frame[:, :, 1]
    red_channel = frame[:, :, 2]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(blue_channel, cmap='Blues')
    plt.title("Modrý kanál")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(green_channel, cmap='Greens')
    plt.title("Zelený kanál")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(red_channel, cmap='Reds')
    plt.title("Červený kanál")
    plt.axis('off')

    plt.show()

else:
    print("Nepodařilo se načíst snímek.")

# +
# Chybí odpověď na otázku, kolik framů se vejde do paměti.

# +
from skimage import io
import numpy as np

image = io.imread(os.path.join(PHOTOS_DIR_PATH, "MAJKL a KEVIN.png"))

print("Typ obrázku:", type(image))
print("Rozměry obrázku:", image.shape)

# +
# se dá řešit třemi způsoby ( po snímcích, Uvolnění paměti po zpracování každého snímku, Zpracování videa v dávkách)

# +
fps = cap.get(cv2.CAP_PROP_FPS)  
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  

print(f"Počet snímků: {frame_count}")
print(f"FPS: {fps}")
print(f"Rozlišení: {width}x{height}")

# Zpracování videa po snímcích
frame_number = 0
cap.set(CAP_PROP_POS_FRAMES, target_frame_number)
while True:
    ret, frame = cap.read()  # Načte jeden snímek
    if not ret:
        break  

    frame_number += 1
    
    cv2.imshow("Frame", frame)

    # Uvolnění paměti pro snímek
    del frame  # This is not necessary to do, because in the loop you rewrite the same space in the memory

    # Stisknu 'q' ukončím zobrazení
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Uvolnění zdrojů
cap.release()
cv2.destroyAllWindows()

# +
# Vybral jsem metodu po snímcích)

# +
# Image is still the same -> No need to load it again.

height, width, _ = image.shape  #rozměry
y1, y2 = height // 4, 3 * height // 2
x1, x2 = width // 4, 3 * width // 2

cropped_image = image[y1:y2, x1:x2]
cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
ax[0].imshow(image_rgb)
ax[0].set_title("Původní obrázek")
ax[0].axis('off')

ax[1].imshow(cropped_image_rgb)
ax[1].set_title("Oříznutý obrázek")
ax[1].axis('off')

plt.show()

# +
# This cell just copy the previous one?!

# +
#horizontálně
flipped_image = cv2.flip(image, 1)
# what about image[:,::-1,:]
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
flipped_image_rgb = cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Původní obrázek
ax[0].imshow(image_rgb)
ax[0].set_title("Původní obrázek")
ax[0].axis('off')

# Zrcadlený obrázek
ax[1].imshow(flipped_image_rgb)
ax[1].set_title("Zrcadlený obrázek")
ax[1].axis('off')

plt.show()

# +
#potom to samý akorát upravím

# +
#vertikálně
flipped_image_vertical = cv2.flip(image, 0)
# How the flip, by the numpy array syntax will look like?

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
flipped_image_vertical_rgb = cv2.cvtColor(flipped_image_vertical, cv2.COLOR_BGR2RGB)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Původní obrázek
ax[0].imshow(image_rgb)
ax[0].set_title("Původní obrázek")
ax[0].axis('off')

# Vertikálně zrcadlený obrázek
ax[1].imshow(flipped_image_vertical_rgb)
ax[1].set_title("Vertikálně zrcadlený obrázek")
ax[1].axis('off')

plt.show()

# +
rotated_image = np.rot90(image, -1)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
rotated_image_rgb = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(image_rgb)
ax[0].set_title("Původní obrázek")
ax[0].axis('off')

ax[1].imshow(rotated_image_rgb)
ax[1].set_title("Otočený obrázek o 90°")
ax[1].axis('off')

plt.show()

# +
# OpenCV načítá obrázky ve formátu BGR, ale matplotlib zobrazuje RGB, Nejprve převedeš obrázek na RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

fig, ax = plt.subplots(1, 4, figsize=(12, 4))

ax[0].imshow(image_rgb)
ax[0].set_title("Původní obrázek (RGB)")
ax[0].axis('off')

# It is problematic to show HSV values, but this can be done this way:
# 1. Take Hue from HSV image
# 2. Add 100% saturation (maximum "color")
# 3. Add high value, but not 100% (because we want to see the hue)
# 4. From these we creat new HSV image, which then we convert to RGB and visualize
ax[1].imshow(cv2.cvtColor(
    np.stack([image_hsv[:, :, 0], 
              np.ones(image_hsv.shape[:2]) * 255, 
              np.ones(image_hsv.shape[:2]) * 200], axis=2).astype(np.uint8), cv2.COLOR_HSV2RGB))
ax[1].set_title("Hue i.e. odstín")
ax[1].axis('off')

# For saturation and value, this is simple, because it do not carry "color" information:
ax[2].imshow(image_hsv[:, :, 1], cmap='gray')
ax[2].set_title("Saturation")
ax[2].axis('off')
ax[3].imshow(image_hsv[:, :, 2], cmap='gray')
ax[3].set_title("Value i.e. intenzita/jas")
ax[3].axis('off')

plt.show()
# -

# # Odpovědi na na otázky z 6. části

# 1) Otáčení pomocí matice (práce s poli), Otáčení pomocí geometrie (např. OpenCV) a hlavní rozdíl je, že matice má limitovaný úhly pod kterýma se může otáčet, zatím co u geometrie můžu mít libovolný úhel

# 2) otáčení v rotační matici se projeví posunem výsledného obrázku (otáčím obrázek kolem bodu)

# 3) změní se a to tak, že se obrázek zvětší, protože rohy obdélníku se posunou mimo původní hranice. + dá se to vypočítat Sinus Cosinus :D

# 4) jak jsem řekl při otáčení hrozí že se rohy posunou mimo původní hranice, takže bych zvětšil plátno

# <span style="color:red">Fajn, tak si to pojďme vyzkoušet:
# 1. vytvoř rotační matici, která obrázek otočí kolem svého středu o 35°.
# 2. aplikuj rotační matici na obrázek, vyzkoušej různé parametry ořezu
# 3. Zkus změnit při tvorbě matice pozici středu otáčení.
#
# </span>


