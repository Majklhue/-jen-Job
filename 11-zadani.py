# # Zadání pro listopad 2024
#
# Budeme pracovat v rámci gitu a tvořit feature branche [Atlasian guide](https://www.atlassian.com/git/tutorials/comparing-workflows). To znamená, že:
# 1. Celý domácí úkol bude odevzdán v jedné větvi Gitu.
# 2. Pro tuto větev bude vytvořen Pull request.
# 3. V rámci Pull requestu na GitHubu bude popsáno co je v řešení a i tam budeme diskutovat jednotlivé "opravy"
#
# Vše je tedy potřeba držet jako `*.py` soubor.
#
# Oproti minule budu rád, když notebooky budou využívat Markdown s hezkým formátováním.

# ## Dataset
#
# Podíváme se blíže na závity. Vytvořil jsem datovou sadu, kterou budeme sdílet. Je dostupná na [HuggingFace](https://huggingface.co/datasets/research-centre-rez/threaded-sockets) a je možné si jí stáhnout pomocí gitu ```git clone git@hf.co:datasets/research-centre-rez/threaded-sockets```

# Tuto cestu je potřeba upravit dle aktuálního umístění staženého datasetu. Zbytek notebooku by měl zůstat spustitelný beze změny
PATH_TO_DATASET = "/Users/gimli/cvr/data/zavity/trojan/checkerboard-with-video/"

import cv2
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import tempfile
import imageio.v3 as iio
import subprocess
import numpy as np

# +
CAMERA_MOVEMENT_START_FRAME_NO = 100  # číslo framu ve videu, kde se kamera už začala pohybovat (na začátku stojí)
FRAMES_IN_MEMORY = 1000  # Načtení tisíce framů do paměti (pokud by se nevešlo, je potřeba upravit)

# Read all frames into memory
vidcap = cv2.VideoCapture(os.path.join(PATH_TO_DATASET, "737_GX010021.MP4"))
# go to the part where video starts to move
vidcap.set(cv2.CAP_PROP_POS_FRAMES, 100)
success = True
frames = []
while success:
    success, frame = vidcap.read()
    if success:
        frames.append(frame)
    if len(frames) >= FRAMES_IN_MEMORY:
        break
# -

# ## 1. posuň výřez zadaný x a y, tak aby odpovídal lépe vstupnímu videu

# +
# Video contains a lot of unused "boundary". Here we try to find the crop of the relevant data:

# Find out the proper crop - these parameters should by adapted according to the real data
y1 = 500
y2 = 1800
x1 = 1300
x2 = 2600

plt.figure(figsize=(15, 4))
ax = plt.subplot(121)
ax.imshow(frames[120][y1:y2, x1:x2,0], cmap="gray")
ax = plt.subplot(122)
ax.imshow(frames[120])
ax.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], color="red")
plt.show()
# -

# Pro následující operaci budeme potřebovat 3rd party knihovnu zvanou Devernay. Ke stažení je [tady](https://www.ipol.im/pub/art/2017/216/). Postupuj dle návodu. Na konci instalace by mělo být možné zavolat v konzoli:
#
# ```devernay```
#
# S následujícím výstupem:
#
# ```
# devernay 1.0 (October 10, 2017)
# Copyright (c) 2016-2017 Rafael Grompone von Gioi and Gregory Randall
#
# usage: devernay <image> [-s S] [-h H] [-l L] [-w W] [-t T] [-p P] [-g G]
#
# image  PGM or ASC file formats are handled
# -s S   set the blur standard deviation (default S=0.0 -> no blurring)
# -h H   set high threshold (default H=0.0)
# -l L   set low threshold (default L=0.0)
# -w W   set line width in PDF and SVG to W (default W=1.3)
# -t T   write TXT output to file T
# -p P   write PDF output to file P
# -g G   write SVG output to file G
#
# examples: devernay image.pgm -p output.pdf
#           devernay image.pgm -t output.txt -p output.pdf -g output.svg
#           devernay image.pgm -p output.pdf -s 1.0 -l 5.0 -h 15.0 -w 0.5
# ```
#
# Následně bude možné pokračovat dalším kódem.
#
# Pokud se ti instalace nepodaří, můžeš v kódu níže nahradit devernayho implementaci knihovní funkcí [Canny](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)

# ## 2. Podívej se na kód níže a popiš co dělá
# - co dělá funkce ```cv2.threshold``` najdeš [zde](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)
# - Devernay algoritmus je varianta hranového detektoru, který dává subpixelovou přesnost (narozdíl od běžných hranových detektorů)
# - Detekci úhlů čar v rámci framu lze udělat i pomocí [Houghovi transformace](https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html)
#

frame_angles = []
hists = []
for frame in tqdm(frames):
    undistorted =  frame[y1:y2, x1:x2,0]

    otsu_threshold, _ = cv2.threshold(undistorted, 0, 255, cv2.THRESH_OTSU)
    
    with tempfile.NamedTemporaryFile(suffix=".pgm", delete=False) as tmpfile:
        filename = tmpfile.name
        iio.imwrite(tmpfile.name, undistorted)  # this must be a grayscale image
        process = subprocess.Popen(
            ["/Users/gimli/projects/scripts/devernay", tmpfile.name,
             "-t", "/dev/stdout",
             "-l", f"{otsu_threshold / 15}",
             "-h", f"{otsu_threshold / 3}",
             "-p", "/Users/gimli/sample.pdf",
             "-s", f"1",
             ], stdout=subprocess.PIPE)
        tmpfile.close()

    # Načtení souboru vytvořeného devernay skriptem a jeho konverze do standardní podoby výstupu hranového detektoru
    result = process.stdout.read().decode("utf-8")
    lines = result.split("\n")
    dev = []
    for line in lines:
        if line != "":
            x, y = line.split(' ')
            dev.append((float(x), float(y)))
    dev = np.array(dev)
    if len(dev) == 0:
        print("Something wrong happen")
    samples = 1000
    choice = np.random.randint(0, len(dev), samples)

    # Tuto část je dobré si namalovat na papír ;)
    xx0 = np.matmul(dev[choice, 0].reshape(-1, 1), np.ones((1, len(choice))))
    yy0 = np.matmul(dev[choice, 1].reshape(-1, 1), np.ones((1, len(choice))))
    xx1 = np.matmul(np.ones((len(choice), 1)), dev[choice, 0].reshape(1, -1))
    yy1 = np.matmul(np.ones((len(choice), 1)), dev[choice, 1].reshape(1, -1))
    valid = np.zeros_like(xx0, dtype=bool)
    valid[xx0 != xx1] = 1
    angles = np.zeros_like(xx0, np.float32)
    angles[xx0 == xx1] = np.pi / 2
    angles[valid] = np.arctan((yy0[valid] - yy1[valid]) / (xx0[valid] - xx1[valid])).reshape(-1)
    angles[np.eye(samples, dtype=bool)] = np.nan
    
    filtered = np.rad2deg(np.abs(angles[~np.isnan(angles)]))
    
    if filename is not None and os.path.exists(filename):
        os.remove(filename)
    
    counts, values = np.histogram(filtered, bins=samples)
    hists.append(filtered)
    frame_angles.append(values[np.argmax(counts)])

# Nyní mám v poli `frame_angles` nějaké úhly. Pokusíme se je využít k tomu, aby se výstupní video neotáčelo, ale závit se pouze posouval.

# Jak vypadají úhly naměřené pro jeden frame si lze snadno zobrazit pomocí histogramů výše
fid = 101  # číslo framu
counts, values = np.histogram(hists[fid], bins=samples)
best = values[np.argmax(counts)]
plt.hist(hists[fid], bins=samples)
plt.axvline(best, color="red")
plt.title(best)
plt.show()

# Jak vypadají úhly pro celou sekvenci?
plt.figure(figsize=(15,3))
plt.plot(frame_angles)
plt.title("Natočení závitu")
plt.xlabel("frame number")
plt.xlabel("angle (°)")
plt.show()

# ## 3. Narovnání křivky
#
# Graf ukazuje, že úhel natočení závitu ve videu se lineárně mění, ale na grafu jsou podezřelá dvě místa, kde se jakoby otáčí směr rotace. To ale ve skutečnosti není pravda. Snímky se stále otáčejí stejným směrem a konstantní rychlostí. 
#
# - Jak je tedy možné mít takovýto graf?
# - Dokážeš s tím něco udělat?
# - Urči úhel o který se závit otočí mezi dvěma po sobě jdoucími snímky

# ## 4. Kompenzuj rotaci přes celé video
#
# Cílem je mít video, kde se závit posouvá. Koukni se v datasetu na `compensated.mp4`


