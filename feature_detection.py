import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

IMG1 = "resources/truck1.png"
IMG2 = "resources/truck2.png"

img1 = cv.imread(IMG1, cv.IMREAD_GRAYSCALE)
img2 = cv.imread(IMG2, cv.IMREAD_GRAYSCALE)


# --------------------- DETECTOR FUNCTIONS ---------------------

def run_sift(image):
    start = time.time()
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    out = cv.drawKeypoints(image, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return out, len(kp), time.time() - start


def run_orb(image):
    start = time.time()
    orb = cv.ORB_create(nfeatures=500)
    kp, des = orb.detectAndCompute(image, None)
    out = cv.drawKeypoints(image, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return out, len(kp), time.time() - start


def run_fast(image):
    start = time.time()
    fast = cv.FastFeatureDetector_create()
    kp = fast.detect(image, None)
    out = cv.drawKeypoints(image, kp, None)
    return out, len(kp), time.time() - start


def run_brief(image):
    start = time.time()
    star = cv.xfeatures2d.StarDetector_create()
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    kp = star.detect(image, None)
    kp, des = brief.compute(image, kp)
    out = cv.drawKeypoints(image, kp, None)
    return out, len(kp), time.time() - start


def run_harris(image):
    start = time.time()
    h = cv.cornerHarris(image, 2, 3, 0.04)
    out = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    out[h > 0.01 * h.max()] = [0, 0, 255]
    # Harris doesn't give kp count so approximate:
    count = np.sum(h > 0.01 * h.max())
    return out, count, time.time() - start


def run_shi_tomasi(image):
    start = time.time()
    corners = cv.goodFeaturesToTrack(image, 200, 0.01, 10)
    out = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    for c in corners:
        x, y = c.ravel()
        cv.circle(out, (int(x), int(y)), 4, (0,255,0), -1)
    return out, len(corners), time.time() - start


# --------------------- BUILD COMPARISON GRID ---------------------

results = [
    ("SIFT", *run_sift(img1)),
    ("ORB", *run_orb(img1)),
    ("FAST", *run_fast(img1)),
    ("BRIEF", *run_brief(img1)),
    ("Harris", *run_harris(img1)),
    ("Shi–Tomasi", *run_shi_tomasi(img1))
]

plt.figure(figsize=(12,8))
for i, (name, img_out, count, t) in enumerate(results, 1):
    plt.subplot(2,3,i)
    plt.title(name)
    plt.imshow(img_out, cmap="gray")
    plt.axis("off")

plt.tight_layout()
plt.savefig("comparison_grid.png")
plt.close()


# --------------------- PERFORMANCE TABLE ---------------------

names = [r[0] for r in results]
keypoints = [r[2] for r in results]
times = [round(r[3]*1000, 2) for r in results]  # in ms

plt.figure(figsize=(8,5))
plt.title("Performance Comparison")
plt.bar(names, keypoints)
for i, v in enumerate(keypoints):
    plt.text(i, v+5, str(v), ha='center')
plt.ylabel("Number of Keypoints")
plt.savefig("performance.png")
plt.close()


# --------------------- FEATURE MATCHING ---------------------

def match_and_save(detector, name):
    if detector == "SIFT":
        d = cv.SIFT_create()
    else:
        d = cv.ORB_create()

    kp1, des1 = d.detectAndCompute(img1, None)
    kp2, des2 = d.detectAndCompute(img2, None)

    bf = cv.BFMatcher(cv.NORM_L2 if detector=="SIFT" else cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:30]

    matched_img = cv.drawMatches(img1, kp1, img2, kp2, matches, None)
    cv.imwrite(f"{name}.png", matched_img)

match_and_save("SIFT", "sift_matches")
match_and_save("ORB", "orb_matches")


# --------------------- PDF REPORT ---------------------

pdf = canvas.Canvas("feature_report.pdf", pagesize=letter)
pdf.setFont("Helvetica-Bold", 20)
pdf.drawString(180, 750, "Feature Detection Report")

pdf.setFont("Helvetica", 12)
pdf.drawString(50, 725, "Comparison of classical feature detectors and matching algorithms.")

pdf.drawImage("comparison_grid.png", 50, 380, width=500, height=320)
pdf.drawImage("performance.png", 150, 260, width=300, height=150)
pdf.drawImage("sift_matches.png", 50, 50, width=250, height=180)
pdf.drawImage("orb_matches.png", 320, 50, width=250, height=180)

pdf.save()

print("DONE! Report saved as feature_report.pdf")
