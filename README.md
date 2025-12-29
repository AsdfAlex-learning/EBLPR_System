# License Plate Recognition System

> **⚠️ Status: Under Construction / Refactoring**
>
> This project is currently undergoing a major refactoring and optimization process. The codebase is being updated to improve detection accuracy and robustness, particularly for double-row license plates.

## Overview

This project aims to implement a robust license plate recognition system. The current focus is on the **Plate Localization** module, which has been significantly enhanced to handle various lighting conditions and plate types.

## Core Logic: Plate Localization (`plate_detector.py`)

The plate localization algorithm has been redesigned with the following pipeline:

1.  **Center Crop**: The image is cropped to focus on the central region, reducing processing time and background noise.
2.  **Dual-Direction Sobel Edge Detection**:
    -   Combines **Sobel X** (vertical edges) and **Sobel Y** (horizontal edges) with equal weights.
    -   This ensures that both the vertical strokes of digits/letters and the horizontal strokes of Chinese characters are captured.
3.  **Pre-Dilation**:
    -   Applies a small (3x3) dilation to the binary edge image.
    -   This strengthens weak edges, ensuring they survive the subsequent morphological operations.
4.  **Morphological Closing**:
    -   Uses a small (3x3) kernel to connect broken plate borders.
    -   Avoids using large kernels that might merge the plate with surrounding background clutter.
5.  **Contour Analysis (RETR_TREE)**:
    -   Extracts contours using the `RETR_TREE` mode to detect nested structures.
    -   Leverages the hierarchy to find "parent" contours (plate borders) that contain "child" contours (characters).
6.  **Candidate Scoring**:
    -   Candidates are scored based on **Aspect Ratio (AR)** and **Distance from Center**.
    -   **High Weight on AR**: The system strongly prioritizes candidates with an AR close to **1.98** (standard for double-row plates).
7.  **Projection Refinement**:
    -   The detected region undergoes a projection analysis (horizontal and vertical) to precisely crop the plate boundaries, removing excess padding.

## Getting Started

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running Tests

To verify the plate localization logic:

```bash
python 01_Test_locate_plate.py
```

## TODO

-   [x] Refactor Plate Localization (Dual Sobel, AR Scoring)
-   [ ] Optimize Character Segmentation
-   [ ] Update Character Recognition Logic
-   [ ] End-to-end Pipeline Testing

---
*The rest of this documentation is intentionally left blank as the project evolves.*
