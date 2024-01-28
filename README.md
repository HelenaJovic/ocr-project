
## OCR for Handwritten Text Analysis

## Overview

This project focuses on developing an Optical Character Recognition (OCR) system to recognize and interpret handwritten texts. It is particularly tailored for processing images containing handwritten characters and converting them into machine-encoded text. The project utilizes Python along with several libraries like OpenCV, Pandas, Scikit-learn, and TensorFlow/Keras for image processing, data handling, machine learning, and deep learning functionalities.

## Key Features

- **Handwritten Text Recognition:** Ability to process images and extract handwritten text.
- **Character Recognition:** Utilizes neural networks to recognize individual characters.
- **Distance Calculation:** Implements Hemming distance calculation for accuracy assessment.
- **Image Processing:** Utilizes OpenCV for pre-processing of images to enhance OCR accuracy.
- **K-Means Clustering:** Applies K-Means for space detection between words or characters.

## Dependencies

- Python 3.x
- OpenCV (cv2)
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow/Keras

## Installation

Ensure that you have Python installed on your system. You can install the required libraries using pip:

```
pip install numpy pandas opencv-python matplotlib scikit-learn tensorflow keras
```

## Usage

To use this OCR system, place your images in the 'pictures' folder and run the Python script:

```
python Optical Character Recognition (OCR) Project.py
```

The script will process the images and output the recognized text along with the calculated Hemming distances.

## File Structure

- `SC23-G6-RA-43-2020.py`: Main Python script for OCR processing.
- `pictures/`: Directory containing images for OCR.

## How It Works

1. **Image Preprocessing:** Images are converted to grayscale and binarized for further processing.
2. **Region of Interest (ROI) Selection:** Contours are detected, and regions of interest are extracted as individual characters.
3. **Character Recognition:** Extracted characters are fed into a neural network trained to recognize handwritten characters.
4. **Post-Processing:** The output text is formatted, and Hemming distances are calculated to assess the accuracy.

## Contributing

Contributions to this project are welcome. Feel free to fork the repository, make improvements, and submit a pull request.

## License

This project is open-source and available under the [MIT License](LICENSE).

## Contact

For any queries or suggestions, please contact:

- Helena Jovic - [h.jovix@gmail.com](mailto:h.jovix@gmail.com)

---

This readme provides a comprehensive guide to understanding and utilizing the OCR system. Feel free to customize it further according to your project's specifics and requirements.
